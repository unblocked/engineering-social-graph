"""Reviewer recommendation engine.

Combines code expertise and social graph proximity to suggest the best
reviewers for a given file or directory. The algorithm ranks candidates by:

1. Expertise score — how well does this person know the changed code?
2. Social proximity — how closely do they collaborate with the PR author?
3. Review load balance — prefer reviewers who aren't already overrepresented

The final score is a weighted combination:
    score = 0.6 * expertise + 0.3 * proximity + 0.1 * (1 - load)
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from social_graph.decay import THREE_SIGMA_DAYS, days_since, exponential_time_decay
from social_graph.graph.expertise import experts_for_path
from social_graph.models import SocialEdge

# Weights for the composite reviewer score
WEIGHT_EXPERTISE = 0.6
WEIGHT_PROXIMITY = 0.3
WEIGHT_LOAD_BALANCE = 0.1


def _login_from_email(email: str) -> str:
    """Normalize an email to a login prefix, matching GitHub noreply format."""
    login = email.split("@")[0]
    if "+" in login:
        login = login.split("+", 1)[1]
    return login


def suggest_reviewers(
    repo_path: str | Path,
    changed_paths: list[str],
    edges: list[SocialEdge],
    author_login: str | None = None,
    top_n: int = 5,
    mode: str = "prs",
) -> list[dict]:
    """Suggest reviewers for a set of changed file paths.

    Args:
        repo_path: Path to the local git clone.
        changed_paths: List of file/directory paths that changed.
        edges: Social graph edges (from cache).
        author_login: The PR author's login (excluded from suggestions).
        top_n: Number of reviewers to return.
        mode: ``"prs"`` (default) uses PR edges for proximity;
              ``"commits"`` derives proximity from git log co-commits.

    Returns:
        List of dicts with keys: login, score, expertise, proximity, reason.
    """
    if mode not in ("prs", "commits"):
        raise ValueError(f"mode must be 'prs' or 'commits', got {mode!r}")

    # Build a mapping from email-prefix → GitHub login so that expertise
    # keys (email-based) can be unified with social-edge keys (GitHub logins).
    email_to_login: dict[str, str] = {}
    for edge in edges:
        for login in (edge.author.login, edge.reviewer.login):
            email_to_login.setdefault(login, login)

    # 1. Aggregate expertise across all changed paths
    expertise_scores: dict[str, float] = defaultdict(float)
    expertise_counts: dict[str, int] = defaultdict(int)

    for path in changed_paths:
        rankings = experts_for_path(str(repo_path), path, top_n=20)
        for r in rankings:
            # experts_for_path returns emails as login; normalize to prefix
            email_prefix = _login_from_email(r.user.login)
            # Map to GitHub login if known, otherwise keep the prefix
            login = email_to_login.get(email_prefix, email_prefix)
            expertise_scores[login] += r.score
            expertise_counts[login] += 1

    # Normalize expertise: average across paths, then normalize to [0, 1]
    avg_expertise: dict[str, float] = {}
    for login, total in expertise_scores.items():
        avg_expertise[login] = total / expertise_counts[login]

    max_exp = max(avg_expertise.values()) if avg_expertise else 1.0
    norm_expertise = {k: v / max_exp for k, v in avg_expertise.items()}

    # 2. Compute social proximity
    proximity: dict[str, float] = {}
    if mode == "commits":
        proximity = co_commit_proximity(repo_path, changed_paths, author_login)
    elif author_login:
        for edge in edges:
            if edge.author.login == author_login:
                proximity[edge.reviewer.login] = max(
                    proximity.get(edge.reviewer.login, 0), edge.weight
                )
            if edge.reviewer.login == author_login:
                proximity[edge.author.login] = max(
                    proximity.get(edge.author.login, 0), edge.weight * 0.5
                )

    # 3. Compute review load
    if mode == "commits":
        norm_load = {}  # no meaningful load signal from commits alone
    else:
        review_counts: dict[str, int] = defaultdict(int)
        for edge in edges:
            review_counts[edge.reviewer.login] += 1
        max_reviews = max(review_counts.values()) if review_counts else 1
        norm_load = {k: v / max_reviews for k, v in review_counts.items()}

    # 4. Combine scores for all candidates
    all_candidates = set(norm_expertise.keys()) | set(proximity.keys())
    if author_login:
        all_candidates.discard(author_login)

    results = []
    for login in all_candidates:
        exp = norm_expertise.get(login, 0)
        prox = proximity.get(login, 0)
        load = norm_load.get(login, 0)

        score = (
            WEIGHT_EXPERTISE * exp
            + WEIGHT_PROXIMITY * prox
            + WEIGHT_LOAD_BALANCE * (1 - load)
        )

        # Build a human-readable reason
        reasons = []
        if exp > 0.5:
            reasons.append("domain expert")
        elif exp > 0.2:
            reasons.append("knows the code")
        if prox > 0.5:
            reasons.append("close collaborator")
        elif prox > 0.2:
            reasons.append("has reviewed your work")
        if norm_load and load < 0.3:
            reasons.append("low review load")

        results.append({
            "login": login,
            "score": round(score, 3),
            "expertise": round(exp, 3),
            "proximity": round(prox, 3),
            "reason": ", ".join(reasons) if reasons else "relevant contributor",
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_n]


def co_commit_proximity(
    repo_path: str | Path,
    changed_paths: list[str],
    author_login: str | None = None,
) -> dict[str, float]:
    """Derive "closeness" from shared file commits.

    Runs ``git log`` on each changed path, collects all committers, scores by
    time-decayed commit frequency, and normalizes to [0, 1].

    Args:
        repo_path: Path to the local git clone.
        changed_paths: File paths to inspect.
        author_login: Login to exclude from results (the PR author).

    Returns:
        Mapping of login -> proximity score in [0, 1].
    """
    raw_scores: dict[str, float] = defaultdict(float)
    # Note: this uses the decay window (~336 days) which is intentionally wider
    # than the default --since-git ("6 months ago") to capture longer-term
    # co-commit relationships, with recency handled by time decay.
    since = f"{THREE_SIGMA_DAYS} days ago"

    for path in changed_paths:
        try:
            result = subprocess.run(
                [
                    "git", "log",
                    "--format=%aE\t%aI",
                    f"--since={since}",
                    "--", path,
                ],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            continue
        if result.returncode != 0 or not result.stdout.strip():
            continue

        for line in result.stdout.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            email, iso_date = parts

            login = _login_from_email(email)

            try:
                dt = datetime.fromisoformat(iso_date)
                decay = exponential_time_decay(days_since(dt))
            except ValueError:
                decay = 0.5
            raw_scores[login] += decay

    # Exclude author
    if author_login:
        raw_scores.pop(author_login, None)

    if not raw_scores:
        return {}

    # Normalize to [0, 1]
    max_score = max(raw_scores.values())
    if max_score == 0:
        return {}

    return {login: score / max_score for login, score in raw_scores.items()}
