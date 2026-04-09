"""Reviewer recommendation engine.

Combines code expertise and social graph proximity to suggest the best
reviewers for a given file or directory. The algorithm ranks candidates by:

1. Expertise score — how well does this person know the changed code?
2. Social proximity — how closely do they collaborate with the PR author?
3. Review load balance — prefer reviewers who are not already overrepresented

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

WEIGHT_EXPERTISE = 0.6
WEIGHT_PROXIMITY = 0.3
WEIGHT_LOAD_BALANCE = 0.1


def _login_from_email(email: str) -> str:
    """Normalize an email to a login prefix, matching GitHub noreply format."""
    login = email.split("@", 1)[0]
    if "+" in login:
        login = login.split("+", 1)[1]
    return login.lower()


def _normalize_login(value: str | None) -> str | None:
    """Normalize either a login or an email to the canonical lowercase login."""
    if not value:
        return None
    if "@" in value:
        return _login_from_email(value)
    return value.lower()


def suggest_reviewers(
    repo_path: str | Path,
    changed_paths: list[str],
    edges: list[SocialEdge],
    author_login: str | None = None,
    top_n: int = 5,
    mode: str = "prs",
) -> list[dict]:
    """Suggest reviewers for a set of changed file paths."""
    if mode not in ("prs", "commits"):
        raise ValueError(f"mode must be 'prs' or 'commits', got {mode!r}")

    normalized_author = _normalize_login(author_login)

    expertise_scores: dict[str, float] = defaultdict(float)
    expertise_counts: dict[str, int] = defaultdict(int)

    for path in changed_paths:
        rankings = experts_for_path(str(repo_path), path, top_n=20)
        for ranking in rankings:
            login = _normalize_login(ranking.user.login)
            if login is None:
                continue
            expertise_scores[login] += ranking.score
            expertise_counts[login] += 1

    avg_expertise = {
        login: total / expertise_counts[login]
        for login, total in expertise_scores.items()
    }
    max_exp = max(avg_expertise.values()) if avg_expertise else 1.0
    norm_expertise = {login: score / max_exp for login, score in avg_expertise.items()}

    proximity: dict[str, float] = {}
    if mode == "commits":
        raw_proximity = co_commit_proximity(repo_path, changed_paths, normalized_author)
        for login, score in raw_proximity.items():
            normalized_login = _normalize_login(login)
            if normalized_login is None:
                continue
            proximity[normalized_login] = max(proximity.get(normalized_login, 0.0), score)
    elif normalized_author:
        for edge in edges:
            author = _normalize_login(edge.author.login)
            reviewer = _normalize_login(edge.reviewer.login)
            if author == normalized_author and reviewer is not None:
                proximity[reviewer] = max(proximity.get(reviewer, 0.0), edge.weight)
            if reviewer == normalized_author and author is not None:
                proximity[author] = max(proximity.get(author, 0.0), edge.weight * 0.5)

    if mode == "commits":
        norm_load: dict[str, float] = {}
    else:
        review_counts: dict[str, int] = defaultdict(int)
        for edge in edges:
            reviewer = _normalize_login(edge.reviewer.login)
            if reviewer is not None:
                review_counts[reviewer] += 1
        max_reviews = max(review_counts.values()) if review_counts else 1
        norm_load = {login: count / max_reviews for login, count in review_counts.items()}

    all_candidates = set(norm_expertise) | set(proximity)
    if normalized_author:
        all_candidates.discard(normalized_author)

    results = []
    for login in all_candidates:
        exp = norm_expertise.get(login, 0.0)
        prox = proximity.get(login, 0.0)
        load = norm_load.get(login, 0.0)
        score = (
            WEIGHT_EXPERTISE * exp
            + WEIGHT_PROXIMITY * prox
            + WEIGHT_LOAD_BALANCE * (1 - load)
        )

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

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:top_n]


def co_commit_proximity(
    repo_path: str | Path,
    changed_paths: list[str],
    author_login: str | None = None,
) -> dict[str, float]:
    """Derive reviewer proximity from shared file commit history."""
    raw_scores: dict[str, float] = defaultdict(float)
    since = f"{THREE_SIGMA_DAYS} days ago"
    normalized_author = _normalize_login(author_login)

    for path in changed_paths:
        try:
            result = subprocess.run(
                [
                    "git",
                    "log",
                    "--format=%aE	%aI",
                    f"--since={since}",
                    "--",
                    path,
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
            parts = line.split("	", 1)
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

    if normalized_author:
        raw_scores.pop(normalized_author, None)

    if not raw_scores:
        return {}

    max_score = max(raw_scores.values())
    if max_score == 0:
        return {}

    return {login: score / max_score for login, score in raw_scores.items()}
