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

from collections import defaultdict
from pathlib import Path

from social_graph.graph.expertise import experts_for_path
from social_graph.models import SocialEdge

# Weights for the composite reviewer score
WEIGHT_EXPERTISE = 0.6
WEIGHT_PROXIMITY = 0.3
WEIGHT_LOAD_BALANCE = 0.1


def _login_from_email(email: str) -> str:
    """Extract a likely login from an email address.

    experts_for_path returns email addresses as the User.login field.
    Strip the domain and any noreply prefix (e.g. '12345+user@users.noreply.github.com').
    """
    local = email.split("@")[0]
    if "+" in local:
        local = local.split("+", 1)[1]
    return local


def suggest_reviewers(
    repo_path: str | Path,
    changed_paths: list[str],
    edges: list[SocialEdge],
    author_login: str | None = None,
    top_n: int = 5,
) -> list[dict]:
    """Suggest reviewers for a set of changed file paths.

    Args:
        repo_path: Path to the local git clone.
        changed_paths: List of file/directory paths that changed.
        edges: Social graph edges (from cache).
        author_login: The PR author's login (excluded from suggestions).
        top_n: Number of reviewers to return.

    Returns:
        List of dicts with keys: login, score, expertise, proximity, reason.
    """
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
    if max_exp == 0.0:
        max_exp = 1.0
    norm_expertise = {k: v / max_exp for k, v in avg_expertise.items()}

    # 2. Compute social proximity to the author
    proximity: dict[str, float] = {}
    if author_login:
        for edge in edges:
            if edge.author.login == author_login:
                proximity[edge.reviewer.login] = max(
                    proximity.get(edge.reviewer.login, 0), edge.weight
                )
            if edge.reviewer.login == author_login:
                proximity[edge.author.login] = max(
                    proximity.get(edge.author.login, 0), edge.weight * 0.5
                )

    # Normalize proximity to [0, 1] like the other signals
    if proximity:
        max_prox = max(proximity.values())
        if max_prox > 0:
            proximity = {k: v / max_prox for k, v in proximity.items()}

    # 3. Compute review load (how often someone appears as reviewer in edges)
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
        if load < 0.3:
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
