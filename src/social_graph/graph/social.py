"""Social/collaboration graph builder.

Builds a weighted directed graph of author->reviewer relationships from PR
review activity.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

import networkx as nx

from social_graph.decay import THREE_SIGMA_DAYS, days_since, exponential_time_decay
from social_graph.models import PullRequest, PullRequestState, SocialEdge, User
from social_graph.scoring import MINIMUM_SOCIAL_WEIGHT, score_pr_interactions


MIN_PRS_PER_MONTH = 0.0


def build_social_graph(
    pull_requests: list[PullRequest],
    now: datetime | None = None,
    min_pr_rate: float = MIN_PRS_PER_MONTH,
) -> list[SocialEdge]:
    """Build weighted social edges from PR interactions.

    Algorithm:
    1. For each author, iterate their closed PRs within the lookback window
    2. Per PR: score reviewers (log dampening * avg weight, normalize within PR)
    3. Multiply by Gaussian time decay based on PR's updated_at
    4. Accumulate per reviewer across all PRs
    5. Final normalization per author, discard edges < 0.1
    6. Filter out low-activity contributors by PR rate (PRs/month since first PR)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    since = now - timedelta(days=THREE_SIGMA_DAYS)

    # Compute PR rate (PRs/month) to filter low-activity contributors
    authored_count: dict[str, int] = defaultdict(int)
    first_pr: dict[str, datetime] = {}
    for pr in pull_requests:
        login = pr.author.login
        authored_count[login] += 1
        if login not in first_pr or pr.created_at < first_pr[login]:
            first_pr[login] = pr.created_at

    active_logins: set[str] = set()
    for login, count in authored_count.items():
        months_active = max(1.0, (now - first_pr[login]).days / 30.44)
        if count / months_active >= min_pr_rate:
            active_logins.add(login)

    # Exclude bots/service accounts entirely
    bot_logins = {pr.author.login for pr in pull_requests if pr.author.is_bot_or_service}
    active_logins -= bot_logins

    # Group PRs by author
    prs_by_author: dict[User, list[PullRequest]] = defaultdict(list)
    for pr in pull_requests:
        if pr.state not in (PullRequestState.CLOSED, PullRequestState.MERGED):
            continue
        if pr.updated_at < since:
            continue
        if pr.author.login not in active_logins:
            continue
        prs_by_author[pr.author].append(pr)

    edges: list[SocialEdge] = []

    for author, author_prs in prs_by_author.items():
        # Accumulate reviewer scores across all this author's PRs
        reviewer_totals: dict[User, float] = defaultdict(float)

        for pr in author_prs:
            # Score interactions within this PR (already normalized 0-1)
            pr_scores = score_pr_interactions(pr.interactions, author)
            if not pr_scores:
                continue

            # Apply time decay
            delta = days_since(pr.updated_at, now)
            decay_factor = exponential_time_decay(delta)

            for reviewer, normalized_score in pr_scores.items():
                reviewer_totals[reviewer] += normalized_score * decay_factor

        if not reviewer_totals:
            continue

        # Final normalization across all PRs for this author
        max_total = max(reviewer_totals.values())
        if max_total == 0:
            continue

        for reviewer, total in reviewer_totals.items():
            if reviewer.is_bot_or_service:
                continue
            if min_pr_rate > 0 and reviewer.login not in active_logins:
                continue
            weight = total / max_total
            if weight >= MINIMUM_SOCIAL_WEIGHT:
                edges.append(SocialEdge(author=author, reviewer=reviewer, weight=weight))

    return edges


def edges_to_networkx(edges: list[SocialEdge]) -> nx.DiGraph:
    """Convert social edges to a NetworkX directed graph."""
    g = nx.DiGraph()
    for edge in edges:
        g.add_edge(
            edge.author.login,
            edge.reviewer.login,
            weight=edge.weight,
            author_name=edge.author.name or edge.author.login,
            reviewer_name=edge.reviewer.name or edge.reviewer.login,
        )
    return g
