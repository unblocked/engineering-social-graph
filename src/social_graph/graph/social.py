"""Social/collaboration graph builder.

Builds a weighted directed graph of author->reviewer relationships from PR
review activity.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import networkx as nx

from social_graph.decay import THREE_SIGMA_DAYS, days_since, exponential_time_decay
from social_graph.models import PullRequest, PullRequestState, SocialEdge, User
from social_graph.scoring import MINIMUM_SOCIAL_WEIGHT, score_pr_interactions


MIN_PRS_PER_MONTH = 7.0
MIN_ACTIVE_AUTHOR_FRACTION = 0.2
DEFAULT_MIN_PR_RATE_STEPS = (7.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.0)


@dataclass(frozen=True)
class SocialGraphDiagnostics:
    min_pr_rate: float
    total_authors: int
    active_authors: int
    retained_fraction: float
    edges: int
    authors_with_edges: int
    reviewers_with_edges: int

    def meets_health_threshold(self, min_active_author_fraction: float = MIN_ACTIVE_AUTHOR_FRACTION) -> bool:
        return self.edges > 0 and self.retained_fraction >= min_active_author_fraction


def _candidate_pull_requests(
    pull_requests: list[PullRequest],
    now: datetime,
) -> list[PullRequest]:
    """Return PRs that are eligible to contribute to the social graph."""
    since = now - timedelta(days=THREE_SIGMA_DAYS)
    return [
        pr
        for pr in pull_requests
        if pr.state in (PullRequestState.CLOSED, PullRequestState.MERGED)
        and pr.updated_at >= since
        and not pr.author.is_bot_or_service
    ]


def _compute_active_logins(
    pull_requests: list[PullRequest],
    now: datetime,
    min_pr_rate: float,
) -> tuple[set[str], int]:
    """Return the graph-eligible authors retained by the PR-rate filter.

    Intentionally uses only the PRs that are eligible to contribute to the
    current social graph, not a contributor's full authored history. This keeps
    the PR-rate filter aligned with the graph's recent-window semantics instead
    of diluting current activity with stale history.
    """
    authored_count: dict[str, int] = defaultdict(int)
    first_pr: dict[str, datetime] = {}

    for pr in pull_requests:
        login = pr.author.login
        authored_count[login] += 1
        if login not in first_pr or pr.created_at < first_pr[login]:
            first_pr[login] = pr.created_at

    eligible_logins = set(authored_count)
    if min_pr_rate <= 0:
        return eligible_logins, len(eligible_logins)

    active_logins: set[str] = set()
    for login in eligible_logins:
        count = authored_count[login]
        months_active = max(1.0, (now - first_pr[login]).days / 30.44)
        if count / months_active >= min_pr_rate:
            active_logins.add(login)

    return active_logins, len(eligible_logins)


def _build_social_graph_from_candidates(
    candidate_prs: list[PullRequest],
    now: datetime,
    min_pr_rate: float,
) -> tuple[list[SocialEdge], set[str], int]:
    """Build social edges from already-filtered candidate PRs."""
    active_logins, total_authors = _compute_active_logins(candidate_prs, now, min_pr_rate)

    # Group PRs by author
    prs_by_author: dict[User, list[PullRequest]] = defaultdict(list)
    for pr in candidate_prs:
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

    return edges, active_logins, total_authors


def _build_social_graph_diagnostics_from_candidates(
    candidate_prs: list[PullRequest],
    now: datetime,
    min_pr_rate: float,
) -> tuple[list[SocialEdge], SocialGraphDiagnostics]:
    """Build social edges and diagnostics from already-filtered candidate PRs."""
    edges, active_logins, total_authors = _build_social_graph_from_candidates(
        candidate_prs, now, min_pr_rate
    )
    retained_fraction = len(active_logins) / total_authors if total_authors else 0.0
    authors_with_edges = {e.author.login for e in edges}
    reviewers_with_edges = {e.reviewer.login for e in edges}

    diagnostics = SocialGraphDiagnostics(
        min_pr_rate=min_pr_rate,
        total_authors=total_authors,
        active_authors=len(active_logins),
        retained_fraction=retained_fraction,
        edges=len(edges),
        authors_with_edges=len(authors_with_edges),
        reviewers_with_edges=len(reviewers_with_edges),
    )
    return edges, diagnostics


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

    candidate_prs = _candidate_pull_requests(pull_requests, now)
    edges, _, _ = _build_social_graph_from_candidates(candidate_prs, now, min_pr_rate)
    return edges


def build_social_graph_with_diagnostics(
    pull_requests: list[PullRequest],
    now: datetime | None = None,
    min_pr_rate: float = MIN_PRS_PER_MONTH,
) -> tuple[list[SocialEdge], SocialGraphDiagnostics]:
    """Build social edges and report how aggressively the PR-rate filter cut the graph."""
    if now is None:
        now = datetime.now(timezone.utc)

    candidate_prs = _candidate_pull_requests(pull_requests, now)
    return _build_social_graph_diagnostics_from_candidates(candidate_prs, now, min_pr_rate)


def resolve_social_graph_threshold(
    pull_requests: list[PullRequest],
    now: datetime | None = None,
    min_pr_rate: float | None = None,
    min_active_author_fraction: float = MIN_ACTIVE_AUTHOR_FRACTION,
) -> tuple[list[SocialEdge], SocialGraphDiagnostics, list[SocialGraphDiagnostics]]:
    """Choose a usable PR-rate threshold for the social graph.

    If ``min_pr_rate`` is provided, it is treated as a strict user override.
    Otherwise the default threshold is tried first and then stepped down until
    the retained-author fraction and edge count look healthy.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    candidate_prs = _candidate_pull_requests(pull_requests, now)

    if min_pr_rate is not None:
        edges, diagnostics = _build_social_graph_diagnostics_from_candidates(
            candidate_prs, now, min_pr_rate
        )
        return edges, diagnostics, [diagnostics]

    attempts: list[SocialGraphDiagnostics] = []
    chosen_edges: list[SocialEdge] = []
    chosen_diag: SocialGraphDiagnostics | None = None

    for candidate in DEFAULT_MIN_PR_RATE_STEPS:
        edges, diagnostics = _build_social_graph_diagnostics_from_candidates(
            candidate_prs, now, candidate
        )
        attempts.append(diagnostics)
        chosen_edges = edges
        chosen_diag = diagnostics
        if diagnostics.meets_health_threshold(min_active_author_fraction):
            break

    assert chosen_diag is not None
    return chosen_edges, chosen_diag, attempts


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
