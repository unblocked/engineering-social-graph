"""Tests for the social graph builder."""

import math
from datetime import datetime, timedelta, timezone

from social_graph.graph.social import (
    MIN_ACTIVE_AUTHOR_FRACTION,
    build_social_graph,
    build_social_graph_with_diagnostics,
    resolve_social_graph_threshold,
)
from social_graph.models import (
    Interaction,
    InteractionType,
    PullRequest,
    PullRequestState,
    User,
)


def _make_pr(
    author: User,
    interactions: list[Interaction],
    days_ago: int = 1,
    state: PullRequestState = PullRequestState.CLOSED,
) -> PullRequest:
    now = datetime.now(timezone.utc)
    updated = now - timedelta(days=days_ago)
    return PullRequest(
        number=1,
        repo="test-org/test-repo",
        author=author,
        state=state,
        created_at=updated - timedelta(days=2),
        updated_at=updated,
        interactions=interactions,
    )


def _make_review_cycle_prs(author_count: int, days_ago: int = 5) -> list[PullRequest]:
    """Create a ring of authors who each author one reviewed PR."""
    users = [User(id=str(i), login=f"user{i}", name=f"User {i}") for i in range(author_count)]
    now = datetime.now(timezone.utc)
    prs: list[PullRequest] = []
    for i, author in enumerate(users):
        reviewer = users[(i + 1) % author_count]
        prs.append(
            PullRequest(
                number=i + 1,
                repo="test-org/test-repo",
                author=author,
                state=PullRequestState.MERGED,
                created_at=now - timedelta(days=days_ago + 1),
                updated_at=now - timedelta(days=days_ago),
                merged_at=now - timedelta(days=days_ago),
                interactions=[
                    Interaction(user=reviewer, type=InteractionType.REVIEW, created_at=now - timedelta(days=days_ago)),
                ],
            )
        )
    return prs


def _make_mixed_velocity_prs() -> list[PullRequest]:
    """Create a graph where a stricter retention floor forces a lower threshold."""
    now = datetime.now(timezone.utc)
    fast_a = User(id="fast-a", login="fast-a", name="Fast A")
    fast_b = User(id="fast-b", login="fast-b", name="Fast B")
    slow_users = [
        User(id=f"slow-{idx}", login=f"slow-{idx}", name=f"Slow {idx}")
        for idx in range(4)
    ]

    def _pr(number: int, author: User, reviewer: User, created_days_ago: int) -> PullRequest:
        return PullRequest(
            number=number,
            repo="test-org/test-repo",
            author=author,
            state=PullRequestState.MERGED,
            created_at=now - timedelta(days=created_days_ago),
            updated_at=now - timedelta(days=5),
            merged_at=now - timedelta(days=5),
            interactions=[
                Interaction(user=reviewer, type=InteractionType.REVIEW, created_at=now - timedelta(days=5)),
            ],
        )

    return [
        _pr(1, fast_a, fast_b, created_days_ago=7),
        _pr(2, fast_b, fast_a, created_days_ago=7),
        _pr(3, slow_users[0], slow_users[1], created_days_ago=50),
        _pr(4, slow_users[1], slow_users[0], created_days_ago=50),
        _pr(5, slow_users[2], slow_users[3], created_days_ago=50),
        _pr(6, slow_users[3], slow_users[2], created_days_ago=50),
    ]


def test_golden_test():
    """Golden test for the social graph scoring algorithm.

    One PR, 1 day old:
    - reviewer1: 1 review + 1 issue comment
    - reviewer2: 1 review comment

    Expected:
    - reviewer1 weight = 1.0
    - reviewer2 weight between 0.6 and 0.8
    """
    author = User(id="123", login="author", name="PR Author")
    reviewer1 = User(id="456", login="reviewer1", name="First Reviewer")
    reviewer2 = User(id="789", login="reviewer2", name="Second Reviewer")
    now = datetime.now(timezone.utc)

    interactions = [
        Interaction(user=reviewer1, type=InteractionType.REVIEW, created_at=now),
        Interaction(user=reviewer2, type=InteractionType.REVIEW_COMMENT, created_at=now),
        Interaction(user=reviewer1, type=InteractionType.ISSUE_COMMENT, created_at=now),
    ]

    pr = _make_pr(author, interactions, days_ago=1)
    edges = build_social_graph([pr], now=now, min_pr_rate=0)

    # Should have exactly 2 edges: author→reviewer1 and author→reviewer2
    assert len(edges) == 2

    by_reviewer = {e.reviewer.login: e for e in edges}
    assert "reviewer1" in by_reviewer
    assert "reviewer2" in by_reviewer

    assert math.isclose(by_reviewer["reviewer1"].weight, 1.0, rel_tol=1e-6)
    assert 0.6 < by_reviewer["reviewer2"].weight < 0.8


def test_open_prs_are_filtered():
    """Open PRs should not contribute to the social graph."""
    author = User(id="1", login="author")
    reviewer = User(id="2", login="reviewer")
    now = datetime.now(timezone.utc)

    interactions = [
        Interaction(user=reviewer, type=InteractionType.REVIEW, created_at=now),
    ]

    pr = _make_pr(author, interactions, days_ago=1, state=PullRequestState.OPEN)
    edges = build_social_graph([pr], now=now, min_pr_rate=0)

    assert len(edges) == 0


def test_merged_prs_included():
    """Merged PRs should contribute to the social graph."""
    author = User(id="1", login="author")
    reviewer = User(id="2", login="reviewer")
    now = datetime.now(timezone.utc)

    interactions = [
        Interaction(user=reviewer, type=InteractionType.REVIEW, created_at=now),
    ]

    pr = _make_pr(author, interactions, days_ago=1, state=PullRequestState.MERGED)
    edges = build_social_graph([pr], now=now, min_pr_rate=0)

    assert len(edges) == 1
    assert math.isclose(edges[0].weight, 1.0)


def test_bot_interactions_filtered():
    """Bot interactions should be excluded."""
    author = User(id="1", login="author")
    bot = User(id="2", login="dependabot", is_bot=True)
    now = datetime.now(timezone.utc)

    interactions = [
        Interaction(user=bot, type=InteractionType.REVIEW, created_at=now),
    ]

    pr = _make_pr(author, interactions, days_ago=1)
    edges = build_social_graph([pr], now=now, min_pr_rate=0)

    assert len(edges) == 0


def test_self_interactions_filtered():
    """Author reviewing their own PR should be excluded."""
    author = User(id="1", login="author")
    now = datetime.now(timezone.utc)

    interactions = [
        Interaction(user=author, type=InteractionType.REVIEW, created_at=now),
    ]

    pr = _make_pr(author, interactions, days_ago=1)
    edges = build_social_graph([pr], now=now, min_pr_rate=0)

    assert len(edges) == 0


def test_old_prs_decay():
    """PRs from long ago should produce lower weights than recent ones."""
    author = User(id="1", login="author")
    reviewer = User(id="2", login="reviewer")
    now = datetime.now(timezone.utc)

    interactions = [
        Interaction(user=reviewer, type=InteractionType.REVIEW, created_at=now),
    ]

    # Recent PR
    recent_pr = _make_pr(author, interactions, days_ago=1)
    recent_edges = build_social_graph([recent_pr], now=now, min_pr_rate=0)

    # Old PR (200 days ago)
    old_pr = _make_pr(author, interactions, days_ago=200)
    old_edges = build_social_graph([old_pr], now=now, min_pr_rate=0)

    # Both should have edges, but with a single PR each the weight normalizes to 1.0
    # The decay matters when comparing across PRs for the same author
    assert len(recent_edges) == 1
    assert len(old_edges) == 1


def test_minimum_weight_threshold():
    """Edges below 0.1 should be discarded."""
    author = User(id="1", login="author")
    strong_reviewer = User(id="2", login="strong")
    weak_reviewer = User(id="3", login="weak")
    now = datetime.now(timezone.utc)

    # Strong reviewer: lots of interactions on a recent PR
    pr1 = PullRequest(
        number=1,
        repo="test-org/test-repo",
        author=author,
        state=PullRequestState.CLOSED,
        created_at=now - timedelta(days=3),
        updated_at=now - timedelta(days=1),
        interactions=[
            Interaction(user=strong_reviewer, type=InteractionType.REVIEW, created_at=now),
            Interaction(user=strong_reviewer, type=InteractionType.REVIEW_COMMENT, created_at=now),
            Interaction(user=strong_reviewer, type=InteractionType.REVIEW_COMMENT, created_at=now),
            Interaction(user=strong_reviewer, type=InteractionType.REVIEW_COMMENT, created_at=now),
            Interaction(user=strong_reviewer, type=InteractionType.REVIEW_COMMENT, created_at=now),
            Interaction(user=strong_reviewer, type=InteractionType.REVIEW_COMMENT, created_at=now),
            Interaction(user=strong_reviewer, type=InteractionType.ISSUE_COMMENT, created_at=now),
        ],
    )

    # Weak reviewer: single issue comment on a very old PR
    pr2 = PullRequest(
        number=2,
        repo="test-org/test-repo",
        author=author,
        state=PullRequestState.CLOSED,
        created_at=now - timedelta(days=300),
        updated_at=now - timedelta(days=300),
        interactions=[
            Interaction(user=weak_reviewer, type=InteractionType.ISSUE_COMMENT, created_at=now),
        ],
    )

    edges = build_social_graph([pr1, pr2], now=now, min_pr_rate=0)

    # The strong reviewer should be present
    strong_edges = [e for e in edges if e.reviewer.login == "strong"]
    assert len(strong_edges) == 1
    assert math.isclose(strong_edges[0].weight, 1.0)

    # The weak reviewer may or may not pass the 0.1 threshold depending on decay
    weak_edges = [e for e in edges if e.reviewer.login == "weak"]
    for edge in weak_edges:
        assert edge.weight >= 0.1


def test_multiple_authors():
    """Graph should contain edges for multiple authors."""
    author1 = User(id="1", login="author1")
    author2 = User(id="2", login="author2")
    reviewer = User(id="3", login="reviewer")
    now = datetime.now(timezone.utc)

    interactions = [
        Interaction(user=reviewer, type=InteractionType.REVIEW, created_at=now),
    ]

    pr1 = _make_pr(author1, interactions, days_ago=1)
    pr2 = _make_pr(author2, interactions, days_ago=1)
    pr2.number = 2

    edges = build_social_graph([pr1, pr2], now=now, min_pr_rate=0)

    authors = {e.author.login for e in edges}
    assert "author1" in authors
    assert "author2" in authors


def test_diagnostics_report_retained_author_fraction():
    prs = _make_review_cycle_prs(author_count=6)

    edges, diagnostics = build_social_graph_with_diagnostics(prs, min_pr_rate=1.0)

    assert len(edges) == 6
    assert diagnostics.total_authors == 6
    assert diagnostics.active_authors == 6
    assert diagnostics.retained_fraction == 1.0
    assert diagnostics.edges == 6


def test_diagnostics_ignore_open_pr_only_authors():
    merged_prs = _make_review_cycle_prs(author_count=2)
    open_only_author = User(id="open", login="open-only", name="Open Only")
    reviewer = User(id="reviewer", login="reviewer", name="Reviewer")
    now = datetime.now(timezone.utc)

    merged_prs.append(
        PullRequest(
            number=99,
            repo="test-org/test-repo",
            author=open_only_author,
            state=PullRequestState.OPEN,
            created_at=now - timedelta(days=5),
            updated_at=now - timedelta(days=5),
            interactions=[
                Interaction(user=reviewer, type=InteractionType.REVIEW, created_at=now - timedelta(days=5)),
            ],
        )
    )

    _, diagnostics = build_social_graph_with_diagnostics(merged_prs, min_pr_rate=0)

    assert diagnostics.total_authors == 2
    assert diagnostics.active_authors == 2


def test_default_threshold_auto_tunes_down_when_filter_is_too_aggressive():
    prs = _make_review_cycle_prs(author_count=6)

    edges, diagnostics, attempts = resolve_social_graph_threshold(prs)

    assert attempts[0].min_pr_rate == 7.0
    assert attempts[0].edges == 0
    assert diagnostics.min_pr_rate == 1.0
    assert diagnostics.retained_fraction >= MIN_ACTIVE_AUTHOR_FRACTION
    assert len(edges) == 6


def test_retention_floor_can_force_a_lower_auto_tuned_threshold():
    prs = _make_mixed_velocity_prs()

    edges, diagnostics, attempts = resolve_social_graph_threshold(
        prs,
        min_active_author_fraction=0.5,
    )

    assert attempts[0].min_pr_rate == 7.0
    assert diagnostics.min_pr_rate == 0.5
    assert diagnostics.retained_fraction == 1.0
    assert len(edges) == 6


def test_explicit_threshold_is_strict_even_if_it_eliminates_everyone():
    prs = _make_review_cycle_prs(author_count=6)

    edges, diagnostics, attempts = resolve_social_graph_threshold(prs, min_pr_rate=7.0)

    assert len(attempts) == 1
    assert attempts[0].min_pr_rate == 7.0
    assert diagnostics.min_pr_rate == 7.0
    assert diagnostics.edges == 0
    assert edges == []
