"""Tests for the SQLite PR cache."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from social_graph.fetcher.cache import PRCache
from social_graph.models import (
    Interaction,
    InteractionType,
    PullRequest,
    PullRequestState,
    User,
)


def _make_test_pr() -> PullRequest:
    author = User(id="1", login="author", name="Author")
    reviewer = User(id="2", login="reviewer", name="Reviewer")
    now = datetime.now(timezone.utc)
    return PullRequest(
        number=42,
        repo="org/repo",
        author=author,
        state=PullRequestState.MERGED,
        created_at=now,
        updated_at=now,
        merged_at=now,
        interactions=[
            Interaction(user=reviewer, type=InteractionType.REVIEW, created_at=now),
        ],
    )


def test_upsert_and_get():
    with tempfile.TemporaryDirectory() as tmp:
        cache = PRCache(Path(tmp) / "test.db")
        pr = _make_test_pr()
        cache.upsert_pr(pr)

        prs = cache.get_prs("org/repo")
        assert len(prs) == 1
        assert prs[0].number == 42
        assert prs[0].author.login == "author"
        assert len(prs[0].interactions) == 1
        assert prs[0].interactions[0].type == InteractionType.REVIEW

        cache.close()


def test_upsert_replaces_interactions():
    with tempfile.TemporaryDirectory() as tmp:
        cache = PRCache(Path(tmp) / "test.db")
        pr = _make_test_pr()
        cache.upsert_pr(pr)

        # Add another interaction and upsert again
        reviewer2 = User(id="3", login="reviewer2")
        pr.interactions.append(
            Interaction(user=reviewer2, type=InteractionType.ISSUE_COMMENT, created_at=datetime.now(timezone.utc))
        )
        cache.upsert_pr(pr)

        prs = cache.get_prs("org/repo")
        assert len(prs) == 1
        assert len(prs[0].interactions) == 2

        cache.close()


def test_filter_by_state():
    with tempfile.TemporaryDirectory() as tmp:
        cache = PRCache(Path(tmp) / "test.db")
        pr = _make_test_pr()
        cache.upsert_pr(pr)

        merged = cache.get_prs("org/repo", state=PullRequestState.MERGED)
        assert len(merged) == 1

        closed = cache.get_prs("org/repo", state=PullRequestState.CLOSED)
        assert len(closed) == 0

        cache.close()


def test_sync_meta():
    with tempfile.TemporaryDirectory() as tmp:
        cache = PRCache(Path(tmp) / "test.db")

        assert cache.get_last_synced("org/repo") is None

        now = datetime.now(timezone.utc)
        cache.set_last_synced("org/repo", now)

        synced = cache.get_last_synced("org/repo")
        assert synced is not None
        assert synced.year == now.year

        cache.close()


def test_merged_pr_counts():
    with tempfile.TemporaryDirectory() as tmp:
        cache = PRCache(Path(tmp) / "test.db")

        author1 = User(id="1", login="alice", name="Alice")
        author2 = User(id="2", login="bob", name="Bob")
        now = datetime.now(timezone.utc)

        for i in range(3):
            cache.upsert_pr(PullRequest(
                number=i, repo="org/repo", author=author1,
                state=PullRequestState.MERGED, created_at=now, updated_at=now,
            ))
        cache.upsert_pr(PullRequest(
            number=10, repo="org/repo", author=author2,
            state=PullRequestState.MERGED, created_at=now, updated_at=now,
        ))

        counts = cache.get_merged_pr_counts("org/repo")
        assert counts[author1] == 3
        assert counts[author2] == 1

        cache.close()


def test_has_build_data_depends_on_completed_build_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        cache = PRCache(Path(tmp) / "test.db")

        assert not cache.has_build_data()

        # A completed build can legitimately produce zero social edges for
        # a small or low-activity repo, so build completion must not depend
        # on the social_edges table being non-empty.
        cache.save_build_meta({
            "built_at": datetime.now(timezone.utc).isoformat(),
            "edge_count": "0",
            "area_count": "0",
            "community_count": "0",
        })

        assert cache.has_build_data()

        cache.close()
