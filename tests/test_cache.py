"""Tests for the SQLite PR cache."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from social_graph.fetcher.cache import PRCache
from social_graph.models import (
    Commit,
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


def _make_test_commit() -> Commit:
    now = datetime.now(timezone.utc)
    return Commit(
        sha="abc123",
        repo="org/repo",
        author=User(id="alice", login="alice", name="Alice"),
        committed_at=now,
        files=("src/main.py", "README.md"),
        coauthors=(User(id="bob", login="bob", name="Bob"),),
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


def test_commit_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        cache = PRCache(Path(tmp) / "test.db")
        commit = _make_test_commit()

        cache.upsert_commit(commit)

        commits = cache.get_all_commits(repo="org/repo")
        assert len(commits) == 1
        assert commits[0].sha == "abc123"
        assert commits[0].author.login == "alice"
        assert commits[0].files == ("src/main.py", "README.md")
        assert commits[0].coauthors[0].login == "bob"

        cache.delete_commits("org/repo")
        assert cache.get_all_commits(repo="org/repo") == []

        cache.close()
