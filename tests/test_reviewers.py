"""Tests for the reviewer recommendation engine."""

from __future__ import annotations

from unittest.mock import patch

from social_graph.graph.reviewers import (
    WEIGHT_EXPERTISE,
    WEIGHT_LOAD_BALANCE,
    WEIGHT_PROXIMITY,
    co_commit_proximity,
    suggest_reviewers,
)
from social_graph.models import ExpertRanking, SocialEdge, User


def _make_edge(author: str, reviewer: str, weight: float) -> SocialEdge:
    return SocialEdge(
        author=User(id=author, login=author),
        reviewer=User(id=reviewer, login=reviewer),
        weight=weight,
    )


def _mock_experts(path, target, top_n=20):
    """Return fake expertise rankings for testing."""
    experts = {
        "alice": 5.0,
        "bob": 3.0,
        "charlie": 1.0,
        "dave": 0.5,
    }
    return [
        ExpertRanking(user=User(id=k, login=k), score=v, domain=target)
        for k, v in experts.items()
    ]


class TestSuggestReviewers:
    """Tests for reviewer suggestion algorithm."""

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    def test_returns_ranked_results(self, mock_exp):
        """Should return reviewers ranked by composite score."""
        edges = [
            _make_edge("author1", "alice", 0.9),
            _make_edge("author1", "bob", 0.5),
        ]
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=edges,
            author_login="author1",
        )
        assert len(results) > 0
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    def test_excludes_author(self, mock_exp):
        """PR author should not appear in suggestions."""
        edges = [_make_edge("alice", "bob", 0.9)]
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=edges,
            author_login="alice",
        )
        logins = [r["login"] for r in results]
        assert "alice" not in logins

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    def test_expertise_boosts_score(self, mock_exp):
        """Domain experts should rank higher than non-experts."""
        edges = [
            _make_edge("author1", "alice", 0.3),
            _make_edge("author1", "dave", 0.3),
        ]
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=edges,
            author_login="author1",
        )
        result_map = {r["login"]: r for r in results}
        # alice (expertise=5.0) should rank above dave (expertise=0.5)
        assert result_map["alice"]["score"] > result_map["dave"]["score"]

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    def test_proximity_boosts_score(self, mock_exp):
        """Close collaborators should rank higher, all else being equal."""
        edges = [
            _make_edge("author1", "bob", 0.9),
            _make_edge("author1", "charlie", 0.1),
        ]
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=edges,
            author_login="author1",
        )
        result_map = {r["login"]: r for r in results}
        # bob has similar expertise to charlie but much higher proximity
        assert result_map["bob"]["proximity"] > result_map["charlie"]["proximity"]

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    def test_respects_top_n(self, mock_exp):
        """Should return at most top_n results."""
        edges = [_make_edge("author1", "alice", 0.5)]
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=edges,
            author_login="author1",
            top_n=2,
        )
        assert len(results) <= 2

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    def test_includes_reason(self, mock_exp):
        """Each suggestion should include a human-readable reason."""
        edges = [_make_edge("author1", "alice", 0.9)]
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=edges,
            author_login="author1",
        )
        assert all("reason" in r for r in results)
        # alice is both an expert and close collaborator
        alice = next(r for r in results if r["login"] == "alice")
        assert "domain expert" in alice["reason"]
        assert "close collaborator" in alice["reason"]

    @patch("social_graph.graph.reviewers.experts_for_path", return_value=[])
    def test_no_experts_returns_empty(self, mock_exp):
        """Should return empty list when no expertise data exists."""
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["nonexistent.py"],
            edges=[],
            author_login="author1",
        )
        assert results == []

    def test_weights_sum_to_one(self):
        """Scoring weights should sum to 1.0."""
        total = WEIGHT_EXPERTISE + WEIGHT_PROXIMITY + WEIGHT_LOAD_BALANCE
        assert abs(total - 1.0) < 1e-9


# Shared fake git log output for co_commit_proximity tests
_FAKE_GIT_LOG = (
    "alice@example.com\t2026-03-01T00:00:00+00:00\n"
    "bob@example.com\t2026-03-15T00:00:00+00:00\n"
    "alice@example.com\t2026-02-01T00:00:00+00:00\n"
    "charlie@example.com\t2026-01-01T00:00:00+00:00\n"
)


def _mock_subprocess_run(*args, **kwargs):
    """Return a fake CompletedProcess with git log output."""
    import subprocess as sp

    return sp.CompletedProcess(args=args[0], returncode=0, stdout=_FAKE_GIT_LOG, stderr="")


def _mock_subprocess_run_empty(*args, **kwargs):
    import subprocess as sp

    return sp.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")


class TestCoCommitProximity:
    """Tests for commit-based proximity scoring."""

    @patch("social_graph.graph.reviewers.subprocess.run", side_effect=_mock_subprocess_run)
    def test_returns_proximity_scores(self, mock_run):
        """Should return scores for authors found in git log."""
        result = co_commit_proximity(
            repo_path="/fake",
            changed_paths=["src/main.py"],
        )
        assert "alice" in result
        assert "bob" in result
        assert "charlie" in result
        assert len(result) == 3

    @patch("social_graph.graph.reviewers.subprocess.run", side_effect=_mock_subprocess_run)
    def test_excludes_author(self, mock_run):
        """Author login should be excluded from results."""
        result = co_commit_proximity(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            author_login="alice",
        )
        assert "alice" not in result
        assert "bob" in result

    @patch("social_graph.graph.reviewers.subprocess.run", side_effect=_mock_subprocess_run)
    def test_normalizes_to_zero_one(self, mock_run):
        """All scores should be between 0 and 1 inclusive."""
        result = co_commit_proximity(
            repo_path="/fake",
            changed_paths=["src/main.py"],
        )
        for score in result.values():
            assert 0 <= score <= 1
        # At least one score should be exactly 1.0 (the max)
        assert any(score == 1.0 for score in result.values())

    @patch("social_graph.graph.reviewers.subprocess.run", side_effect=_mock_subprocess_run_empty)
    def test_empty_log_returns_empty(self, mock_run):
        """Empty git log output should return empty dict."""
        result = co_commit_proximity(
            repo_path="/fake",
            changed_paths=["src/main.py"],
        )
        assert result == {}


class TestCommitMode:
    """Tests for mode='commits' in suggest_reviewers."""

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    @patch(
        "social_graph.graph.reviewers.co_commit_proximity",
        return_value={"bob": 0.8, "charlie": 0.3},
    )
    def test_commit_mode_uses_co_commit_proximity(self, mock_co, mock_exp):
        """mode='commits' should delegate proximity to co_commit_proximity."""
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=[],
            author_login="author1",
            mode="commits",
        )
        mock_co.assert_called_once_with("/fake", ["src/main.py"], "author1")
        result_map = {r["login"]: r for r in results}
        assert result_map["bob"]["proximity"] == 0.8
        assert result_map["charlie"]["proximity"] == 0.3

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    def test_pr_mode_is_default(self, mock_exp):
        """Calling without mode should use PR edges (default behavior)."""
        edges = [
            _make_edge("author1", "alice", 0.9),
            _make_edge("author1", "bob", 0.5),
        ]
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=edges,
            author_login="author1",
        )
        assert len(results) > 0
        result_map = {r["login"]: r for r in results}
        assert result_map["alice"]["proximity"] == 0.9

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    @patch(
        "social_graph.graph.reviewers.co_commit_proximity",
        return_value={"bob": 0.5},
    )
    def test_commit_mode_no_edges_needed(self, mock_co, mock_exp):
        """mode='commits' should work with an empty edges list."""
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=[],
            author_login="author1",
            mode="commits",
        )
        assert len(results) > 0
        logins = [r["login"] for r in results]
        assert "bob" in logins
