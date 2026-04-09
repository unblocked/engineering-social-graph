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
    experts = {
        "alice": 5.0,
        "bob": 3.0,
        "charlie": 1.0,
        "dave": 0.5,
    }
    return [
        ExpertRanking(user=User(id=login, login=login), score=score, domain=target)
        for login, score in experts.items()
    ]


class TestSuggestReviewers:
    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    def test_returns_ranked_results(self, mock_exp):
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
        scores = [result["score"] for result in results]
        assert scores == sorted(scores, reverse=True)

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    def test_excludes_author(self, mock_exp):
        edges = [_make_edge("alice", "bob", 0.9)]
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=edges,
            author_login="alice",
        )
        assert "alice" not in [result["login"] for result in results]

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    def test_expertise_boosts_score(self, mock_exp):
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
        result_map = {result["login"]: result for result in results}
        assert result_map["alice"]["score"] > result_map["dave"]["score"]

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    def test_proximity_boosts_score(self, mock_exp):
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
        result_map = {result["login"]: result for result in results}
        assert result_map["bob"]["proximity"] > result_map["charlie"]["proximity"]

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    def test_respects_top_n(self, mock_exp):
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=[_make_edge("author1", "alice", 0.5)],
            author_login="author1",
            top_n=2,
        )
        assert len(results) <= 2

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    def test_includes_reason(self, mock_exp):
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=[_make_edge("author1", "alice", 0.9)],
            author_login="author1",
        )
        alice = next(result for result in results if result["login"] == "alice")
        assert "domain expert" in alice["reason"]
        assert "close collaborator" in alice["reason"]

    @patch("social_graph.graph.reviewers.experts_for_path", return_value=[])
    def test_no_experts_returns_empty(self, mock_exp):
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["nonexistent.py"],
            edges=[],
            author_login="author1",
        )
        assert results == []

    def test_weights_sum_to_one(self):
        total = WEIGHT_EXPERTISE + WEIGHT_PROXIMITY + WEIGHT_LOAD_BALANCE
        assert abs(total - 1.0) < 1e-9


_FAKE_GIT_LOG = (
    "alice@example.com\t2026-03-01T00:00:00+00:00\n"
    "bob@example.com\t2026-03-15T00:00:00+00:00\n"
    "alice@example.com\t2026-02-01T00:00:00+00:00\n"
    "charlie@example.com\t2026-01-01T00:00:00+00:00\n"
)


_FAKE_GIT_LOG_MIXED_CASE = (
    "Alice@example.com\t2026-03-15T00:00:00+00:00\n"
    "BOB@example.com\t2026-03-10T00:00:00+00:00\n"
)

def _mock_subprocess_run(*args, **kwargs):
    import subprocess as sp

    return sp.CompletedProcess(args=args[0], returncode=0, stdout=_FAKE_GIT_LOG, stderr="")


def _mock_subprocess_run_mixed_case(*args, **kwargs):
    import subprocess as sp

    return sp.CompletedProcess(args=args[0], returncode=0, stdout=_FAKE_GIT_LOG_MIXED_CASE, stderr="")


def _mock_subprocess_run_empty(*args, **kwargs):
    import subprocess as sp

    return sp.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")


class TestCoCommitProximity:
    @patch("social_graph.graph.reviewers.subprocess.run", side_effect=_mock_subprocess_run)
    def test_returns_proximity_scores(self, mock_run):
        result = co_commit_proximity(repo_path="/fake", changed_paths=["src/main.py"])
        assert {"alice", "bob", "charlie"} <= set(result)

    @patch("social_graph.graph.reviewers.subprocess.run", side_effect=_mock_subprocess_run)
    def test_excludes_author(self, mock_run):
        result = co_commit_proximity(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            author_login="alice",
        )
        assert "alice" not in result
        assert "bob" in result

    @patch("social_graph.graph.reviewers.subprocess.run", side_effect=_mock_subprocess_run)
    def test_normalizes_to_zero_one(self, mock_run):
        result = co_commit_proximity(repo_path="/fake", changed_paths=["src/main.py"])
        assert all(0 <= score <= 1 for score in result.values())
        assert any(score == 1.0 for score in result.values())

    @patch("social_graph.graph.reviewers.subprocess.run", side_effect=_mock_subprocess_run_empty)
    def test_empty_log_returns_empty(self, mock_run):
        assert co_commit_proximity(repo_path="/fake", changed_paths=["src/main.py"]) == {}

    @patch("social_graph.graph.reviewers.subprocess.run", side_effect=_mock_subprocess_run_mixed_case)
    def test_lowercases_email_prefixes(self, mock_run):
        result = co_commit_proximity(repo_path="/fake", changed_paths=["src/main.py"])
        assert "alice" in result
        assert "bob" in result
        assert "Alice" not in result
        assert "BOB" not in result


class TestCommitMode:
    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    @patch("social_graph.graph.reviewers.co_commit_proximity", return_value={"bob": 0.8, "charlie": 0.3})
    def test_commit_mode_uses_co_commit_proximity(self, mock_co, mock_exp):
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=[],
            author_login="author1",
            mode="commits",
        )
        mock_co.assert_called_once_with("/fake", ["src/main.py"], "author1")
        result_map = {result["login"]: result for result in results}
        assert result_map["bob"]["proximity"] == 0.8
        assert result_map["charlie"]["proximity"] == 0.3

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    def test_pr_mode_is_default(self, mock_exp):
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=[_make_edge("author1", "alice", 0.9), _make_edge("author1", "bob", 0.5)],
            author_login="author1",
        )
        result_map = {result["login"]: result for result in results}
        assert result_map["alice"]["proximity"] == 0.9

    @patch("social_graph.graph.reviewers.experts_for_path", side_effect=_mock_experts)
    @patch("social_graph.graph.reviewers.co_commit_proximity", return_value={"bob": 0.5})
    def test_commit_mode_no_edges_needed(self, mock_co, mock_exp):
        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=[],
            author_login="author1",
            mode="commits",
        )
        assert "bob" in [result["login"] for result in results]

    @patch("social_graph.graph.reviewers.experts_for_path")
    @patch("social_graph.graph.reviewers.co_commit_proximity")
    def test_mixed_case_identity_merges_and_excludes_author(self, mock_co, mock_exp):
        mock_exp.return_value = [
            ExpertRanking(user=User(id="1", login="alice@example.com"), score=5.0, domain="src/main.py"),
            ExpertRanking(user=User(id="2", login="bob@example.com"), score=3.0, domain="src/main.py"),
        ]
        mock_co.return_value = {"Alice": 0.9, "Bob": 0.4}

        results = suggest_reviewers(
            repo_path="/fake",
            changed_paths=["src/main.py"],
            edges=[],
            author_login="ALICE",
            mode="commits",
        )

        result_map = {result["login"]: result for result in results}
        assert "alice" not in result_map
        assert result_map["bob"]["expertise"] > 0
        assert result_map["bob"]["proximity"] == 0.4
