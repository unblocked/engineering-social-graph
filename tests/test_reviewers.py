"""Tests for the reviewer recommendation engine."""

from __future__ import annotations

from unittest.mock import patch

from social_graph.graph.reviewers import (
    WEIGHT_EXPERTISE,
    WEIGHT_LOAD_BALANCE,
    WEIGHT_PROXIMITY,
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
