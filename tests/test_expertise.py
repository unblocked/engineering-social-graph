"""Tests for the expertise graph builder."""

from unittest.mock import patch

from social_graph.graph.expertise import experts_for_repo
from social_graph.models import User


def test_experts_for_repo_ranking():
    """Top merged PR authors should be ranked correctly."""
    alice = User(id="1", login="alice", name="Alice")
    bob = User(id="2", login="bob", name="Bob")
    charlie = User(id="3", login="charlie", name="Charlie")

    counts = {alice: 50, bob: 30, charlie: 10}
    rankings = experts_for_repo(counts, "org/repo", top_n=3)

    assert len(rankings) == 3
    assert rankings[0].user == alice
    assert rankings[0].score == 50.0
    assert rankings[1].user == bob
    assert rankings[2].user == charlie


def test_experts_for_repo_top_n():
    """Should respect top_n limit."""
    users = {User(id=str(i), login=f"user{i}"): i * 10 for i in range(10)}
    rankings = experts_for_repo(users, "org/repo", top_n=3)
    assert len(rankings) == 3


def test_experts_for_path_with_mock_git():
    """Test file expertise with mocked git log output."""
    mock_output = (
        "Alice\talice@example.com\t2026-03-01T10:00:00+00:00\n"
        "Bob\tbob@example.com\t2026-03-15T10:00:00+00:00\n"
        "Alice\talice@example.com\t2026-03-20T10:00:00+00:00\n"
        "Alice\talice@example.com\t2026-02-01T10:00:00+00:00\n"
        "Bob\tbob@example.com\t2026-01-01T10:00:00+00:00\n"
    )

    with patch("social_graph.graph.expertise.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = mock_output

        from social_graph.graph.expertise import experts_for_path

        rankings = experts_for_path("/fake/repo", "src/main.py", apply_decay=False)

    assert len(rankings) == 2
    # Alice has 3 commits, Bob has 2
    assert rankings[0].user.login == "alice@example.com"
    assert rankings[0].score == 3.0
    assert rankings[1].user.login == "bob@example.com"
    assert rankings[1].score == 2.0


def test_experts_for_path_empty_git_log():
    """Should return empty list when git log has no results."""
    with patch("social_graph.graph.expertise.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""

        from social_graph.graph.expertise import experts_for_path

        rankings = experts_for_path("/fake/repo", "nonexistent.py")

    assert rankings == []
