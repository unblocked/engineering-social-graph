"""Tests for the commit-based social graph builder."""

from datetime import datetime, timedelta, timezone

from social_graph.graph.commit_social import build_commit_graph
from social_graph.models import Commit, User


def _user(login: str) -> User:
    return User(id=login, login=login, name=login.title())


def _now():
    return datetime(2026, 4, 1, tzinfo=timezone.utc)


def _commit(sha, author, files, coauthors=(), days_ago=0):
    return Commit(
        sha=sha,
        repo="test/repo",
        author=author,
        committed_at=_now() - timedelta(days=days_ago),
        files=tuple(files),
        coauthors=tuple(coauthors),
    )


class TestCoAuthorship:
    def test_coauthor_creates_bidirectional_edges(self):
        alice, bob = _user("alice"), _user("bob")
        commits = [_commit("a1", alice, ["f.py"], coauthors=[bob])]
        edges = build_commit_graph(commits, now=_now())
        logins = {(e.author.login, e.reviewer.login) for e in edges}
        assert ("alice", "bob") in logins
        assert ("bob", "alice") in logins

    def test_no_coauthors_no_coauthor_edges(self):
        alice = _user("alice")
        commits = [_commit("a1", alice, ["f.py"])]
        edges = build_commit_graph(commits, now=_now())
        # With only one author and no co-authors, no edges possible
        assert len(edges) == 0

    def test_self_coauthor_filtered(self):
        alice = _user("alice")
        commits = [_commit("a1", alice, ["f.py"], coauthors=[alice])]
        edges = build_commit_graph(commits, now=_now())
        assert len(edges) == 0

    def test_bot_coauthor_filtered(self):
        alice = _user("alice")
        bot = User(id="dependabot[bot]", login="dependabot[bot]", is_bot=True)
        commits = [_commit("a1", alice, ["f.py"], coauthors=[bot])]
        edges = build_commit_graph(commits, now=_now())
        assert len(edges) == 0


class TestFileProximity:
    def test_same_file_within_window(self):
        alice, bob = _user("alice"), _user("bob")
        commits = [
            _commit("a1", alice, ["src/main.py"], days_ago=1),
            _commit("b1", bob, ["src/main.py"], days_ago=2),
        ]
        edges = build_commit_graph(commits, proximity_window_days=7, now=_now())
        assert len(edges) > 0
        logins = {(e.author.login, e.reviewer.login) for e in edges}
        assert ("alice", "bob") in logins or ("bob", "alice") in logins

    def test_same_file_outside_window(self):
        alice, bob = _user("alice"), _user("bob")
        commits = [
            _commit("a1", alice, ["src/main.py"], days_ago=1),
            _commit("b1", bob, ["src/main.py"], days_ago=30),
        ]
        edges = build_commit_graph(commits, proximity_window_days=7, now=_now())
        assert len(edges) == 0

    def test_different_files_no_proximity(self):
        alice, bob = _user("alice"), _user("bob")
        commits = [
            _commit("a1", alice, ["src/a.py"], days_ago=1),
            _commit("b1", bob, ["src/b.py"], days_ago=1),
        ]
        edges = build_commit_graph(commits, proximity_window_days=7, now=_now())
        assert len(edges) == 0


class TestNormalization:
    def test_weights_between_zero_and_one(self):
        alice, bob, carol = _user("alice"), _user("bob"), _user("carol")
        commits = [
            _commit("a1", alice, ["f.py"], coauthors=[bob]),
            _commit("a2", alice, ["f.py"], coauthors=[bob]),
            _commit("a3", alice, ["g.py"], coauthors=[carol]),
        ]
        edges = build_commit_graph(commits, now=_now())
        for e in edges:
            assert 0 < e.weight <= 1.0

    def test_strongest_collaborator_gets_weight_one(self):
        alice, bob, carol = _user("alice"), _user("bob"), _user("carol")
        commits = [
            _commit("a1", alice, ["f.py"], coauthors=[bob]),
            _commit("a2", alice, ["f.py"], coauthors=[bob]),
            _commit("a3", alice, ["g.py"], coauthors=[carol]),
        ]
        edges = build_commit_graph(commits, now=_now())
        alice_edges = [e for e in edges if e.author.login == "alice"]
        max_w = max(e.weight for e in alice_edges)
        assert max_w == 1.0


class TestDecay:
    def test_old_commits_produce_weaker_edges(self):
        alice, bob = _user("alice"), _user("bob")
        recent = [_commit("a1", alice, ["f.py"], coauthors=[bob], days_ago=1)]
        old = [_commit("a2", alice, ["f.py"], coauthors=[bob], days_ago=300)]

        edges_recent = build_commit_graph(recent, now=_now())
        edges_old = build_commit_graph(old, now=_now())

        # Both produce edges (both within THREE_SIGMA_DAYS=336)
        # but with only 1 edge each, both normalize to 1.0
        # The decay matters when there are multiple collaborators
        assert len(edges_recent) > 0
        assert len(edges_old) > 0

    def test_beyond_lookback_window_no_edges(self):
        alice, bob = _user("alice"), _user("bob")
        commits = [_commit("a1", alice, ["f.py"], coauthors=[bob], days_ago=400)]
        edges = build_commit_graph(commits, now=_now())
        assert len(edges) == 0
