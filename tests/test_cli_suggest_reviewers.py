"""CLI tests for suggest-reviewers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from click.testing import CliRunner

from social_graph.cli import cli
from social_graph.models import SocialEdge, User


class DummyConfig:
    def __init__(self) -> None:
        self.github_token = "token"
        self.anthropic_api_key = ""
        self.cache_dir = Path("/tmp/social-graph-tests")
        self.min_pr_rate = 7.0

    @property
    def cache_db_path(self) -> Path:
        return self.cache_dir / "cache.db"

    def ensure_cache_dir(self) -> None:
        return None


def _edge() -> SocialEdge:
    return SocialEdge(
        author=User(id="author1", login="author1"),
        reviewer=User(id="reviewer1", login="reviewer1"),
        weight=0.8,
    )


def test_suggest_reviewers_commit_mode_bypasses_cache(monkeypatch):
    captured = {}

    monkeypatch.setattr("social_graph.cli.Config", DummyConfig)

    class RaisingCache:
        def __init__(self, *args, **kwargs):
            raise AssertionError("PRCache should not be used in commit mode")

    monkeypatch.setattr("social_graph.fetcher.cache.PRCache", RaisingCache)

    def fake_subprocess_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="12345+Alice@users.noreply.github.com\n",
            stderr="",
        )

    monkeypatch.setattr("subprocess.run", fake_subprocess_run)

    def fake_suggest_reviewers(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("social_graph.graph.reviewers.suggest_reviewers", fake_suggest_reviewers)

    result = CliRunner().invoke(cli, ["suggest-reviewers", "--mode", "commits", "src/main.py"])

    assert result.exit_code == 0
    assert captured["mode"] == "commits"
    assert captured["edges"] == []
    assert captured["author_login"] == "alice"


def test_suggest_reviewers_prs_mode_uses_cached_edges_and_accepts_legacy_build_meta(monkeypatch):
    captured = {}
    edges = [_edge()]

    monkeypatch.setattr("social_graph.cli.Config", DummyConfig)

    class FakeCache:
        def __init__(self, *args, **kwargs):
            pass

        def has_build_data(self):
            return True

        def load_build_meta(self):
            return {}

        def load_social_edges(self):
            return edges

        def close(self):
            return None

    monkeypatch.setattr("social_graph.fetcher.cache.PRCache", FakeCache)

    def fake_suggest_reviewers(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("social_graph.graph.reviewers.suggest_reviewers", fake_suggest_reviewers)

    result = CliRunner().invoke(cli, ["suggest-reviewers", "--author", "author1", "src/main.py"])

    assert result.exit_code == 0
    assert captured["mode"] == "prs"
    assert captured["edges"] == edges
    assert captured["author_login"] == "author1"


def test_suggest_reviewers_prs_mode_rejects_commit_build_data(monkeypatch):
    monkeypatch.setattr("social_graph.cli.Config", DummyConfig)

    class FakeCache:
        def __init__(self, *args, **kwargs):
            pass

        def has_build_data(self):
            return True

        def load_build_meta(self):
            return {"mode": "commit"}

        def load_social_edges(self):
            raise AssertionError("commit-mode cache should be rejected before loading edges")

        def close(self):
            return None

    monkeypatch.setattr("social_graph.fetcher.cache.PRCache", FakeCache)

    def fake_suggest_reviewers(**kwargs):
        raise AssertionError("suggest_reviewers should not be called when cache mode is invalid")

    monkeypatch.setattr("social_graph.graph.reviewers.suggest_reviewers", fake_suggest_reviewers)

    result = CliRunner().invoke(cli, ["suggest-reviewers", "--author", "author1", "src/main.py"])

    assert result.exit_code == 1
    assert "generated in commit mode" in result.output
