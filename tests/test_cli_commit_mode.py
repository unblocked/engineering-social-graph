"""CLI tests for commit-mode build and export."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from social_graph.cli import cli
from social_graph.models import BuildMode, SocialEdge, User


class DummyConfig:
    def __init__(self) -> None:
        self.github_token = ""
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
        author=User(id="alice", login="alice"),
        reviewer=User(id="bob", login="bob"),
        weight=0.75,
    )


def test_export_uses_cached_edges(monkeypatch, tmp_path):
    monkeypatch.setattr("social_graph.cli.Config", DummyConfig)
    exported = {}

    class FakeCache:
        def __init__(self, *args, **kwargs):
            pass

        def has_build_data(self):
            return True

        def load_social_edges(self):
            return [_edge()]

        def close(self):
            return None

    def fake_export_json(edges, path):
        exported["edges"] = edges
        exported["path"] = path

    monkeypatch.setattr("social_graph.fetcher.cache.PRCache", FakeCache)
    monkeypatch.setattr("social_graph.output.json_export.export_json", fake_export_json)

    out = tmp_path / "graph.json"
    result = CliRunner().invoke(cli, ["export", "--format", "json", "--output", str(out)])

    assert result.exit_code == 0
    assert exported["edges"][0].reviewer.login == "bob"
    assert exported["path"] == out


def test_build_commit_mode_rejects_multiple_repos(monkeypatch):
    monkeypatch.setattr("social_graph.cli.Config", DummyConfig)

    def fake_run_build(**kwargs):
        raise AssertionError("run_build should not be called for invalid multi-repo commit mode")

    monkeypatch.setattr("social_graph.builder.run_build", fake_run_build)

    result = CliRunner().invoke(
        cli,
        ["build", "--mode", "commit", "--org", "acme", "--repo", "repo-a", "--repo", "repo-b"],
    )

    assert result.exit_code == 1
    assert "exactly one repository" in result.output


def test_build_commit_mode_passes_commit_mode_to_builder(monkeypatch):
    monkeypatch.setattr("social_graph.cli.Config", DummyConfig)
    captured = {}

    def fake_run_build(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("social_graph.builder.run_build", fake_run_build)

    result = CliRunner().invoke(
        cli,
        ["build", "--mode", "commit", "--org", "acme", "--repo", "repo-a"],
    )

    assert result.exit_code == 0
    assert captured["mode"] == BuildMode.COMMIT
    assert captured["repos"] == ["repo-a"]
