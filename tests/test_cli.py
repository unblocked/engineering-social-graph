"""CLI behavior tests for build/export flows."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from click.testing import CliRunner

from social_graph.cli import cli
from social_graph.fetcher.cache import PRCache
from social_graph.models import Interaction, InteractionType, PullRequest, PullRequestState, SocialEdge, User


def _merged_review_pr(number: int, author: User, reviewer: User, created_days_ago: int) -> PullRequest:
    now = datetime.now(timezone.utc)
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


def _fake_config_class(cache_dir):
    class FakeConfig:
        def __init__(self) -> None:
            self.github_token = ""
            self.anthropic_api_key = ""
            self.cache_dir = cache_dir

        @property
        def cache_db_path(self):
            return self.cache_dir / "cache.db"

        def ensure_cache_dir(self) -> None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    return FakeConfig


def test_build_allows_explicit_min_pr_rate_below_retention_floor_when_edges_exist(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = PRCache(cache_dir / "cache.db")

    fast_a = User(id="fast-a", login="fast-a", name="Fast A")
    fast_b = User(id="fast-b", login="fast-b", name="Fast B")
    slow_a = User(id="slow-a", login="slow-a", name="Slow A")
    slow_b = User(id="slow-b", login="slow-b", name="Slow B")

    for pr in [
        _merged_review_pr(1, fast_a, fast_b, created_days_ago=7),
        _merged_review_pr(2, fast_b, fast_a, created_days_ago=7),
        _merged_review_pr(3, slow_a, slow_b, created_days_ago=50),
        _merged_review_pr(4, slow_b, slow_a, created_days_ago=50),
    ]:
        cache.upsert_pr(pr)
    cache.close()

    monkeypatch.setattr("social_graph.cli.resolve_github_remote", lambda _: ("test-org", "test-repo"))
    monkeypatch.setattr("social_graph.cli.Config", _fake_config_class(cache_dir))
    monkeypatch.setattr("social_graph.output.visualization.compute_lines_from_git", lambda *args, **kwargs: {})
    monkeypatch.setattr("social_graph.graph.code_expertise.detect_areas", lambda *args, **kwargs: [])
    monkeypatch.setattr("social_graph.output.visualization.detect_communities", lambda edges: ([], {}))
    monkeypatch.setattr("social_graph.output.grid._derive_team_labels", lambda *args, **kwargs: {})

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build",
            "--repo-path",
            str(tmp_path),
            "--skip-fetch",
            "--min-pr-rate",
            "1",
            "--min-active-author-fraction",
            "0.9",
        ],
    )

    assert result.exit_code == 0
    normalized_output = " ".join(result.output.split())
    assert "Continuing because --min-pr-rate was explicitly set." in normalized_output


def test_export_uses_cached_social_edges_instead_of_rebuilding(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = PRCache(cache_dir / "cache.db")

    author = User(id="1", login="author", name="Author")
    reviewer = User(id="2", login="reviewer", name="Reviewer")
    cache.save_social_edges([SocialEdge(author=author, reviewer=reviewer, weight=0.75)])
    cache.save_build_meta({
        "built_at": datetime.now(timezone.utc).isoformat(),
        "edge_count": "1",
    })
    cache.close()

    monkeypatch.setattr("social_graph.cli.Config", _fake_config_class(cache_dir))

    runner = CliRunner()
    output_path = tmp_path / "graph.json"
    result = runner.invoke(
        cli,
        ["export", "--format", "json", "--output", str(output_path)],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert '"weight": 0.75' in output_path.read_text()
