"""Tests for build-mode plumbing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from social_graph.builder import _build_commit_mode, run_build
from social_graph.config import Config
from social_graph.fetcher.cache import PRCache


def test_build_commit_mode_rejects_multiple_repos():
    with tempfile.TemporaryDirectory() as tmp:
        cache = PRCache(Path(tmp) / "cache.db")
        with pytest.raises(ValueError, match="exactly one repository"):
            _build_commit_mode(
                cache=cache,
                org="acme",
                repos=["repo-a", "repo-b"],
                repo_path="/tmp/repo",
                since_git="6 months ago",
                log=lambda _: None,
            )
        cache.close()


def test_run_build_rejects_unsupported_mode():
    with tempfile.TemporaryDirectory() as tmp:
        config = Config(github_token="token", anthropic_api_key="", cache_dir=Path(tmp))
        config.ensure_cache_dir()

        with pytest.raises(ValueError, match="Unsupported build mode"):
            run_build(
                config=config,
                org="acme",
                repos=["repo"],
                repo_path="/tmp/repo",
                mode="weird",
                log=lambda _: None,
            )
