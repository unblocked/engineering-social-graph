"""Build pipeline — single command to fetch, compute, and store everything.

Steps:
1. Fetch PRs from GitHub (GraphQL)
2. Build social graph edges
3. Compute LOC per user (git log)
4. Detect code area expertise (git log per directory)
5. Community detection (Louvain)
6. Team labels (Anthropic Haiku)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from social_graph.config import Config
from social_graph.fetcher.cache import PRCache


def run_build(
    config: Config,
    org: str,
    repos: list[str],
    repo_path: str,
    since_prs: str | None = None,
    since_git: str = "6 months ago",
    limit: int | None = None,
    skip_fetch: bool = False,
    fresh: bool = False,
    log: Callable[[str], None] = print,
) -> None:
    """Execute the full build pipeline and store all results in SQLite."""
    cache = PRCache(config.cache_db_path)

    try:
        if skip_fetch:
            log("[dim]Skipping fetch — using cached PR data.[/dim]")
        else:
            # Step 1: Fetch PRs
            log("[cyan]Step 1/6:[/cyan] Fetching PRs from GitHub...")
            from social_graph.fetcher.github import GitHubFetcher

            fetcher = GitHubFetcher(config.github_token)
            since_dt = None
            if since_prs:
                since_dt = datetime.fromisoformat(since_prs).replace(tzinfo=timezone.utc)

            for repo_name in repos:
                full_name = f"{org}/{repo_name}"
                fetch_since = since_dt
                if fetch_since is None and not fresh:
                    fetch_since = cache.get_last_synced(full_name)

                if fetch_since:
                    log(f"  Fetching PRs for {full_name} since {fetch_since.date()}...")
                else:
                    log(f"  Fetching PRs for {full_name}...")

                prs = fetcher.fetch_repo_prs(full_name, since=fetch_since, limit=limit)
                for pr in prs:
                    cache.upsert_pr(pr)
                if prs:
                    cache.set_last_synced(full_name, datetime.now(timezone.utc))
                log(f"  Cached {len(prs)} PRs.")

        # Step 2: Build social graph
        log("[cyan]Step 2/6:[/cyan] Building social graph...")
        from social_graph.graph.social import build_social_graph

        all_prs = cache.get_all_prs()
        edges = build_social_graph(all_prs)
        cache.save_social_edges(edges)
        log(f"  {len(edges)} edges.")

        # Step 3: Compute LOC per user
        log(f"[cyan]Step 3/6:[/cyan] Computing LOC from git log (since {since_git})...")
        from social_graph.output.visualization import compute_lines_from_git

        # Build a mapping from git author names and email prefixes to GitHub logins
        # so that compute_lines_from_git can resolve authors whose email prefix
        # doesn't match their GitHub login (e.g., "jane.doe" → "jdoe").
        known_users: dict[str, str] = {}
        logins: set[str] = set()
        for pr in all_prs:
            login = pr.author.login
            known_users[login] = login
            logins.add(login)
            for interaction in pr.interactions:
                ilogin = interaction.user.login
                known_users[ilogin] = ilogin
                logins.add(ilogin)

        # Second pass: add name mappings only where they don't collide
        # with a known login (avoids misattribution when a display name
        # matches another user's login).
        for pr in all_prs:
            if pr.author.name and pr.author.name not in logins:
                known_users.setdefault(pr.author.name, pr.author.login)
            for interaction in pr.interactions:
                if interaction.user.name and interaction.user.name not in logins:
                    known_users.setdefault(interaction.user.name, interaction.user.login)

        lines_by_user = compute_lines_from_git(repo_path, since_git, known_users=known_users)
        cache.save_lines_by_user(lines_by_user)
        top = sorted(lines_by_user.items(), key=lambda x: -x[1])[:5]
        for name, loc in top:
            log(f"  {name}: {loc:,} lines")

        # Step 4: Detect code area expertise
        log("[cyan]Step 4/6:[/cyan] Detecting code area experts...")
        from social_graph.graph.code_expertise import detect_areas

        areas = detect_areas(repo_path, max_depth=2, min_commits=10, since=since_git)
        cache.save_code_areas(areas)
        log(f"  {len(areas)} areas.")

        # Step 5: Community detection
        log("[cyan]Step 5/6:[/cyan] Detecting communities...")
        from social_graph.output.visualization import detect_communities

        communities, member_to_community = detect_communities(edges)
        cache.save_communities(communities)
        log(f"  {len(communities)} communities.")

        # Step 6: Team labels
        log("[cyan]Step 6/6:[/cyan] Generating team labels...")
        names = {}
        for e in edges:
            names[e.author.login] = e.author.display_name
            names[e.reviewer.login] = e.reviewer.display_name

        from social_graph.output.grid import _derive_team_labels

        team_labels = _derive_team_labels(communities, areas, names, api_key=config.anthropic_api_key)
        cache.save_team_labels(team_labels)
        for idx, label in sorted(team_labels.items()):
            log(f"  Team {idx + 1}: {label}")

        # Save build metadata
        cache.save_build_meta({
            "org": org,
            "repos": ",".join(repos),
            "repo_path": repo_path,
            "since_git": since_git,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "pr_count": str(len(all_prs)),
            "edge_count": str(len(edges)),
            "area_count": str(len(areas)),
            "community_count": str(len(communities)),
        })

        log("[green]Build complete.[/green]")

    finally:
        cache.close()
