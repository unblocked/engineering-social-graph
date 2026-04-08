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
    min_pr_rate: float | None = None,
    min_active_author_fraction: float = 0.2,
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
        from social_graph.graph.social import (
            MIN_PRS_PER_MONTH,
            resolve_social_graph_threshold,
        )

        all_prs = cache.get_all_prs()
        edges, graph_diag, graph_attempts = resolve_social_graph_threshold(
            all_prs,
            min_pr_rate=min_pr_rate,
            min_active_author_fraction=min_active_author_fraction,
        )
        auto_tuned = min_pr_rate is None and len(graph_attempts) > 1

        if min_pr_rate is not None:
            if graph_diag.edges == 0:
                attempted = ", ".join(f"{attempt.min_pr_rate:g}" for attempt in graph_attempts)
                retained_pct = graph_diag.retained_fraction * 100
                log(
                    f"[red]Error:[/red] --min-pr-rate {min_pr_rate:g} retained only "
                    f"{graph_diag.active_authors}/{graph_diag.total_authors} authors "
                    f"({retained_pct:.0f}%) and produced {graph_diag.edges} edges."
                )
                log(
                    "Try a lower --min-pr-rate, or omit it to let the build auto-tune "
                    f"downward from the default {MIN_PRS_PER_MONTH:g}. "
                    f"Attempted threshold(s): {attempted}."
                )
                raise SystemExit(1)
            if graph_diag.retained_fraction < min_active_author_fraction:
                retained_pct = graph_diag.retained_fraction * 100
                log(
                    f"[yellow]Warning:[/yellow] --min-pr-rate {min_pr_rate:g} retained only "
                    f"{graph_diag.active_authors}/{graph_diag.total_authors} authors "
                    f"({retained_pct:.0f}%), below the configured retention floor of "
                    f"{min_active_author_fraction * 100:.0f}%. Continuing because "
                    "--min-pr-rate was explicitly set."
                )
        elif auto_tuned:
            retained_pct = graph_diag.retained_fraction * 100
            log(
                f"[yellow]Warning:[/yellow] Default min PR rate {MIN_PRS_PER_MONTH:g} was too aggressive "
                "for this repo. Auto-tuned downward to "
                f"[cyan]{graph_diag.min_pr_rate:g}[/cyan], retaining "
                f"{graph_diag.active_authors}/{graph_diag.total_authors} authors "
                f"({retained_pct:.0f}%) and producing {graph_diag.edges} edges."
            )
        elif graph_diag.total_authors and graph_diag.retained_fraction < min_active_author_fraction:
            retained_pct = graph_diag.retained_fraction * 100
            log(
                f"[yellow]Warning:[/yellow] Min PR rate {graph_diag.min_pr_rate:g} retained only "
                f"{graph_diag.active_authors}/{graph_diag.total_authors} authors "
                f"({retained_pct:.0f}%), though the graph still produced {graph_diag.edges} edges."
            )

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
            "min_pr_rate": str(graph_diag.min_pr_rate),
            "active_author_count": str(graph_diag.active_authors),
            "author_count": str(graph_diag.total_authors),
            "active_author_fraction": f"{graph_diag.retained_fraction:.6f}",
            "min_active_author_fraction": f"{min_active_author_fraction:.6f}",
            "threshold_auto_tuned": str(auto_tuned).lower(),
        })

        log("[green]Build complete.[/green]")

    finally:
        cache.close()
