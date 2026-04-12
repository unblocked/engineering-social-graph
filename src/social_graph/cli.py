"""CLI entry point for social-graph."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from social_graph.config import DEFAULT_MIN_PR_RATE, Config, resolve_github_remote, secrets_set

console = Console()


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Build a company social graph from PR history."""
    ctx.ensure_object(dict)
    config = Config()
    config.ensure_cache_dir()
    ctx.obj["config"] = config


# ---------------------------------------------------------------------------
# auth — store API keys
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--github", "github_token", default=None, help="GitHub personal access token.")
@click.option("--anthropic", "anthropic_key", default=None, help="Anthropic API key.")
def auth(github_token: str | None, anthropic_key: str | None) -> None:
    """Store API keys in the system keychain.

    \b
    Examples:
        social-graph auth --github ghp_xxxx
        social-graph auth --anthropic sk-ant-xxxx
        social-graph auth  (prompts interactively)
    """
    if github_token is None and anthropic_key is None:
        github_token = click.prompt("GitHub token (enter to skip)", default="", show_default=False, hide_input=True)
        anthropic_key = click.prompt("Anthropic API key (enter to skip)", default="", show_default=False, hide_input=True)
        github_token = github_token.strip() or None
        anthropic_key = anthropic_key.strip() or None

    if github_token is None and anthropic_key is None:
        console.print("[yellow]No keys provided.[/yellow]")
        return

    stored = []
    if github_token:
        if secrets_set("GITHUB_TOKEN", github_token.strip()):
            stored.append("GitHub token")
        else:
            console.print("[red]Failed to store GitHub token.[/red] Set GITHUB_TOKEN env var instead.")

    if anthropic_key:
        if secrets_set("ANTHROPIC_API_KEY", anthropic_key.strip()):
            stored.append("Anthropic API key")
        else:
            console.print("[red]Failed to store Anthropic key.[/red] Set ANTHROPIC_API_KEY env var instead.")

    if stored:
        console.print(f"[green]Stored: {', '.join(stored)}[/green]")

    if not stored and (github_token or anthropic_key):
        if sys.platform not in ("darwin", "win32"):
            console.print("[yellow]Hint:[/yellow] On Linux, install the `keyring` package or use environment variables.")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# status — show config and build state
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current configuration and build status."""
    config: Config = ctx.obj["config"]

    console.print("[bold]Credentials[/bold]")
    console.print(f"  GitHub token:    {'[green]configured[/green]' if config.github_token else '[red]not set[/red]'}")
    console.print(f"  Anthropic key:   {'[green]configured[/green]' if config.anthropic_api_key else '[dim]not set (optional)[/dim]'}")
    console.print()

    from social_graph.fetcher.cache import PRCache

    cache = PRCache(config.cache_db_path)
    try:
        if not cache.has_build_data():
            console.print("[bold]Build data[/bold]")
            console.print("  No build data found. Run [cyan]social-graph build[/cyan] first.")
            return

        meta = cache.load_build_meta()
        team_labels = cache.load_team_labels()

        console.print("[bold]Build data[/bold]")
        console.print(f"  Last built:      {meta.get('built_at', 'unknown')[:19]}")
        console.print(f"  Repos:           {meta.get('repos', '?')}")
        console.print(f"  PRs:             {meta.get('pr_count', '?')}")
        console.print(f"  Edges:           {meta.get('edge_count', '?')}")
        console.print(f"  Code areas:      {meta.get('area_count', '?')}")
        console.print(f"  Communities:     {meta.get('community_count', '?')}")
        if team_labels:
            console.print(f"  Team labels:     {', '.join(team_labels[k] for k in sorted(team_labels))}")

        console.print()
        console.print(f"  Cache: {config.cache_db_path}")
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# reset — clear all cached data
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
@click.pass_context
def reset(ctx: click.Context, yes: bool) -> None:
    """Delete all cached data (PRs, build results, visualizations)."""
    config: Config = ctx.obj["config"]
    db = config.cache_db_path
    html = config.cache_dir / "social-graph.html"

    if not db.exists() and not html.exists():
        console.print("[yellow]Nothing to reset — cache is already empty.[/yellow]")
        return

    if not yes:
        click.confirm("This will delete all cached PRs and build data. Continue?", abort=True)

    if db.exists():
        db.unlink()
    if html.exists():
        html.unlink()

    console.print("[green]Cache cleared.[/green] Run [cyan]social-graph build[/cyan] to start fresh.")


# ---------------------------------------------------------------------------
# label-teams — regenerate AI team labels
# ---------------------------------------------------------------------------

@cli.command("label-teams")
@click.pass_context
def label_teams(ctx: click.Context) -> None:
    """Re-generate team labels using AI (requires Anthropic API key).

    Reads communities and code areas from the cache and generates
    descriptive team names using Claude. Run 'auth --anthropic' first if needed.
    """
    config: Config = ctx.obj["config"]
    if not config.anthropic_api_key:
        console.print("[red]Error:[/red] No Anthropic API key found.")
        console.print("Run [cyan]social-graph auth --anthropic KEY[/cyan] to store one.")
        raise SystemExit(1)

    from social_graph.fetcher.cache import PRCache

    cache = PRCache(config.cache_db_path)
    try:
        if not cache.has_build_data():
            console.print("[red]No build data found.[/red] Run [cyan]social-graph build[/cyan] first.")
            raise SystemExit(1)

        edges = cache.load_social_edges()
        communities, _ = cache.load_communities()
        areas = cache.load_code_areas()

        names = {}
        for e in edges:
            names[e.author.login] = e.author.display_name
            names[e.reviewer.login] = e.reviewer.display_name

        console.print(f"Generating AI team labels for [cyan]{len(communities)}[/cyan] teams...")

        from social_graph.output.grid import _derive_team_labels

        team_labels = _derive_team_labels(communities, areas, names, api_key=config.anthropic_api_key)
        cache.save_team_labels(team_labels)

        for idx, label in sorted(team_labels.items()):
            console.print(f"  Team {idx + 1}: [green]{label}[/green]")
        console.print("[green]Team labels updated.[/green]")
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# experts — subcommand group
# ---------------------------------------------------------------------------

@cli.group()
def experts() -> None:
    """Find domain experts."""
    pass


@experts.command("path")
@click.argument("target_path")
@click.option("--repo-path", default=".", help="Path to the local git repo.")
@click.option("--top", default=5, help="Number of experts to show.")
@click.option("--by-churn", is_flag=True, help="Score by lines changed instead of commit count.")
@click.pass_context
def experts_path(ctx: click.Context, target_path: str, repo_path: str, top: int, by_churn: bool) -> None:
    """Find experts for a file or directory based on git history."""
    from social_graph.graph.expertise import experts_for_path, experts_for_path_by_churn
    from social_graph.output.table import print_expert_rankings

    if by_churn:
        rankings = experts_for_path_by_churn(repo_path, target_path, top_n=top)
    else:
        rankings = experts_for_path(repo_path, target_path, top_n=top)

    if rankings:
        print_expert_rankings(rankings, title=f"Experts for {target_path}")
    else:
        console.print(f"[yellow]No contributors found for {target_path}[/yellow]")


@experts.command("repo")
@click.option("--org", default=None, help="GitHub organization (auto-detected from git remote).")
@click.option("--repo", default=None, help="Repository name (auto-detected from git remote).")
@click.option("--repo-path", default=".", help="Path to local git repo clone (default: current directory).")
@click.option("--top", default=5, help="Number of experts to show.")
@click.pass_context
def experts_repo(ctx: click.Context, org: str | None, repo: str | None, repo_path: str, top: int) -> None:
    """Find top contributors by merged PR count."""
    config: Config = ctx.obj["config"]
    resolved_org, resolved_repos = _resolve_org_repo(repo_path, org, (repo,) if repo else ())

    from social_graph.fetcher.cache import PRCache
    from social_graph.graph.expertise import experts_for_repo
    from social_graph.output.table import print_expert_rankings

    cache = PRCache(config.cache_db_path)
    try:
        full_name = f"{resolved_org}/{resolved_repos[0]}"
        counts = cache.get_merged_pr_counts(full_name)
        if not counts:
            console.print(f"[yellow]No merged PRs found for {full_name}. Run 'build' first.[/yellow]")
            return
        rankings = experts_for_repo(counts, full_name, top_n=top)
        print_expert_rankings(rankings, title=f"Top contributors to {full_name}")
    finally:
        cache.close()


@experts.command("areas")
@click.option("--repo-path", default=".", help="Path to the local git repo.")
@click.option("--since", default="6 months ago", help="Git log lookback period.")
@click.option("--depth", default=2, help="Max directory depth to scan.")
@click.option("--top", default=5, help="Number of experts per area.")
@click.option("--min-commits", default=10, help="Minimum commits for an area to be included.")
@click.pass_context
def experts_areas(ctx: click.Context, repo_path: str, since: str, depth: int, top: int, min_commits: int) -> None:
    """Detect code areas and rank experts for each (uses git log, no LLM)."""
    from social_graph.graph.code_expertise import detect_areas

    console.print(f"Scanning [cyan]{repo_path}[/cyan] (since {since}, depth {depth})...")
    areas = detect_areas(repo_path, max_depth=depth, min_commits=min_commits, since=since)

    if not areas:
        console.print("[yellow]No areas found with enough activity.[/yellow]")
        return

    from rich.table import Table

    for area in areas[:20]:
        table = Table(title=f"{area.name} — {area.path}")
        table.add_column("Expert", style="cyan")
        table.add_column("Score", justify="right", style="magenta")
        table.add_column("Commits", justify="right")
        table.add_column("Lines", justify="right")
        table.add_column("Files", justify="right")
        table.add_column("Last", justify="right", style="dim")
        table.add_column("Bus Factor", justify="right")

        for i, expert in enumerate(area.experts[:top]):
            bf = str(area.bus_factor) if i == 0 else ""
            table.add_row(
                expert.login,
                f"{expert.score:.3f}",
                str(expert.commits),
                f"{expert.lines:,}",
                str(expert.files_touched),
                f"{expert.last_commit_days_ago}d ago",
                bf,
            )

        console.print(table)
        console.print()


@experts.command("domain")
@click.argument("domain_name")
@click.option("--repo-path", default=".", help="Path to the local git repo.")
@click.option("--domains-file", type=click.Path(exists=True), help="Path to domains.json mapping file.")
@click.option("--top", default=5, help="Number of experts to show.")
@click.pass_context
def experts_domain(ctx: click.Context, domain_name: str, repo_path: str, domains_file: str | None, top: int) -> None:
    """Find experts for a domain area based on directory mappings."""
    from social_graph.graph.expertise import experts_for_domain, load_domain_map
    from social_graph.output.table import print_expert_rankings

    if domains_file:
        domain_map = load_domain_map(Path(domains_file))
    else:
        default_path = Path(repo_path) / "domains.json"
        if default_path.exists():
            domain_map = load_domain_map(default_path)
        else:
            console.print(
                "[yellow]No domains.json found. Create one or use --domains-file, "
                "or run 'social-graph experts domains --auto' to generate one.[/yellow]"
            )
            return

    if domain_name not in domain_map:
        available = ", ".join(sorted(domain_map.keys()))
        console.print(f"[yellow]Domain '{domain_name}' not found. Available: {available}[/yellow]")
        return

    rankings = experts_for_domain(repo_path, domain_name, domain_map, top_n=top)
    if rankings:
        print_expert_rankings(rankings, title=f"Experts for domain: {domain_name}")
    else:
        console.print(f"[yellow]No contributors found for domain '{domain_name}'[/yellow]")


@experts.command("domains")
@click.option("--repo-path", default=".", help="Path to the local git repo.")
@click.option("--auto", "auto_label", is_flag=True, help="Auto-label domains using Claude Haiku.")
@click.option("--output", default="domains.json", help="Output path for the domain map.")
@click.pass_context
def experts_domains(ctx: click.Context, repo_path: str, auto_label: bool, output: str) -> None:
    """Generate or view domain→directory mappings."""
    import json

    output_path = Path(output)

    if auto_label:
        config: Config = ctx.obj["config"]
        if not config.anthropic_api_key:
            console.print("[red]Error:[/red] No Anthropic API key. Run [cyan]social-graph auth --anthropic KEY[/cyan].")
            raise SystemExit(1)

        from social_graph.graph.expertise import auto_generate_domain_map

        console.print("Analyzing directory structure with Claude Haiku...")
        domain_map = auto_generate_domain_map(repo_path, config.anthropic_api_key)
        with open(output_path, "w") as f:
            json.dump(domain_map, f, indent=2)
        console.print(f"Wrote domain map to [green]{output_path}[/green]:")
        for domain, paths in domain_map.items():
            console.print(f"  [cyan]{domain}[/cyan]: {', '.join(paths)}")
    elif output_path.exists():
        from social_graph.graph.expertise import load_domain_map

        domain_map = load_domain_map(output_path)
        for domain, paths in domain_map.items():
            console.print(f"  [cyan]{domain}[/cyan]: {', '.join(paths)}")
    else:
        console.print("[yellow]No domains.json found. Use --auto to generate one.[/yellow]")


# ---------------------------------------------------------------------------
# simulate — what-if scenario planning
# ---------------------------------------------------------------------------

@cli.group()
def simulate() -> None:
    """Run what-if simulations on the social graph.

    \b
    Subcommands:
        lose   — What happens if someone leaves?
        reorg  — What if we restructure teams?
        hire   — Where should a new hire go?
    """
    pass


@simulate.command("lose")
@click.argument("login")
@click.option("--top-backfills", default=3, help="Number of backfill candidates per area.")
@click.pass_context
def simulate_lose(ctx: click.Context, login: str, top_backfills: int) -> None:
    """Simulate a person leaving — show impact and knowledge transfer plan.

    \b
    Examples:
        social-graph simulate lose alice
        social-graph simulate lose alice --top-backfills 3
    """
    from social_graph.fetcher.cache import PRCache
    from social_graph.graph.simulate import simulate_departure

    config: Config = ctx.obj["config"]
    cache = PRCache(config.cache_db_path)
    try:
        if not cache.has_build_data():
            console.print("[red]No build data found.[/red] Run [cyan]social-graph build[/cyan] first.")
            raise SystemExit(1)

        edges = cache.load_social_edges()
        areas = cache.load_code_areas()
        _, member_to_community = cache.load_communities()
        team_labels = cache.load_team_labels()
    finally:
        cache.close()

    # Verify the person exists
    all_logins = set()
    for e in edges:
        all_logins.add(e.author.login)
        all_logins.add(e.reviewer.login)
    for a in areas:
        for exp in a.experts:
            all_logins.add(exp.login)

    if login not in all_logins:
        console.print(f"[red]Error:[/red] @{login} not found in the social graph.")
        close_matches = [l for l in sorted(all_logins) if login.lower() in l.lower()]
        if close_matches:
            console.print(f"  Did you mean: {', '.join(f'@{m}' for m in close_matches[:5])}?")
        raise SystemExit(1)

    report = simulate_departure(login, areas, edges, member_to_community, team_labels)
    _print_departure_report(report, top_backfills)


@simulate.command("reorg")
@click.argument("moves", nargs=-1, required=True)
@click.pass_context
def simulate_reorg_cmd(ctx: click.Context, moves: tuple[str, ...]) -> None:
    """Simulate a team reorganisation — show silo and coverage impact.

    Each MOVE is "login:team_number" (0-indexed).

    \b
    Examples:
        social-graph simulate reorg alice:1 bob:1 charlie:0
    """
    from social_graph.fetcher.cache import PRCache
    from social_graph.graph.simulate import simulate_reorg

    config: Config = ctx.obj["config"]
    cache = PRCache(config.cache_db_path)
    try:
        if not cache.has_build_data():
            console.print("[red]No build data found.[/red] Run [cyan]social-graph build[/cyan] first.")
            raise SystemExit(1)

        edges = cache.load_social_edges()
        areas = cache.load_code_areas()
        communities, member_to_community = cache.load_communities()
        team_labels = cache.load_team_labels()
    finally:
        cache.close()

    # Parse moves
    changes: list[tuple[str, int]] = []
    for move in moves:
        if ":" not in move:
            console.print(f"[red]Error:[/red] Invalid move format: {move!r}. Expected 'login:team_number'.")
            raise SystemExit(1)
        login_str, team_str = move.rsplit(":", 1)
        try:
            team_idx = int(team_str)
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid team number: {team_str!r}.")
            raise SystemExit(1)
        changes.append((login_str, team_idx))

    report = simulate_reorg(changes, areas, edges, communities, member_to_community, team_labels)
    _print_reorg_report(report, team_labels)


@simulate.command("hire")
@click.argument("area_paths", nargs=-1, required=True)
@click.pass_context
def simulate_hire_cmd(ctx: click.Context, area_paths: tuple[str, ...]) -> None:
    """Recommend team placement and onboarding buddy for a new hire.

    AREA_PATHS are code area paths (as shown in 'experts areas').

    \b
    Examples:
        social-graph simulate hire src/api src/auth
        social-graph simulate hire src/graph
    """
    from social_graph.fetcher.cache import PRCache
    from social_graph.graph.simulate import simulate_hire

    config: Config = ctx.obj["config"]
    cache = PRCache(config.cache_db_path)
    try:
        if not cache.has_build_data():
            console.print("[red]No build data found.[/red] Run [cyan]social-graph build[/cyan] first.")
            raise SystemExit(1)

        edges = cache.load_social_edges()
        areas = cache.load_code_areas()
        communities, member_to_community = cache.load_communities()
        team_labels = cache.load_team_labels()
    finally:
        cache.close()

    # Validate area paths
    known_paths = {a.path for a in areas}
    for path in area_paths:
        if path not in known_paths:
            console.print(f"[yellow]Warning:[/yellow] Area '{path}' not found in build data.")
            close = [p for p in sorted(known_paths) if path in p]
            if close:
                console.print(f"  Did you mean: {', '.join(close[:5])}?")

    report = simulate_hire(list(area_paths), areas, edges, communities, member_to_community, team_labels)
    _print_hire_report(report)


# ---------------------------------------------------------------------------
# simulate output helpers
# ---------------------------------------------------------------------------

def _print_departure_report(report, top_backfills: int) -> None:
    from rich.panel import Panel
    from rich.table import Table

    # Header
    team_count = len(report.areas_affected)
    critical = report.critical_areas
    monopolies = len(report.knowledge_monopolies)

    severity = "red" if critical > 0 else "yellow" if team_count > 0 else "green"
    header = (
        f"[bold]What if @{report.login} leaves?[/bold]\n\n"
        f"  Areas affected:       [cyan]{team_count}[/cyan]\n"
        f"  Critical (bus→0):     [{severity}]{critical}[/{severity}]\n"
        f"  Knowledge monopolies: [{'red' if monopolies else 'green'}]{monopolies}[/{'red' if monopolies else 'green'}]\n"
        f"  Org bus factor:       {report.org_bus_factor_before:.1f} → {report.org_bus_factor_after:.1f}"
    )
    console.print(Panel(header, title="Departure Impact", border_style=severity))

    if not report.areas_affected:
        console.print("[green]No code areas affected — this person has no tracked expertise.[/green]")
        return

    # Area impact table
    table = Table(title="Code Area Impact")
    table.add_column("Status", width=10)
    table.add_column("Area", style="cyan")
    table.add_column("Bus Factor", justify="center")
    table.add_column("Top Expert After")

    status_styles = {"critical": "bold red", "degraded": "yellow", "unchanged": "dim", "improved": "green"}
    status_icons = {"critical": "CRITICAL", "degraded": "DEGRADED", "unchanged": "OK", "improved": "BETTER"}

    for impact in report.areas_affected:
        style = status_styles.get(impact.status, "")
        icon = status_icons.get(impact.status, "?")
        bf_str = f"{impact.bus_factor_before} → {impact.bus_factor_after}"
        table.add_row(
            f"[{style}]{icon}[/{style}]",
            impact.area_name,
            bf_str,
            f"@{impact.top_expert_after}" if impact.top_expert_after != "—" else "[dim]none[/dim]",
        )

    console.print(table)
    console.print()

    # Backfill candidates
    if report.backfill_candidates:
        bf_table = Table(title="Backfill Candidates")
        bf_table.add_column("Area", style="cyan")
        bf_table.add_column("Candidate", style="green")
        bf_table.add_column("Score", justify="right", style="magenta")
        bf_table.add_column("Why", no_wrap=True)

        seen_areas: dict[str, int] = {}
        for c in report.backfill_candidates:
            count = seen_areas.get(c.area_path, 0)
            if count >= top_backfills:
                continue
            seen_areas[c.area_path] = count + 1
            bf_table.add_row(
                c.area_name if count == 0 else "",
                f"@{c.login}",
                f"{c.combined_score:.2f}",
                c.reason,
            )

        console.print(bf_table)
        console.print()

    # Pairing sessions
    if report.pairing_sessions:
        ps_table = Table(title="Recommended Knowledge Transfer Sessions")
        ps_table.add_column("Urgency", width=10)
        ps_table.add_column("Area", style="cyan")
        ps_table.add_column("Mentor → Mentee")
        ps_table.add_column("Reason")

        urgency_styles = {"critical": "bold red", "high": "yellow", "medium": "dim"}
        for ps in report.pairing_sessions:
            style = urgency_styles.get(ps.urgency, "")
            ps_table.add_row(
                f"[{style}]{ps.urgency.upper()}[/{style}]",
                ps.area_name,
                f"@{ps.mentor} → @{ps.mentee}",
                ps.reason,
            )

        console.print(ps_table)

    if report.knowledge_monopolies:
        console.print()
        console.print("[bold red]Knowledge Monopolies[/bold red] — these areas have NO other expert:")
        for path in report.knowledge_monopolies:
            console.print(f"  [red]!!![/red] {path}")


def _print_reorg_report(report, team_labels: dict) -> None:
    from rich.panel import Panel
    from rich.table import Table

    # Moves summary
    moves_str = "\n".join(
        f"  @{c.login}: {c.from_team} → {c.to_team}"
        for c in report.changes
    )
    silo_delta = report.silo_score_after - report.silo_score_before
    silo_color = "red" if silo_delta > 0 else "green" if silo_delta < 0 else "dim"

    header = (
        f"[bold]Reorg Simulation[/bold]\n\n"
        f"{moves_str}\n\n"
        f"  Cross-team edges:  {report.cross_team_edges_before} → {report.cross_team_edges_after}\n"
        f"  Silo score:        [{silo_color}]{report.silo_score_before:.0%} → {report.silo_score_after:.0%}[/{silo_color}]"
    )
    border = "red" if report.warnings else "green"
    console.print(Panel(header, title="Reorg Impact", border_style=border))

    if report.areas_affected:
        table = Table(title="Area Team Diversity Changes")
        table.add_column("Status", width=10)
        table.add_column("Area", style="cyan")
        table.add_column("Teams Covering", justify="center")

        for impact in report.areas_affected:
            style = {"improved": "green", "degraded": "yellow", "unchanged": "dim"}.get(impact.status, "")
            icon = {"improved": "BETTER", "degraded": "WORSE", "unchanged": "SAME"}.get(impact.status, "?")
            table.add_row(
                f"[{style}]{icon}[/{style}]",
                impact.area_name,
                f"{impact.teams_before} → {impact.teams_after}",
            )
        console.print(table)

    if report.warnings:
        console.print()
        console.print("[bold yellow]Warnings[/bold yellow]")
        for w in report.warnings:
            console.print(f"  [yellow]![/yellow] {w}")


def _print_hire_report(report) -> None:
    from rich.table import Table

    if not report.placements:
        console.print("[yellow]No matching code areas found for the given paths.[/yellow]")
        return

    table = Table(title="New Hire Placement Recommendations")
    table.add_column("Area", style="cyan")
    table.add_column("Team", style="green")
    table.add_column("Buddy", style="cyan")
    table.add_column("Bus Factor", justify="center")
    table.add_column("Why", no_wrap=True)

    for p in report.placements:
        table.add_row(
            p.target_area_name,
            p.recommended_team,
            f"@{p.onboarding_buddy} ({p.buddy_expertise:.0%})",
            p.bus_factor_improvement,
            p.reason,
        )

    console.print(table)

    if report.org_bus_factor_improvement > 0:
        console.print(
            f"\n[green]Expected org-wide bus factor improvement: "
            f"+{report.org_bus_factor_improvement:.1f}[/green]"
        )


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@cli.command("export")
@click.option("--format", "fmt", type=click.Choice(["json", "graphml", "csv"]), default="json")
@click.option("--output", "-o", default=None, help="Output file path (or directory for CSV).")
@click.option("--min-pr-rate", default=DEFAULT_MIN_PR_RATE, type=float, help="Minimum avg PRs per month for an author/reviewer.")
@click.pass_context
def export_graph(ctx: click.Context, fmt: str, output: str | None, min_pr_rate: float) -> None:
    """Export the social graph to JSON, GraphML, or CSV."""
    config: Config = ctx.obj["config"]
    config.min_pr_rate = min_pr_rate

    from social_graph.fetcher.cache import PRCache
    from social_graph.graph.social import build_social_graph
    from social_graph.output.json_export import export_csv, export_graphml, export_json

    cache = PRCache(config.cache_db_path)
    try:
        prs = cache.get_all_prs()
        edges = build_social_graph(prs, min_pr_rate=config.min_pr_rate)

        if fmt == "csv":
            out_dir = Path(output or "social-graph-data")
            areas = cache.load_code_areas()
            communities, member_to_community = cache.load_communities()
            team_labels = cache.load_team_labels()
            export_csv(edges, areas, communities, team_labels, member_to_community, out_dir)
            console.print(
                f"Exported {len(edges)} edges, {len(areas)} areas, "
                f"{len(communities)} teams to [green]{out_dir}/[/green]"
            )
        else:
            if output is None:
                output = f"social-graph.{fmt}"
            out_path = Path(output)

            if fmt == "json":
                export_json(edges, out_path)
            else:
                export_graphml(edges, out_path)

            console.print(f"Exported {len(edges)} edges to [green]{out_path}[/green]")
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# build / viz / run — core pipeline
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--org", default=None, help="GitHub organization (auto-detected from git remote).")
@click.option("--repo", multiple=True, help="Repository name (auto-detected from git remote).")
@click.option("--repo-path", default=".", help="Path to local git repo clone (default: current directory).")
@click.option("--since-prs", default=None, help="Fetch PRs updated since this date (e.g. '2025-01-01').")
@click.option("--since-git", default="6 months ago", help="Git log lookback period.")
@click.option("--limit", default=None, type=int, help="Max PRs to fetch.")
@click.option("--fresh", is_flag=True, help="Ignore cache timestamps, re-fetch everything.")
@click.option("--skip-fetch", is_flag=True, help="Skip PR fetching, rebuild from cached data only.")
@click.option("--min-pr-rate", default=DEFAULT_MIN_PR_RATE, type=float, help="Minimum avg PRs per month for an author/reviewer.")
@click.pass_context
def build(ctx: click.Context, org: str | None, repo: tuple[str, ...], repo_path: str, since_prs: str | None, since_git: str, limit: int | None, fresh: bool, skip_fetch: bool, min_pr_rate: float) -> None:
    """Fetch PR data, compute social graph, detect teams."""
    _run_build(ctx, org, repo, repo_path, since_prs, since_git, limit, skip_fetch, fresh, min_pr_rate)


@cli.command()
@click.option("--output", "-o", default=None, help="Output HTML file path.")
@click.pass_context
def viz(ctx: click.Context, output: str | None) -> None:
    """Render visualization from pre-built data and open in browser."""
    _run_viz(ctx, output)


@cli.command()
@click.option("--org", default=None, help="GitHub organization (auto-detected from git remote).")
@click.option("--repo", multiple=True, help="Repository name (auto-detected from git remote).")
@click.option("--repo-path", default=".", help="Path to local git repo clone (default: current directory).")
@click.option("--since-prs", default=None, help="Fetch PRs updated since this date (e.g. '2025-01-01').")
@click.option("--since-git", default="6 months ago", help="Git log lookback period.")
@click.option("--limit", default=None, type=int, help="Max PRs to fetch.")
@click.option("--fresh", is_flag=True, help="Ignore cache timestamps, re-fetch everything.")
@click.option("--skip-fetch", is_flag=True, help="Skip PR fetching, rebuild from cached data only.")
@click.option("--min-pr-rate", default=DEFAULT_MIN_PR_RATE, type=float, help="Minimum avg PRs per month for an author/reviewer.")
@click.option("--output", "-o", default=None, help="Output HTML file path.")
@click.pass_context
def run(ctx: click.Context, org: str | None, repo: tuple[str, ...], repo_path: str, since_prs: str | None, since_git: str, limit: int | None, fresh: bool, skip_fetch: bool, min_pr_rate: float, output: str | None) -> None:
    """Build the social graph and open the visualization."""
    _run_build(ctx, org, repo, repo_path, since_prs, since_git, limit, skip_fetch, fresh, min_pr_rate)
    console.print()
    _run_viz(ctx, output)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_org_repo(repo_path: str, org: str | None, repo: tuple[str, ...]) -> tuple[str, list[str]]:
    """Resolve org and repo names — from explicit flags or the git remote."""
    if org and repo:
        return org, list(repo)
    try:
        remote_org, remote_repo = resolve_github_remote(repo_path)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Specify --org and --repo explicitly, or run from inside a GitHub clone.")
        raise SystemExit(1)
    resolved_org = org or remote_org
    resolved_repos = list(repo) if repo else [remote_repo]
    console.print(f"Detected GitHub repo: [cyan]{resolved_org}/{'/'.join(resolved_repos)}[/cyan]")
    return resolved_org, resolved_repos


def _run_build(ctx: click.Context, org: str | None, repo: tuple[str, ...], repo_path: str, since_prs: str | None, since_git: str, limit: int | None, skip_fetch: bool = False, fresh: bool = False, min_pr_rate: float = 7.0) -> None:
    if fresh and skip_fetch:
        console.print("[red]Error:[/red] --fresh and --skip-fetch are mutually exclusive.")
        raise SystemExit(1)

    config: Config = ctx.obj["config"]
    if not config.anthropic_api_key:
        console.print("[yellow]Note:[/yellow] No Anthropic API key — team labels will use code area names instead of AI.")
        console.print("  Run [cyan]social-graph auth --anthropic KEY[/cyan] to enable AI labels.\n")
    if not skip_fetch and not config.github_token:
        console.print("[red]Error:[/red] No GitHub token found.")
        console.print("Run [cyan]social-graph auth --github TOKEN[/cyan] or authenticate with [cyan]gh auth login[/cyan].")
        raise SystemExit(1)

    resolved_org, resolved_repos = _resolve_org_repo(repo_path, org, repo)

    from social_graph.builder import run_build

    run_build(
        config=config,
        org=resolved_org,
        repos=resolved_repos,
        repo_path=repo_path,
        since_prs=since_prs,
        since_git=since_git,
        limit=limit,
        skip_fetch=skip_fetch,
        fresh=fresh,
        min_pr_rate=min_pr_rate,
        log=console.print,
    )


def _run_viz(ctx: click.Context, output: str | None) -> None:
    config: Config = ctx.obj["config"]

    from social_graph.fetcher.cache import PRCache
    from social_graph.output.grid import open_grid_html

    cache = PRCache(config.cache_db_path)
    try:
        if not cache.has_build_data():
            console.print("[red]No build data found.[/red] Run [cyan]social-graph build[/cyan] first.")
            raise SystemExit(1)

        meta = cache.load_build_meta()
        console.print(
            f"Loading build data: {meta.get('edge_count', '?')} edges, "
            f"{meta.get('area_count', '?')} areas, "
            f"built {meta.get('built_at', 'unknown')[:10]}"
        )

        edges = cache.load_social_edges()
        lines_by_user = cache.load_lines_by_user()
        areas = cache.load_code_areas()
        team_labels = cache.load_team_labels()

        console.print(f"Rendering graph with {len(edges)} edges...")
        out_path = Path(output) if output else None
        open_grid_html(edges, out_path, lines_by_user, team_labels=team_labels, areas=areas)
    finally:
        cache.close()


if __name__ == "__main__":
    cli()
