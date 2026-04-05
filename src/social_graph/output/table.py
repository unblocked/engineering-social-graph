"""Rich terminal table output."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from social_graph.models import ExpertRanking, SocialEdge


def print_social_edges(edges: list[SocialEdge], top_n: int = 20) -> None:
    """Print the strongest social graph edges as a Rich table."""
    console = Console()
    table = Table(title="Social Graph — Strongest Collaboration Edges")
    table.add_column("Author", style="cyan")
    table.add_column("Reviewer", style="green")
    table.add_column("Weight", justify="right", style="magenta")

    sorted_edges = sorted(edges, key=lambda e: e.weight, reverse=True)[:top_n]
    for edge in sorted_edges:
        author_label = edge.author.name or edge.author.login
        reviewer_label = edge.reviewer.name or edge.reviewer.login
        table.add_row(author_label, reviewer_label, f"{edge.weight:.3f}")

    console.print(table)


def print_expert_rankings(rankings: list[ExpertRanking], title: str | None = None) -> None:
    """Print expert rankings as a Rich table."""
    console = Console()
    table = Table(title=title or f"Experts for {rankings[0].domain}" if rankings else "Experts")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Score", justify="right", style="magenta")

    for i, ranking in enumerate(rankings, 1):
        name = ranking.user.name or ranking.user.login
        table.add_row(str(i), name, f"{ranking.score:.2f}")

    console.print(table)


def print_author_summary(edges: list[SocialEdge], top_n: int = 10) -> None:
    """Print per-author summary: how many reviewers, total edge weight."""
    console = Console()
    table = Table(title="Author Summary — Most Connected Authors")
    table.add_column("Author", style="cyan")
    table.add_column("Reviewers", justify="right")
    table.add_column("Avg Weight", justify="right", style="magenta")

    from collections import defaultdict

    author_stats: dict[str, list[float]] = defaultdict(list)
    author_names: dict[str, str] = {}
    for edge in edges:
        key = edge.author.login
        author_stats[key].append(edge.weight)
        author_names[key] = edge.author.name or edge.author.login

    ranked = sorted(author_stats.items(), key=lambda x: len(x[1]), reverse=True)[:top_n]
    for login, weights in ranked:
        avg = sum(weights) / len(weights)
        table.add_row(author_names[login], str(len(weights)), f"{avg:.3f}")

    console.print(table)
