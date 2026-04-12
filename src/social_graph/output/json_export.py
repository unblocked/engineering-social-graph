"""JSON, GraphML, and CSV export for social graphs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
from networkx.readwrite import json_graph

from social_graph.graph.social import edges_to_networkx
from social_graph.models import SocialEdge

if TYPE_CHECKING:
    from social_graph.graph.code_expertise import CodeArea


def export_json(edges: list[SocialEdge], path: Path) -> None:
    """Export social graph as JSON (node-link format)."""
    g = edges_to_networkx(edges)
    data = json_graph.node_link_data(g)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def export_graphml(edges: list[SocialEdge], path: Path) -> None:
    """Export social graph as GraphML."""
    g = edges_to_networkx(edges)
    nx.write_graphml(g, str(path))


def export_csv(
    edges: list[SocialEdge],
    areas: list[CodeArea],
    communities: list[set[str]],
    team_labels: dict[int, str],
    member_to_community: dict[str, int],
    output_dir: Path,
) -> None:
    """Export social graph data as CSV files in *output_dir*.

    Creates:
      - edges.csv   — author, reviewer, weight, team_author, team_reviewer
      - experts.csv — area, login, score, commits, lines, files_touched, bus_factor
      - teams.csv   — team_id, label, members, top_areas
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- edges.csv ---
    with open(output_dir / "edges.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["author", "reviewer", "weight", "team_author", "team_reviewer"])
        for e in edges:
            w.writerow([
                e.author.login,
                e.reviewer.login,
                round(e.weight, 4),
                team_labels.get(member_to_community.get(e.author.login, -1), ""),
                team_labels.get(member_to_community.get(e.reviewer.login, -1), ""),
            ])

    # --- experts.csv ---
    with open(output_dir / "experts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["area", "login", "score", "commits", "lines", "files_touched", "bus_factor"])
        for area in areas:
            for expert in area.experts:
                w.writerow([
                    area.name,
                    expert.login,
                    round(expert.score, 4),
                    expert.commits,
                    expert.lines,
                    expert.files_touched,
                    area.bus_factor,
                ])

    # --- teams.csv ---
    # Build top areas per team from expert data
    team_top_areas: dict[int, list[str]] = {}
    for area in areas:
        for expert in area.experts:
            team_idx = member_to_community.get(expert.login, -1)
            if team_idx >= 0:
                team_top_areas.setdefault(team_idx, [])
                if area.name not in team_top_areas[team_idx]:
                    team_top_areas[team_idx].append(area.name)

    with open(output_dir / "teams.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team_id", "label", "members", "top_areas"])
        for idx, members in enumerate(communities):
            label = team_labels.get(idx, f"Team {idx + 1}")
            sorted_members = sorted(members)
            top = team_top_areas.get(idx, [])[:5]
            w.writerow([
                idx,
                label,
                ";".join(sorted_members),
                ";".join(top),
            ])
