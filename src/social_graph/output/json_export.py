"""JSON and GraphML export for social graphs."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from social_graph.graph.social import edges_to_networkx
from social_graph.models import SocialEdge


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
