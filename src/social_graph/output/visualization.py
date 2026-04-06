"""Social graph data builders — community detection, LOC computation, graph data.

Provides the algorithmic core used by grid.py for rendering and builder.py for
the build pipeline. No HTML rendering — that's in grid.py.
"""

from __future__ import annotations

import math
from collections import defaultdict

from social_graph.models import SocialEdge


def _adaptive_resolution(g, target_community_size: float = 4.0) -> float:
    """Compute Louvain resolution parameter adapted to graph size.

    Standard modularity (gamma=1) has a resolution limit that prevents detection
    of communities smaller than ~sqrt(2m) edges (Fortunato & Barthelemy, 2007).
    We scale gamma proportionally to graph size so that communities of the target
    size are reliably detected.

    Reference: S. Fortunato and M. Barthelemy, "Resolution limit in community
    detection," PNAS 104(1), 2007. arXiv:physics/0607100
    """
    n = g.number_of_nodes()
    if n <= 2:
        return 1.0
    return max(1.0, n / (target_community_size * 2.5))


def detect_communities(edges: list[SocialEdge]) -> tuple[list[set[str]], dict[str, int]]:
    """Detect communities using Louvain with mutual-review bonus.

    Mutual review edges (A reviews B AND B reviews A) get a 3x weight boost
    because mutual reviewers are more likely on the same team.

    The resolution parameter is adapted to graph size following Fortunato &
    Barthelemy (2007) to avoid the modularity resolution limit.
    """
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    edge_lookup: dict[tuple[str, str], float] = {}
    for e in edges:
        edge_lookup[(e.author.login, e.reviewer.login)] = e.weight

    g = nx.Graph()
    seen: set[tuple[str, str]] = set()
    for e in edges:
        a, r = e.author.login, e.reviewer.login
        key = tuple(sorted([a, r]))
        if key in seen:
            continue
        seen.add(key)
        w_ar = edge_lookup.get((a, r), 0)
        w_ra = edge_lookup.get((r, a), 0)
        if w_ar > 0 and w_ra > 0:
            # Geometric mean rewards balanced mutual relationships — strong same-team signal.
            # Two people who equally review each other (0.5/0.5) score higher than
            # an imbalanced pair (1.0/0.1) despite similar raw sums.
            combined = math.sqrt(w_ar * w_ra) * 3.0
        else:
            # One-way review relationships are weaker team signals
            combined = (w_ar + w_ra) * 0.5
        g.add_edge(a, r, weight=combined)

    resolution = _adaptive_resolution(g)
    communities = list(louvain_communities(g, weight="weight", resolution=resolution, seed=42))
    member_to_community: dict[str, int] = {}
    for i, community in enumerate(communities):
        for member in community:
            member_to_community[member] = i

    return communities, member_to_community


_CLUSTER_COLORS = [
    "#4e9ee8", "#50c878", "#e8a84e", "#e84e4e", "#9b59b6",
    "#1abc9c", "#e67e22", "#3498db", "#e74c3c", "#2ecc71",
]


def compute_lines_from_git(
    repo_path: str,
    since: str = "6 months ago",
    known_users: dict[str, str] | None = None,
) -> dict[str, int]:
    """Compute lines added+deleted per author using git log --numstat.

    This is far more accurate than PR-level stats because it covers all commits
    and doesn't depend on the limited PR cache window.

    Handles GitHub noreply emails like "12345+jdoe@users.noreply.github.com"
    by extracting the login portion and merging with plain email variants.

    Args:
        known_users: Optional mapping of {name_or_email_prefix: github_login}
            built from PR data, used to resolve git authors whose email prefix
            doesn't match their GitHub login.
    """
    import subprocess
    result = subprocess.run(
        ["git", "-C", repo_path, "log", "--numstat", "--format=%aN%n%aE", f"--since={since}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}

    known_users = known_users or {}

    raw_stats: dict[str, int] = defaultdict(int)
    current_key = ""

    # Format outputs: author name, author email, then numstat lines.
    # We read name+email pairs, then resolve to the best key.
    pending_name = ""
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" not in line:
            if "@" in line:
                # This is the email line
                email_prefix = line.split("@")[0]
                # GitHub noreply format: "12345+username"
                if "+" in email_prefix:
                    email_prefix = email_prefix.split("+", 1)[1]
                # Resolve to GitHub login: try email prefix first, then git author name
                if email_prefix in known_users:
                    current_key = known_users[email_prefix]
                elif pending_name in known_users:
                    current_key = known_users[pending_name]
                else:
                    current_key = email_prefix
                pending_name = ""
            else:
                # This is the name line
                pending_name = line
        else:
            parts = line.split("\t")
            if len(parts) == 3 and current_key:
                added, deleted, _ = parts
                if added != "-" and deleted != "-":
                    raw_stats[current_key] += int(added) + int(deleted)

    # Merge any remaining aliases
    merged: dict[str, int] = defaultdict(int)
    for key, val in raw_stats.items():
        merged[key] += val

    return dict(merged)


def _build_graph_data(
    edges: list[SocialEdge],
    lines_by_user: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build sigma-compatible node/edge JSON from social edges."""
    degree: dict[str, float] = defaultdict(float)
    author_of: dict[str, dict[str, float]] = defaultdict(dict)
    reviewer_of: dict[str, dict[str, float]] = defaultdict(dict)
    names: dict[str, str] = {}
    users: dict[str, object] = {}  # login -> User object

    for e in edges:
        a, r = e.author.login, e.reviewer.login
        degree[a] += e.weight
        degree[r] += e.weight
        author_of[a][r] = e.weight
        reviewer_of[r][a] = e.weight
        names[a] = e.author.display_name
        names[r] = e.reviewer.display_name
        users[a] = e.author
        users[r] = e.reviewer

    # Community detection — Louvain with mutual-review bonus.
    communities, member_to_community = detect_communities(edges)

    # Node sizing by lines of code — sqrt scale normalized to max contributor.
    # sqrt gives visible proportionality: 10x more lines ≈ 3x larger node.
    max_loc = max(lines_by_user.values()) if lines_by_user else 1

    # Build nodes — size by lines of code committed
    nodes_json = []
    for login in sorted(degree.keys()):
        d = degree[login]
        community_idx = member_to_community.get(login, 0)
        color = _CLUSTER_COLORS[community_idx % len(_CLUSTER_COLORS)]
        loc = lines_by_user.get(login, 0) if lines_by_user else 0
        if loc > 0 and max_loc > 0:
            norm = loc / max_loc
            size = 8 + math.sqrt(norm) * 30
        else:
            size = 8

        reviews_given = reviewer_of.get(login, {})
        reviews_received = author_of.get(login, {})
        user = users.get(login)

        nodes_json.append({
            "id": login,
            "label": names.get(login, login),
            "color": color,
            "size": round(size, 1),
            "community": community_idx,
            "degree": round(d, 2),
            "linesOfCode": loc,
            "avatarUrl": user.avatar_url if user else None,
            "htmlUrl": user.html_url if user else None,
            "company": user.company if user else None,
            "location": user.location if user else None,
            "login": login,
            "reviewsGiven": {names.get(k, k): round(v, 3) for k, v in sorted(reviews_given.items(), key=lambda x: -x[1])},
            "reviewsReceived": {names.get(k, k): round(v, 3) for k, v in sorted(reviews_received.items(), key=lambda x: -x[1])},
        })

    # Build undirected edges — merge bidirectional into one edge,
    # but encode the directional imbalance as color tint
    seen: set[tuple[str, str]] = set()
    edges_json = []
    edge_lookup: dict[tuple[str, str], float] = {}
    for e in edges:
        edge_lookup[(e.author.login, e.reviewer.login)] = e.weight

    for e in edges:
        a, r = e.author.login, e.reviewer.login
        key = tuple(sorted([a, r]))
        if key in seen:
            continue
        seen.add(key)

        w_ar = edge_lookup.get((a, r), 0)  # a authored, r reviewed
        w_ra = edge_lookup.get((r, a), 0)  # r authored, a reviewed
        combined = max(w_ar, w_ra)
        # Imbalance: who reviews whom more? 0.5 = balanced, 0 or 1 = one-sided
        total = w_ar + w_ra
        balance = w_ar / total if total > 0 else 0.5

        edges_json.append({
            "source": key[0],
            "target": key[1],
            "weight": round(combined, 3),
            "w_forward": round(w_ar, 3),
            "w_reverse": round(w_ra, 3),
            "balance": round(balance, 2),
            "mutual": w_ar > 0 and w_ra > 0,
            "label": f"{combined:.2f}",
        })

    return nodes_json, edges_json
