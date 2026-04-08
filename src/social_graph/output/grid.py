"""Social network grid + peer tables + interactive Sigma.js graph.

1. Heat map grid: Reviewers / Authors with rgba(0,187,140,weight) cells
2. Per-reviewer peer tables with progress bars
3. Interactive Sigma.js force-directed graph
"""

from __future__ import annotations

import html
import json
import webbrowser
from collections import defaultdict
from pathlib import Path

from social_graph.config import DEFAULT_MODEL
from social_graph.models import SocialEdge


def generate_grid_html(
    edges: list[SocialEdge],
    output_path: Path | None = None,
    lines_by_user: dict[str, int] | None = None,
    team_labels: dict[int, str] | None = None,
    areas=None,
) -> Path:
    """Generate the social network HTML visualization and return its path."""
    if output_path is None:
        output_path = Path.home() / ".social-graph" / "social-graph.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build lookup structures
    authors_of: dict[str, dict[str, float]] = defaultdict(dict)  # author -> {reviewer: weight}
    names: dict[str, str] = {}
    user_data: dict[str, object] = {}  # login -> User object

    for e in edges:
        a = e.author.login
        r = e.reviewer.login
        authors_of[a][r] = e.weight
        names[a] = e.author.display_name
        names[r] = e.reviewer.display_name
        user_data[a] = e.author
        user_data[r] = e.reviewer

    all_users = sorted(names.keys())

    # Build sections
    grid_html = _build_grid(all_users, authors_of, names, user_data)
    peers_html = _build_peer_tables(all_users, authors_of, names, edges, user_data)
    team_labels = team_labels or {}

    teams_html = _build_teams(edges, user_data, names, lines_by_user, areas, team_labels)
    experts_html = _build_experts(areas) if areas else ""
    sigma_html = _build_sigma(edges, lines_by_user, team_labels)

    page = _PAGE_TEMPLATE.format(
        grid=grid_html,
        peers=peers_html,
        teams=teams_html,
        experts=experts_html,
        sigma=sigma_html,
    )
    output_path.write_text(page)
    return output_path


def open_grid_html(
    edges: list[SocialEdge],
    output_path: Path | None = None,
    lines_by_user: dict[str, int] | None = None,
    team_labels: dict[int, str] | None = None,
    areas=None,
) -> None:
    """Generate and open the grid HTML in the default browser."""
    path = generate_grid_html(edges, output_path, lines_by_user, team_labels, areas)
    webbrowser.open(f"file://{path}")


def _esc(s: str) -> str:
    return html.escape(s)


def _avatar_tag(user, size: int = 24) -> str:
    if user and user.avatar_url:
        return f'<img src="{_esc(user.avatar_url)}" width="{size}" height="{size}" class="avatar" title="{_esc(user.display_name)}" /> '
    return ""


def _build_grid(users: list[str], authors_of: dict[str, dict[str, float]], names: dict[str, str], user_data: dict) -> str:
    """Reviewers / Authors heat map grid."""
    rows: list[str] = []

    # Header row: corner cell + author columns
    header_cells = ['<td class="corner">Reviewers ↓ / Authors →</td>']
    for author in users:
        u = user_data.get(author)
        avatar = _avatar_tag(u, 22)
        header_cells.append(f'<td class="header-cell" title="{_esc(names[author])}">{avatar}{_esc(names[author][:12])}</td>')
    rows.append(f'<tr>{"".join(header_cells)}</tr>')

    # Data rows: one per reviewer
    for reviewer in users:
        u = user_data.get(reviewer)
        avatar = _avatar_tag(u, 22)
        cells = [f'<td class="row-label" title="{_esc(names[reviewer])}">{avatar}{_esc(names[reviewer][:12])}</td>']
        for author in users:
            weight = authors_of.get(author, {}).get(reviewer, 0.0)
            if weight < 0.1:
                cells.append('<td class="cell empty"></td>')
            else:
                cells.append(
                    f'<td class="cell" style="background-color: rgba(0, 187, 140, {weight:.2f});" '
                    f'title="{_esc(names[reviewer])} → {_esc(names[author])}: {weight:.3f}">'
                    f'{weight:.2f}</td>'
                )
        rows.append(f'<tr>{"".join(cells)}</tr>')

    return f'<table class="grid-table"><thead>{rows[0]}</thead><tbody>{"".join(rows[1:])}</tbody></table>'


def _build_peer_tables(
    users: list[str],
    authors_of: dict[str, dict[str, float]],
    names: dict[str, str],
    edges: list[SocialEdge],
    user_data: dict,
) -> str:
    """Per-reviewer peer tables with progress bars."""
    # Group by reviewer
    reviewer_authors: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e in edges:
        reviewer_authors[e.reviewer.login].append((e.author.login, e.weight))

    sections: list[str] = []
    for reviewer in users:
        peers = reviewer_authors.get(reviewer, [])
        if not peers:
            continue
        peers.sort(key=lambda x: -x[1])

        rows: list[str] = []
        for author_login, weight in peers:
            pct = int(weight * 100)
            bar_color = '#50c878' if weight > 0.7 else '#4e9ee8' if weight > 0.3 else '#8b949e'
            author_u = user_data.get(author_login)
            avatar = _avatar_tag(author_u, 20)
            rows.append(
                f'<tr>'
                f'<td class="peer-name">{avatar}{_esc(names[author_login])}</td>'
                f'<td class="peer-weight">'
                f'<span class="weight-num">{weight:.3f}</span>'
                f'<span class="bar-bg"><span class="bar-fill" style="width:{pct}%;background:{bar_color}"></span></span>'
                f'</td>'
                f'</tr>'
            )

        reviewer_u = user_data.get(reviewer)
        reviewer_avatar = _avatar_tag(reviewer_u, 24)
        sections.append(
            f'<div class="peer-section">'
            f'<h3>{reviewer_avatar}Peers for {_esc(names[reviewer])}</h3>'
            f'<table class="peer-table"><thead><tr>'
            f'<th>Peer (Author)</th><th>Relevance (0.0 to 1.0)</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
            f'</div>'
        )

    return "".join(sections)


def _build_teams(edges: list[SocialEdge], user_data: dict, names: dict[str, str], lines_by_user: dict[str, int] | None = None, areas=None, team_labels: dict[int, str] | None = None) -> str:
    """Build team cards grouped by detected community clusters."""
    from social_graph.output.visualization import _CLUSTER_COLORS, detect_communities
    from collections import defaultdict

    communities, member_to_community = detect_communities(edges)

    # Compute per-person stats
    edge_lookup: dict[tuple[str, str], float] = {}
    for e in edges:
        edge_lookup[(e.author.login, e.reviewer.login)] = e.weight

    authored_count: dict[str, int] = defaultdict(int)
    reviewed_count: dict[str, int] = defaultdict(int)
    for e in edges:
        authored_count[e.author.login] += 1
        reviewed_count[e.reviewer.login] += 1

    if team_labels is None:
        team_labels = _derive_team_labels(communities, areas, names) if areas else {}

    cards: list[str] = []
    for i, community in enumerate(communities):
        color = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
        members = sorted(community, key=lambda m: -(authored_count.get(m, 0) + reviewed_count.get(m, 0)))

        member_rows: list[str] = []
        for login in members:
            user = user_data.get(login)
            avatar = _avatar_tag(user, 36) if user else ""
            name = _esc(names.get(login, login))
            gh_link = f'<a href="{_esc(user.html_url)}" target="_blank">@{_esc(login)}</a>' if user and user.html_url else f'@{_esc(login)}'

            meta_parts: list[str] = []
            if user and user.company:
                meta_parts.append(_esc(user.company))
            if user and user.location:
                meta_parts.append(_esc(user.location))
            meta = " · ".join(meta_parts)

            authored = authored_count.get(login, 0)
            reviewed = reviewed_count.get(login, 0)
            loc = lines_by_user.get(login, 0) if lines_by_user else 0
            loc_str = f'{loc:,} lines' if loc else ''

            # Find top collaborators within this cluster
            collabs: list[str] = []
            for other in members:
                if other == login:
                    continue
                w_out = edge_lookup.get((login, other), 0)
                w_in = edge_lookup.get((other, login), 0)
                if w_out > 0 or w_in > 0:
                    collabs.append(f'{_esc(names.get(other, other))} ({max(w_out, w_in):.2f})')

            collab_display = ", ".join(collabs[:3])
            if len(collabs) > 3:
                collab_display += f" +{len(collabs) - 3} more"
            collab_text = collab_display if collabs else "\u2014"

            meta_html = f'<div class="tm-meta">{meta}</div>' if meta else ''
            member_rows.append(
                f'<div class="team-member">'
                f'<div class="tm-header">{avatar}'
                f'<div class="tm-info">'
                f'<div class="tm-name">{name}</div>'
                f'<div class="tm-login">{gh_link}</div>'
                f'</div></div>'
                f'{meta_html}'
                f'<div class="tm-stats">'
                f'<span>Authors {authored} edges</span>'
                f'<span>Reviews {reviewed} edges</span>'
                f'{"<span>" + loc_str + "</span>" if loc_str else ""}'
                f'</div>'
                f'<div class="tm-collabs">Collaborates with: {collab_text}</div>'
                f'</div>'
            )

        label = team_labels.get(i, f"Team {i + 1}")
        cards.append(
            f'<div class="team-card stagger-item">'
            f'<div class="team-header" style="border-left: 4px solid {color};">'
            f'<h3>{_esc(label)}</h3>'
            f'<span class="team-count">{len(members)} members</span>'
            f'</div>'
            f'{"".join(member_rows)}'
            f'</div>'
        )

    return "".join(cards)


def _derive_team_labels(communities: list[set[str]], areas, names: dict[str, str], api_key: str | None = None) -> dict[int, str]:
    """Derive team labels using an LLM to interpret code area expertise.

    For each team, computes the most distinctive code areas, then asks
    Claude Haiku to produce a concise team name from those areas + member names.
    Falls back to raw area names if no API key is available.

    Args:
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
    """
    if not areas:
        return {}

    from collections import defaultdict
    import os

    # First, compute distinctive areas per team
    team_area_data: dict[int, list[tuple[str, str, float]]] = {}  # idx -> [(area_name, area_path, distinctiveness)]

    for team_idx, members in enumerate(communities):
        team_area_scores: dict[str, float] = defaultdict(float)
        other_area_scores: dict[str, float] = defaultdict(float)
        area_paths: dict[str, str] = {}

        for area in areas:
            area_paths[area.name] = area.path
            for expert in area.experts:
                if expert.login in members:
                    team_area_scores[area.name] += expert.score
                else:
                    other_area_scores[area.name] += expert.score

        distinctive: list[tuple[str, str, float]] = []
        skip_names = {"projects", "libs", "services", "clients", "src", "models", "common"}
        for area_name, team_score in team_area_scores.items():
            if area_name.lower() in skip_names:
                continue
            other_score = other_area_scores.get(area_name, 0)
            total = team_score + other_score
            if total > 0 and team_score > 0.3:
                distinctiveness = team_score / total
                distinctive.append((area_name, area_paths.get(area_name, ""), distinctiveness))

        distinctive.sort(key=lambda x: -x[2])
        team_area_data[team_idx] = distinctive[:10]

    # Try LLM labeling
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        return _llm_team_labels(communities, team_area_data, names, api_key)

    # Fallback: raw area names
    labels: dict[int, str] = {}
    for team_idx in range(len(communities)):
        top = team_area_data.get(team_idx, [])
        if top:
            labels[team_idx] = " · ".join(name for name, _, _ in top[:3])
        else:
            labels[team_idx] = f"Team {team_idx + 1}"
    return labels


def _llm_team_labels(
    communities: list[set[str]],
    team_area_data: dict[int, list[tuple[str, str, float]]],
    names: dict[str, str],
    api_key: str,
) -> dict[int, str]:
    """Use Claude Haiku to generate concise team labels."""
    try:
        import anthropic
    except ImportError:
        return {}

    # Build a single prompt with all teams
    team_descriptions: list[str] = []
    for team_idx, members in enumerate(communities):
        member_names = [names.get(m, m) for m in sorted(members)]
        top_areas = team_area_data.get(team_idx, [])
        areas_str = ", ".join(
            f"{name} ({path}, {dist:.0%} distinctive)"
            for name, path, dist in top_areas[:8]
        )
        team_descriptions.append(
            f"Team {team_idx + 1}: Members: {', '.join(member_names)}. "
            f"Top code areas: {areas_str or 'unknown'}"
        )

    prompt = (
        "Given these software teams and the code areas they work on most, "
        "give each team a short, descriptive name (2-4 words) that captures what they do. "
        "Return ONLY a JSON object mapping team number to name, like: "
        '{"1": "Backend Platform", "2": "Frontend UI"}\n\n'
        + "\n".join(team_descriptions)
    )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text
    # Extract JSON
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    import json as json_mod
    raw = json_mod.loads(text)
    return {int(k) - 1: v for k, v in raw.items()}  # convert 1-indexed to 0-indexed


def _build_experts(areas) -> str:
    """Build the experts tab content — area cards with ranked contributors and bus factor."""
    if not areas:
        return '<p style="color:#8b949e;">No areas found with enough activity.</p>'

    cards: list[str] = []
    for area in areas[:30]:
        # Bus factor color — how many people could leave before knowledge is lost
        if area.bus_factor <= 1:
            bf_color = "#e84e4e"
            bf_label = "Single owner"
        elif area.bus_factor <= 2:
            bf_color = "#e8a84e"
            bf_label = "At risk"
        elif area.bus_factor <= 4:
            bf_color = "#4e9ee8"
            bf_label = "Shared"
        else:
            bf_color = "#50c878"
            bf_label = "Well covered"

        # Expert rows
        expert_rows: list[str] = []
        for i, expert in enumerate(area.experts[:5]):
            score_pct = int(expert.score * 100)
            bar_color = "#2dd4bf" if expert.score > 0.7 else "#60a5fa" if expert.score > 0.3 else "#71717a"
            expert_rows.append(
                f'<div class="expert-row">'
                f'<span class="expert-name">{_esc(expert.login)}</span>'
                f'<span class="expert-stats">'
                f'<span class="expert-score-bar"><span class="bar-animate" style="width:{score_pct}%;background:{bar_color}"></span></span>'
                f'<span class="expert-score-val">{expert.score:.2f}</span>'
                f'</span>'
                f'<span class="expert-detail">{expert.commits} commits · {expert.lines:,} lines · {expert.files_touched} files · {expert.last_commit_days_ago}d ago</span>'
                f'</div>'
            )

        cards.append(
            f'<div class="area-card stagger-item">'
            f'<div class="area-header">'
            f'<div>'
            f'<div class="area-name">{_esc(area.name)}</div>'
            f'<div class="area-path">{_esc(area.path)}</div>'
            f'</div>'
            f'<div class="area-meta">'
            f'<span class="bus-factor" style="border-color:{bf_color};color:{bf_color};" title="Bus factor: {area.bus_factor}">'
            f'{area.bus_factor} experts · {bf_label}</span>'
            f'<span class="area-stat">{area.contributor_count} contributors · {area.total_commits} commits</span>'
            f'</div>'
            f'</div>'
            f'{"".join(expert_rows)}'
            f'</div>'
        )

    return "".join(cards)


def _build_sigma(edges: list[SocialEdge], lines_by_user: dict[str, int] | None = None, team_labels: dict[int, str] | None = None) -> str:
    """Build an embedded sigma.js graph, reusing the data builder from visualization.py."""
    from social_graph.output.visualization import _build_graph_data

    nodes_json, edges_json = _build_graph_data(edges, lines_by_user)
    # Include team labels in the data for bounding box labels
    labels_map = {str(k): v for k, v in (team_labels or {}).items()}
    graph_data = json.dumps({"nodes": nodes_json, "edges": edges_json, "teamLabels": labels_map}).replace("</", "<\\/")

    return _SIGMA_EMBED.replace("__GRAPH_DATA__", graph_data)


_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Social Graph</title>
<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/geist@1.3.1/dist/fonts/geist-sans/style.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/geist@1.3.1/dist/fonts/geist-mono/style.min.css">
<style>
  :root {{
    --bg-base: #09090b;
    --bg-surface: rgba(24,24,27,0.6);
    --bg-solid: #18181b;
    --bg-elevated: #27272a;
    --border: rgba(255,255,255,0.06);
    --border-active: rgba(255,255,255,0.12);
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --accent: #2dd4bf;
    --accent-dim: rgba(45,212,191,0.15);
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Geist', sans-serif; background: var(--bg-base); color: var(--text-secondary); padding: 24px; }}
  h1 {{ font-size: 20px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px; }}
  h2 {{ font-size: 18px; font-weight: 600; color: var(--text-primary); margin: 32px 0 12px; }}
  h3 {{ font-size: 15px; font-weight: 500; color: var(--text-secondary); margin: 24px 0 8px; }}
  .subtitle {{ font-size: 13px; color: var(--text-muted); margin-bottom: 24px; }}
  .subtitle a {{ color: var(--text-muted); }}

  .avatar {{ border-radius: 50%; vertical-align: middle; margin-right: 4px; border: 1px solid var(--border); }}

  /* Animations */
  @keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}

  /* Stagger utility */
  .stagger-item {{ animation: fadeInUp 0.3s ease-out both; }}
  .stagger-item:nth-child(1) {{ animation-delay: 0ms; }}
  .stagger-item:nth-child(2) {{ animation-delay: 30ms; }}
  .stagger-item:nth-child(3) {{ animation-delay: 60ms; }}
  .stagger-item:nth-child(4) {{ animation-delay: 90ms; }}
  .stagger-item:nth-child(5) {{ animation-delay: 120ms; }}
  .stagger-item:nth-child(6) {{ animation-delay: 150ms; }}
  .stagger-item:nth-child(7) {{ animation-delay: 180ms; }}
  .stagger-item:nth-child(8) {{ animation-delay: 210ms; }}
  .stagger-item:nth-child(9) {{ animation-delay: 240ms; }}
  .stagger-item:nth-child(10) {{ animation-delay: 270ms; }}
  .stagger-item:nth-child(n+11) {{ animation-delay: 300ms; }}

  /* Grid */
  .grid-table {{ border-collapse: separate; border-spacing: 2px; font-size: 12px; margin: 16px 0; }}
  .grid-table td {{ padding: 6px 8px; border: none; text-align: center; min-width: 60px; border-radius: 4px; }}
  .grid-table .corner {{ background: transparent; text-align: left; font-weight: 600; font-size: 11px; color: var(--text-muted); font-family: 'Geist Mono', monospace; white-space: nowrap; }}
  .grid-table .header-cell {{ background: var(--bg-solid); font-weight: 500; color: var(--text-secondary); font-family: 'Geist Mono', monospace; font-size: 11px; max-width: 80px; overflow: hidden; text-overflow: ellipsis; }}
  .grid-table .row-label {{ background: var(--bg-solid); font-weight: 500; text-align: left; color: var(--text-secondary); font-family: 'Geist Mono', monospace; font-size: 11px; white-space: nowrap; }}
  .grid-table .cell {{ color: #fff; font-weight: 500; font-variant-numeric: tabular-nums; font-family: 'Geist Mono', monospace; transition: box-shadow 0.15s; }}
  .grid-table .cell:hover {{ box-shadow: 0 0 8px rgba(45,212,191,0.4); }}
  .grid-table .cell.empty {{ background: var(--bg-base); }}
  .grid-wrapper {{ overflow-x: auto; }}

  /* Peer tables */
  .peer-section {{ margin: 16px 0; }}
  .peer-header {{ font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); }}
  .peer-table {{ border-collapse: collapse; width: 100%; max-width: 600px; margin: 8px 0; }}
  .peer-table th {{ text-align: left; padding: 8px 12px; background: var(--bg-solid); color: var(--text-muted); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid var(--border); }}
  .peer-table td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  .peer-name {{ color: var(--text-secondary); }}
  .peer-weight {{ display: flex; align-items: center; gap: 10px; }}
  .weight-num {{ font-family: 'Geist Mono', monospace; font-variant-numeric: tabular-nums; font-size: 13px; min-width: 45px; }}
  .bar-bg {{ flex: 1; max-width: 120px; height: 6px; background: var(--bg-elevated); border-radius: 3px; }}
  .bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.4s ease-out; }}
  .peer-table tbody tr:hover {{ background: var(--bg-elevated); }}

  /* Tabs */
  .tabs {{ display: flex; gap: 0; margin: 24px 0 0; border-bottom: 1px solid var(--border); position: relative; }}
  .tab {{ padding: 10px 20px; cursor: pointer; color: var(--text-muted); font-size: 14px; border-bottom: 2px solid transparent; transition: color 0.15s; }}
  .tab:hover {{ color: var(--text-secondary); }}
  .tab.active {{ color: var(--text-primary); border-bottom-color: transparent; }}
  .tab-indicator {{
    position: absolute;
    bottom: -1px;
    height: 2px;
    background: var(--accent);
    border-radius: 1px;
    transition: left 0.3s cubic-bezier(0.4, 0, 0.2, 1), width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }}
  .tab-content {{ display: none; padding: 16px 0; }}
  .tab-content.active {{ display: block; animation: fadeInUp 0.2s ease-out; }}

  /* Teams */
  .teams-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }}
  .team-card {{ background: var(--bg-surface); backdrop-filter: blur(12px); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; transition: border-color 0.15s, transform 0.15s; }}
  .team-card:hover {{ border-color: var(--border-active); transform: translateY(-1px); }}
  .team-header {{ padding: 14px 20px; background: rgba(9,9,11,0.5); display: flex; justify-content: space-between; align-items: center; }}
  .team-header h3 {{ font-size: 15px; color: var(--text-primary); margin: 0; }}
  .team-count {{ font-size: 12px; background: var(--accent-dim); color: var(--accent); font-family: 'Geist Mono', monospace; padding: 2px 10px; border-radius: 10px; }}
  .team-member {{ padding: 14px 20px; border-top: 1px solid var(--border); }}
  .team-member:hover {{ background: var(--bg-elevated); }}
  .tm-header {{ display: flex; align-items: center; gap: 10px; }}
  .tm-info {{ flex: 1; }}
  .tm-name {{ font-weight: 600; color: var(--text-primary); font-size: 14px; }}
  .tm-login {{ font-size: 12px; color: var(--text-muted); }}
  .tm-login a {{ color: var(--accent); text-decoration: none; }}
  .tm-login a:hover {{ text-decoration: underline; }}
  .tm-meta {{ font-size: 11px; color: var(--text-muted); margin-top: 4px; }}
  .tm-stats {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }}
  .tm-stats span {{ background: var(--accent-dim); color: var(--accent); padding: 2px 8px; border-radius: 6px; font-size: 11px; font-family: 'Geist Mono', monospace; }}
  .tm-collabs {{ font-size: 11px; color: var(--text-muted); margin-top: 4px; }}

  /* Experts */
  .areas-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 16px; }}
  .area-card {{ background: var(--bg-surface); backdrop-filter: blur(12px); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; transition: border-color 0.15s, transform 0.15s; }}
  .area-card:hover {{ border-color: var(--border-active); transform: translateY(-1px); }}
  .area-header {{ padding: 14px 18px; background: rgba(9,9,11,0.5); display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }}
  .area-name {{ font-size: 15px; font-weight: 600; color: var(--text-primary); }}
  .area-path {{ font-size: 11px; color: var(--text-muted); font-family: 'Geist Mono', monospace; margin-top: 2px; }}
  .area-meta {{ text-align: right; flex-shrink: 0; }}
  .bus-factor {{ display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border: 1.5px solid; border-radius: 10px; }}
  .area-stat {{ display: block; font-size: 11px; color: var(--text-muted); margin-top: 4px; }}
  .expert-row {{ padding: 8px 18px; border-top: 1px solid var(--border); display: grid; grid-template-columns: 140px 1fr; grid-template-rows: auto auto; column-gap: 12px; row-gap: 2px; align-items: center; }}
  .expert-row:hover {{ background: var(--bg-elevated); }}
  .expert-name {{ font-size: 13px; font-family: 'Geist Mono', monospace; color: var(--text-secondary); font-weight: 500; grid-row: 1; }}
  .expert-stats {{ display: flex; align-items: center; gap: 8px; grid-row: 1; }}
  .expert-score-bar {{ width: 80px; height: 6px; background: var(--bg-elevated); border-radius: 3px; display: inline-block; }}
  .expert-score-bar span {{ display: block; height: 100%; border-radius: 3px; }}
  .expert-score-val {{ font-size: 12px; color: var(--text-muted); font-variant-numeric: tabular-nums; min-width: 32px; }}
  .expert-detail {{ grid-column: 1 / -1; font-size: 11px; font-family: 'Geist Mono', monospace; color: var(--text-muted); }}

  #tab-graph.active {{ padding: 0; margin: 0 -24px -24px; }}
</style>
</head>
<body>
  <h1>Social Graph</h1>
  <p class="subtitle">Developer collaboration patterns derived from pull request review history.</p>

  <div class="tabs">
    <div class="tab" data-tab="grid">Heat Map</div>
    <div class="tab" data-tab="peers">Peer Tables</div>
    <div class="tab" data-tab="teams">Teams</div>
    <div class="tab" data-tab="experts">Experts</div>
    <div class="tab active" data-tab="graph">Graph</div>
    <div class="tab-indicator"></div>
  </div>

  <div class="tab-content" id="tab-grid">
    <h2>Collaboration Heat Map</h2>
    <div class="grid-wrapper">
      {grid}
    </div>
  </div>

  <div class="tab-content" id="tab-peers">
    <h2>Peer Relationships</h2>
    {peers}
  </div>

  <div class="tab-content" id="tab-teams">
    <h2>Teams</h2>
    <p class="subtitle" style="margin-bottom:16px;">Auto-detected teams based on PR review collaboration patterns. People who frequently review each other's code are grouped together.</p>
    <div class="teams-grid">
      {teams}
    </div>
  </div>

  <div class="tab-content" id="tab-experts">
    <h2>Code Area Experts</h2>
    <p class="subtitle" style="margin-bottom:16px;">Auto-detected from git history. Expertise score = recency (35%) + time-decayed lines (30%) + file breadth (20%) + commits (15%), per <a href="https://arxiv.org/abs/2208.07501">Cury et al. 2022</a>. Bus factor via coverage-based iterative removal (<a href="https://arxiv.org/abs/1604.06766">Avelino et al. 2016</a>).</p>
    <div class="areas-grid">
      {experts}
    </div>
  </div>

  <div class="tab-content active" id="tab-graph">
    {sigma}
  </div>

  <script>
    const tabs = document.querySelectorAll('.tab');
    const indicator = document.querySelector('.tab-indicator');

    function updateIndicator(tab) {{
      indicator.style.left = tab.offsetLeft + 'px';
      indicator.style.width = tab.offsetWidth + 'px';
    }}

    const activeTab = document.querySelector('.tab.active');
    if (activeTab) updateIndicator(activeTab);

    tabs.forEach(tab => {{
      tab.addEventListener('click', () => {{
        tabs.forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        updateIndicator(tab);
        const content = document.getElementById('tab-' + tab.dataset.tab);
        content.classList.add('active');
        content.querySelectorAll('.bar-animate').forEach(bar => {{
          bar.style.animation = 'none';
          bar.offsetHeight;
          bar.style.animation = '';
        }});
        if (tab.dataset.tab === 'graph') {{
          window.dispatchEvent(new Event('init-sigma'));
        }}
      }});
    }});

    window.addEventListener('resize', () => {{
      const current = document.querySelector('.tab.active');
      if (current) updateIndicator(current);
    }});
  </script>
</body>
</html>
"""


_SIGMA_EMBED = """\
<div style="display:flex;height:calc(100vh - 90px);min-height:500px;">
  <div id="sigma-container" style="flex:1;height:100%;background:#0a0a0f;"></div>
  <div id="sigma-sidebar" style="width:320px;height:100%;padding:16px;overflow-y:auto;font-size:13px;background:rgba(24,24,27,0.6);backdrop-filter:blur(16px);border-left:1px solid rgba(255,255,255,0.06);font-family:'Geist',sans-serif;">
    <p style="color:#71717a;font-size:12px;">Hover to highlight connections. Click a node for details.</p>
    <div id="sigma-node-info" style="display:none;">
      <hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:12px 0;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
        <img id="sigma-avatar" style="width:40px;height:40px;border-radius:50%;border:2px solid rgba(255,255,255,0.06);" src="" alt="">
        <div>
          <div id="sigma-name" style="font-weight:600;color:#fafafa;"></div>
          <div id="sigma-login" style="font-size:12px;color:#a1a1aa;"></div>
        </div>
      </div>
      <div id="sigma-meta" style="font-size:11px;color:#71717a;margin-bottom:8px;"></div>
      <div id="sigma-stat" style="font-size:12px;color:#a1a1aa;margin-bottom:12px;"></div>
      <h4 style="font-size:12px;color:#71717a;font-family:'Geist',sans-serif;text-transform:uppercase;letter-spacing:0.5px;margin:12px 0 6px;" id="sigma-given-label">Reviews PRs from</h4>
      <ul id="sigma-given" style="list-style:none;padding:0;margin:0;"></ul>
      <h4 style="font-size:12px;color:#71717a;font-family:'Geist',sans-serif;text-transform:uppercase;letter-spacing:0.5px;margin:12px 0 6px;" id="sigma-received-label">PRs reviewed by</h4>
      <ul id="sigma-received" style="list-style:none;padding:0;margin:0;"></ul>
    </div>
  </div>
</div>
<script src="https://unpkg.com/graphology@0.25.4/dist/graphology.umd.min.js"></script>
<script src="https://unpkg.com/sigma@2.4.0/build/sigma.min.js"></script>
<script type="module">
// Defer init until the tab is visible — Sigma needs a visible container to measure dimensions
let initialized = false;
async function initSigma() {
  if (initialized) return;
  initialized = true;

const faMod = await import('https://esm.sh/graphology-layout-forceatlas2@0.10.1');
const FA2 = faMod.default || faMod;
const UndirectedGraph = graphology.UndirectedGraph;
const SigmaClass = (typeof Sigma === 'function') ? Sigma : Sigma?.Sigma || Sigma?.default;

const graphData = __GRAPH_DATA__;

// Pre-load avatars
const avatarImages = new Map();
await Promise.all(graphData.nodes.filter(n => n.avatarUrl).map(n =>
  new Promise(resolve => {
    const img = new Image(); img.crossOrigin = 'anonymous';
    img.onload = () => { avatarImages.set(n.id, img); resolve(); };
    img.onerror = () => resolve(); img.src = n.avatarUrl;
  })
));

const graph = new UndirectedGraph();
const nodeDataById = new Map(graphData.nodes.map(n => [n.id, n]));

const N = graphData.nodes.length;
let seed = 42;
function rng() { seed = (seed * 1664525 + 1013904223) & 0xffffffff; return (seed >>> 0) / 0xffffffff; }

const degree = {};
graphData.edges.forEach(e => { degree[e.source] = (degree[e.source] || 0) + 1; degree[e.target] = (degree[e.target] || 0) + 1; });
const maxDegree = Math.max(1, ...Object.values(degree));

graphData.nodes.forEach((n, i) => {
  const angle = (2 * Math.PI * i) / N;
  const norm = (degree[n.id] || 0) / maxDegree;
  const radius = 100 + (1 - norm) * 700 + rng() * 150;
  graph.addNode(n.id, { label: n.label, color: n.color, size: n.size, x: Math.cos(angle) * radius, y: Math.sin(angle) * radius });
});

function weightToColor(w) { const lo = 0x33, hi = 0xDD, v = Math.round(lo + (hi - lo) * w); return '#' + v.toString(16).padStart(2, '0').repeat(3); }

graphData.edges.forEach((e, i) => {
  graph.addUndirectedEdgeWithKey('e' + i, e.source, e.target, { size: 0.5 + e.weight * 3, weight: e.weight, color: weightToColor(e.weight), label: e.weight.toFixed(2) });
});

const fa2Settings = FA2.inferSettings(graph);
fa2Settings.scalingRatio = 60; fa2Settings.edgeWeightInfluence = 1;
FA2.assign(graph, { iterations: 300, settings: fa2Settings });

// Post-layout: pull nodes toward community centroids to create clear visual groups.
// Compute centroid per community, then shift each node toward its centroid.
const communityNodes = {};
graphData.nodes.forEach(n => {
  const c = n.community;
  if (!communityNodes[c]) communityNodes[c] = [];
  communityNodes[c].push(n.id);
});

// Compute centroids from FA2 positions
const centroids = {};
for (const [c, members] of Object.entries(communityNodes)) {
  let cx = 0, cy = 0;
  members.forEach(id => { const a = graph.getNodeAttributes(id); cx += a.x; cy += a.y; });
  centroids[c] = { x: cx / members.length, y: cy / members.length };
}

// Layout: teams packed in a circle structure.
// Most-connected team at center, others arranged around it.
// Within each team, nodes arranged in a circle.

// 1. Rank teams by total cross-team edge weight (connectivity to the org)
const teamConnectivity = {};
const communityKeys = Object.keys(communityNodes);
communityKeys.forEach(c => { teamConnectivity[c] = 0; });
graphData.edges.forEach(e => {
  const cS = graphData.nodes.find(n => n.id === e.source)?.community;
  const cT = graphData.nodes.find(n => n.id === e.target)?.community;
  if (cS !== undefined && cT !== undefined && cS !== cT) {
    teamConnectivity[cS] = (teamConnectivity[cS] || 0) + e.weight;
    teamConnectivity[cT] = (teamConnectivity[cT] || 0) + e.weight;
  }
});

// Sort: most connected first (center), least connected last (outer)
const rankedTeams = communityKeys.sort((a, b) => (teamConnectivity[b] || 0) - (teamConnectivity[a] || 0));

// 2. Compute team radii (how much space each team needs based on member count)
const nodeSpacing = 200; // space between nodes within a team
const teamRadii = {};
rankedTeams.forEach(c => {
  const n = communityNodes[c].length;
  if (n === 1) teamRadii[c] = 0;
  else if (n === 2) teamRadii[c] = nodeSpacing * 0.6;
  else teamRadii[c] = (nodeSpacing * n) / (2 * Math.PI) + 40;
});

// 3. Place teams: first team at center, rest packed around it with no overlap
const teamCenters = {};
const padding = 350; // gap between team bounding circles

if (rankedTeams.length > 0) {
  teamCenters[rankedTeams[0]] = { x: 0, y: 0 };
}
if (rankedTeams.length > 1) {
  // Pack remaining teams in a ring around the center team
  const centerR = teamRadii[rankedTeams[0]] || 0;
  const outerTeams = rankedTeams.slice(1);
  const ringRadius = centerR + padding + Math.max(...outerTeams.map(c => teamRadii[c] || 0));
  outerTeams.forEach((c, i) => {
    const angle = (2 * Math.PI * i) / outerTeams.length - Math.PI / 2;
    teamCenters[c] = {
      x: Math.cos(angle) * ringRadius,
      y: Math.sin(angle) * ringRadius,
    };
  });
}

// 4. Place nodes within each team in a circle
rankedTeams.forEach(c => {
  const members = communityNodes[c];
  const center = teamCenters[c];
  const n = members.length;
  const r = teamRadii[c];

  // Sort members by their degree (most connected in team at top)
  const memberDegree = {};
  members.forEach(id => { memberDegree[id] = degree[id] || 0; });
  members.sort((a, b) => (memberDegree[b] || 0) - (memberDegree[a] || 0));

  if (n === 1) {
    const attrs = graph.getNodeAttributes(members[0]);
    attrs.x = center.x;
    attrs.y = center.y;
  } else {
    members.forEach((id, i) => {
      const angle = (2 * Math.PI * i) / n - Math.PI / 2;
      const attrs = graph.getNodeAttributes(id);
      attrs.x = center.x + Math.cos(angle) * r;
      attrs.y = center.y + Math.sin(angle) * r;
    });
  }
});

const container = document.getElementById('sigma-container');
let hoveredNode, hoveredNeighbors;

const renderer = new SigmaClass(graph, container, {
  zoomingRatio: 1.02,
  renderEdgeLabels: false, enableEdgeEvents: false, defaultEdgeColor: '#555',
  labelRenderedSizeThreshold: 6, labelColor: { color: '#a1a1aa' },
  nodeReducer: (node, data) => {
    const res = { ...data };
    if (hoveredNode) {
      if (hoveredNode !== node && (!hoveredNeighbors || !hoveredNeighbors.has(node))) { res.color = '#222'; res.label = ''; }
      else { res.label = data.label; res.zIndex = 1; }
    }
    return res;
  },
  edgeReducer: (edge, data) => {
    const res = { ...data };
    if (hoveredNode) { if (!graph.extremities(edge).includes(hoveredNode)) res.hidden = true; else { res.color = '#fff'; res.forceLabel = true; } }
    return res;
  },
  defaultDrawNodeHover: (ctx, data) => {
    if (!data.label) return;
    ctx.font = '600 13px Geist, sans-serif';
    const tw = ctx.measureText(data.label).width, p = 8, bw = tw + p*2, bh = 14 + p*2;
    ctx.fillStyle = 'rgba(9,9,11,0.9)'; ctx.beginPath(); ctx.roundRect(data.x - bw/2, data.y - data.size - bh - 4, bw, bh, 6); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(data.label, data.x, data.y - data.size - bh/2 - 4);
  },
});

// Build team label lookup from node data
const teamLabels = {};
const teamColors = {};
graphData.nodes.forEach(n => {
  if (!teamColors[n.community]) teamColors[n.community] = n.color;
});

function drawOverlay() {
  const ctx = renderer.getCanvases().mouse.getContext('2d');
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  // Draw bounding circles per community
  const clusterData = {};
  graph.forEachNode((nodeId, attrs) => {
    const n = graphData.nodes.find(nd => nd.id === nodeId);
    if (!n) return;
    const pos = renderer.graphToViewport({ x: attrs.x, y: attrs.y });
    const c = n.community;
    if (!clusterData[c]) clusterData[c] = { positions: [], color: n.color };
    clusterData[c].positions.push(pos);
  });

  const teamLabels = graphData.teamLabels || {};
  for (const [c, data] of Object.entries(clusterData)) {
    const positions = data.positions;
    // Compute centroid
    let cx = 0, cy = 0;
    positions.forEach(p => { cx += p.x; cy += p.y; });
    cx /= positions.length; cy /= positions.length;
    // Compute radius: max distance from centroid + padding
    let maxR = 0;
    positions.forEach(p => {
      const d = Math.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2);
      if (d > maxR) maxR = d;
    });
    const r = maxR + 50;
    // Filled circle
    ctx.fillStyle = data.color + '08';
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    // Border
    ctx.strokeStyle = data.color + '20';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    // Team label at top of circle
    const label = teamLabels[c] || '';
    if (label) {
      ctx.font = '600 11px Geist, sans-serif';
      ctx.fillStyle = data.color + 'BB';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(label, cx, cy - r - 6);
    }
  }

  // Draw avatars on top
  graph.forEachNode((nodeId, attrs) => {
    const img = avatarImages.get(nodeId); if (!img) return;
    const pos = renderer.graphToViewport({ x: attrs.x, y: attrs.y });
    const s = attrs.size / renderer.getCamera().ratio;
    if (s < 3) return;
    ctx.save(); ctx.beginPath(); ctx.arc(pos.x, pos.y, s, 0, Math.PI * 2); ctx.closePath(); ctx.clip();
    ctx.drawImage(img, pos.x - s, pos.y - s, s * 2, s * 2); ctx.restore();
    ctx.beginPath(); ctx.arc(pos.x, pos.y, s, 0, Math.PI * 2); ctx.strokeStyle = attrs.color; ctx.lineWidth = 2; ctx.stroke();
  });
}
renderer.on('afterRender', drawOverlay);
// Kick initial draw after Sigma settles
setTimeout(() => renderer.refresh(), 100);

renderer.on('enterNode', ({ node }) => { hoveredNode = node; hoveredNeighbors = new Set(graph.neighbors(node)); renderer.refresh({ skipIndexation: true }); });
renderer.on('leaveNode', () => { hoveredNode = undefined; hoveredNeighbors = undefined; renderer.refresh({ skipIndexation: true }); });

function barColor(w) { return w > 0.7 ? '#2dd4bf' : w > 0.3 ? '#60a5fa' : '#71717a'; }
function fillList(id, data) {
  const el = document.getElementById(id); el.innerHTML = '';
  Object.entries(data).forEach(([name, w]) => {
    const li = document.createElement('li');
    li.style.cssText = 'display:flex;justify-content:space-between;align-items:center;padding:3px 0;';
    li.innerHTML = '<span>' + name + '</span><span style="display:flex;align-items:center;gap:6px;">' +
      '<span style="width:50px;height:6px;background:#27272a;border-radius:3px;display:inline-block;"><span style="display:block;width:' + Math.round(w*100) + '%;height:100%;background:' + barColor(w) + ';border-radius:3px;"></span></span>' +
      '<span style="color:#a1a1aa;font-family:Geist Mono,monospace;font-size:12px;min-width:40px;text-align:right;">' + w.toFixed(3) + '</span></span>';
    el.appendChild(li);
  });
  if (!Object.keys(data).length) el.innerHTML = '<li style="color:#555;font-size:12px;">None</li>';
}

renderer.on('clickNode', ({ node }) => {
  const nd = nodeDataById.get(node); if (!nd) return;
  const av = document.getElementById('sigma-avatar');
  if (nd.avatarUrl) { av.src = nd.avatarUrl; av.style.display = 'block'; } else av.style.display = 'none';
  document.getElementById('sigma-name').textContent = nd.label;
  const loginEl = document.getElementById('sigma-login');
  loginEl.innerHTML = nd.htmlUrl ? '<a href="' + nd.htmlUrl + '" target="_blank" style="color:#2dd4bf;">@' + nd.login + '</a>' : '@' + nd.login;
  const meta = [];
  if (nd.company) meta.push('\\ud83c\\udfe2 ' + nd.company);
  if (nd.location) meta.push('\\ud83d\\udccd ' + nd.location);
  document.getElementById('sigma-meta').textContent = meta.join('  ');
  document.getElementById('sigma-stat').textContent = 'Community ' + (nd.community + 1) + ' \\u00b7 Weighted degree: ' + nd.degree;
  document.getElementById('sigma-given-label').textContent = nd.label + ' reviews PRs from';
  document.getElementById('sigma-received-label').textContent = 'Who reviews ' + nd.label + "\\'s PRs";
  fillList('sigma-given', nd.reviewsGiven);
  fillList('sigma-received', nd.reviewsReceived);
  document.getElementById('sigma-node-info').style.display = 'block';
});
renderer.on('clickStage', () => { document.getElementById('sigma-node-info').style.display = 'none'; });

} // end initSigma

window.addEventListener('init-sigma', initSigma);
// Init immediately since graph tab is the default
initSigma();
</script>
"""
