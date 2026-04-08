"""What-if simulation engine for engineering org planning.

Answers three classes of questions:

**Departure** (`simulate_departure`):
  "What if Alice leaves?" → Which code areas lose coverage, what's the new
  bus factor, who are the best candidates to backfill, and what pairing
  sessions would transfer the most knowledge?

**Reorganisation** (`simulate_reorg`):
  "What if we move people between teams?" → How does cross-team review
  coverage change, do we create new silos, what's the net effect on bus
  factors?

**New hire placement** (`simulate_hire`):
  "We're hiring for these areas." → Which existing team should they join
  to maximise knowledge distribution? Who should be their onboarding buddy
  based on social connectivity + area expertise?

All simulations are pure functions over the cached graph data — no side
effects, no mutations, no API calls.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from social_graph.graph.code_expertise import BUS_FACTOR_AUTHOR_THRESHOLD, CodeArea
from social_graph.models import SocialEdge


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

@dataclass
class AreaImpact:
    """Before/after state of a single code area."""
    area_path: str
    area_name: str
    bus_factor_before: int
    bus_factor_after: int
    top_expert_before: str
    top_expert_after: str
    coverage_delta: float  # negative means worse
    status: str  # "critical", "degraded", "unchanged", "improved"


@dataclass
class BackfillCandidate:
    """A person who could cover an area vacated by a departing expert."""
    login: str
    area_path: str
    area_name: str
    expertise_score: float
    social_proximity: float  # to the departing person
    combined_score: float
    reason: str


@dataclass
class PairingSession:
    """A recommended knowledge-transfer pairing."""
    mentor: str  # person with knowledge (or departing person)
    mentee: str  # person who should learn
    area_name: str
    area_path: str
    urgency: str  # "critical", "high", "medium"
    reason: str


@dataclass
class DepartureReport:
    """Full impact report for a person leaving."""
    login: str
    areas_affected: list[AreaImpact]
    backfill_candidates: list[BackfillCandidate]
    pairing_sessions: list[PairingSession]
    org_bus_factor_before: float  # avg across all areas
    org_bus_factor_after: float
    critical_areas: int  # areas that drop to bus_factor 0
    knowledge_monopolies: list[str]  # areas where this person is the ONLY expert


@dataclass
class TeamChange:
    """A single person moving teams."""
    login: str
    from_team: str
    to_team: str


@dataclass
class ReorgAreaImpact:
    """Team-diversity change for a single code area during a reorg."""
    area_path: str
    area_name: str
    teams_before: int
    teams_after: int
    status: str  # "improved", "degraded", "unchanged"


@dataclass
class ReorgImpact:
    """Impact of a team reorganisation."""
    changes: list[TeamChange]
    cross_team_edges_before: int
    cross_team_edges_after: int
    silo_score_before: float  # 0=no silos, 1=fully siloed
    silo_score_after: float
    areas_affected: list[ReorgAreaImpact]
    warnings: list[str]


@dataclass
class HirePlacement:
    """Recommendation for where to place a new hire."""
    target_area: str
    target_area_name: str
    recommended_team: str
    onboarding_buddy: str
    buddy_expertise: float
    bus_factor_improvement: str  # e.g. "1 → 2"
    reason: str


@dataclass
class HireReport:
    """Full report for new hire placement."""
    target_areas: list[str]
    placements: list[HirePlacement]
    org_bus_factor_improvement: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _social_weight_between(
    login_a: str,
    login_b: str,
    edge_index: dict[tuple[str, str], float],
) -> float:
    """Bidirectional social weight between two people."""
    w1 = edge_index.get((login_a, login_b), 0.0)
    w2 = edge_index.get((login_b, login_a), 0.0)
    if w1 and w2:
        return (w1 + w2) / 2
    return w1 or w2


def _build_edge_index(edges: list[SocialEdge]) -> dict[tuple[str, str], float]:
    """Build a lookup dict from social edges."""
    return {(e.author.login, e.reviewer.login): e.weight for e in edges}


def _recompute_bus_factor(experts_remaining: list[tuple[str, float]]) -> int:
    """Recompute bus factor from (login, score) pairs.

    Sorts by score descending to find the top score, then counts how many
    experts are within BUS_FACTOR_AUTHOR_THRESHOLD of it.
    """
    if not experts_remaining:
        return 0
    experts_remaining = sorted(experts_remaining, key=lambda x: x[1], reverse=True)
    top = experts_remaining[0][1]
    if top == 0:
        return 0
    threshold = top * BUS_FACTOR_AUTHOR_THRESHOLD
    return sum(1 for _, s in experts_remaining if s >= threshold)


def _avg_bus_factor(areas: list[CodeArea]) -> float:
    if not areas:
        return 0.0
    return sum(a.bus_factor for a in areas) / len(areas)


# ---------------------------------------------------------------------------
# Departure simulation
# ---------------------------------------------------------------------------

def simulate_departure(
    login: str,
    areas: list[CodeArea],
    edges: list[SocialEdge],
    member_to_community: dict[str, int],
    team_labels: dict[int, str],
) -> DepartureReport:
    """Simulate removing a person from the org and measure the impact.

    For each code area where they are an expert:
    - Recompute bus factor without them
    - Find the best backfill candidates (expertise + social proximity)
    - Generate pairing session recommendations for knowledge transfer
    """
    edge_index = _build_edge_index(edges)

    # All people in the social graph (computed once)
    all_people = set()
    for edge in edges:
        all_people.add(edge.author.login)
        all_people.add(edge.reviewer.login)
    all_people.discard(login)

    # Find all areas where this person is an expert
    affected_areas: list[CodeArea] = []
    for area in areas:
        if any(e.login == login for e in area.experts):
            affected_areas.append(area)

    knowledge_monopolies: list[str] = []
    area_impacts: list[AreaImpact] = []
    all_backfills: list[BackfillCandidate] = []
    all_pairings: list[PairingSession] = []

    for area in affected_areas:
        bf_before = area.bus_factor
        top_before = area.experts[0].login if area.experts else "—"

        remaining = [(e.login, e.score) for e in area.experts if e.login != login]
        bf_after = _recompute_bus_factor(remaining)
        top_after = remaining[0][0] if remaining else "—"

        if bf_after == 0:
            status = "critical"
            knowledge_monopolies.append(area.path)
        elif bf_after < bf_before:
            status = "degraded"
        elif bf_after > bf_before:
            # Can happen: removing the top expert lowers the threshold,
            # which may qualify more remaining experts.
            status = "improved"
        else:
            status = "unchanged"

        area_impacts.append(AreaImpact(
            area_path=area.path,
            area_name=area.name,
            bus_factor_before=bf_before,
            bus_factor_after=bf_after,
            top_expert_before=top_before,
            top_expert_after=top_after,
            coverage_delta=bf_after - bf_before,
            status=status,
        ))

        # Find backfill candidates — one per person per area
        area_backfills: list[BackfillCandidate] = []
        for candidate_login in all_people:
            existing_exp = next(
                (e.score for e in area.experts if e.login == candidate_login), 0.0,
            )
            social_prox = _social_weight_between(candidate_login, login, edge_index)
            combined = 0.5 * existing_exp + 0.5 * social_prox

            if combined > 0.20:
                reason = _backfill_reason(existing_exp, social_prox, login)
                area_backfills.append(BackfillCandidate(
                    login=candidate_login,
                    area_path=area.path,
                    area_name=area.name,
                    expertise_score=existing_exp,
                    social_proximity=social_prox,
                    combined_score=combined,
                    reason=reason,
                ))

        area_backfills.sort(key=lambda b: b.combined_score, reverse=True)
        all_backfills.extend(area_backfills)

        # Generate pairing sessions for at-risk areas
        if bf_after <= 1:
            if remaining:
                mentee = remaining[0][0]
                mentee_reason = "next in line"
            elif area_backfills:
                mentee = area_backfills[0].login
                mentee_reason = "best backfill candidate"
            else:
                mentee = None
                mentee_reason = ""

            if mentee:
                urgency = "critical" if bf_after == 0 else "high"
                all_pairings.append(PairingSession(
                    mentor=login,
                    mentee=mentee,
                    area_name=area.name,
                    area_path=area.path,
                    urgency=urgency,
                    reason=f"bus factor drops to {bf_after}; @{mentee} is {mentee_reason}",
                ))

                # Also pair with socially-connected people who could ramp up
                for c in area_backfills[:3]:
                    if c.login != mentee and c.social_proximity > 0.2:
                        all_pairings.append(PairingSession(
                            mentor=login,
                            mentee=c.login,
                            area_name=area.name,
                            area_path=area.path,
                            urgency="medium",
                            reason=f"social proximity {c.social_proximity:.0%} — can ramp up with pairing",
                        ))

    # Sort area impacts: critical first
    status_order = {"critical": 0, "degraded": 1, "unchanged": 2, "improved": 3}
    area_impacts.sort(key=lambda a: (status_order.get(a.status, 9), -a.bus_factor_before))

    return DepartureReport(
        login=login,
        areas_affected=area_impacts,
        backfill_candidates=all_backfills,
        pairing_sessions=all_pairings,
        org_bus_factor_before=_avg_bus_factor(areas),
        org_bus_factor_after=_avg_bus_factor_without(areas, login),
        critical_areas=len([a for a in area_impacts if a.status == "critical"]),
        knowledge_monopolies=knowledge_monopolies,
    )


def _backfill_reason(expertise: float, social: float, departing: str) -> str:
    """Build a concise reason string for a backfill candidate."""
    parts: list[str] = []
    if expertise >= 0.3:
        parts.append(f"{expertise:.0%} expertise")
    if social >= 0.3:
        parts.append(f"collaborates with @{departing}")
    if not parts:
        if expertise > 0:
            parts.append(f"{expertise:.0%} expertise")
        elif social > 0:
            parts.append(f"knows @{departing}")
    return ", ".join(parts) or "available"


def _avg_bus_factor_without(areas: list[CodeArea], login: str) -> float:
    """Compute avg bus factor across all areas after removing a person."""
    if not areas:
        return 0.0
    total = 0.0
    for area in areas:
        remaining = [(e.login, e.score) for e in area.experts if e.login != login]
        total += _recompute_bus_factor(remaining)
    return total / len(areas)


# ---------------------------------------------------------------------------
# Reorg simulation
# ---------------------------------------------------------------------------

def simulate_reorg(
    changes: list[tuple[str, int]],
    areas: list[CodeArea],
    edges: list[SocialEdge],
    communities: list[set[str]],
    member_to_community: dict[str, int],
    team_labels: dict[int, str],
) -> ReorgImpact:
    """Simulate moving people between teams and measure the impact.

    Args:
        changes: List of (login, new_community_idx) pairs.
        areas: Code areas with expertise data.
        edges: Social graph edges.
        communities: Current community sets.
        member_to_community: Current login → community mapping.
        team_labels: Community index → label.
    """
    # Build before state
    m2c_before = dict(member_to_community)
    m2c_after = dict(member_to_community)

    team_changes: list[TeamChange] = []
    for login, new_idx in changes:
        old_idx = m2c_before.get(login)
        old_label = team_labels.get(old_idx, f"Team {old_idx}") if old_idx is not None else "Unknown"
        new_label = team_labels.get(new_idx, f"Team {new_idx}")
        team_changes.append(TeamChange(login=login, from_team=old_label, to_team=new_label))
        m2c_after[login] = new_idx

    # Count cross-team edges before and after
    cross_before = _count_cross_team_edges(edges, m2c_before)
    cross_after = _count_cross_team_edges(edges, m2c_after)

    # Silo score: ratio of same-team to total edges (1.0 = fully siloed)
    total_edges = len(edges)
    silo_before = 1.0 - (cross_before / total_edges) if total_edges else 0.0
    silo_after = 1.0 - (cross_after / total_edges) if total_edges else 0.0

    # Check area impacts: does moving someone change bus factor distribution?
    area_impacts: list[ReorgAreaImpact] = []
    moved_logins = {login for login, _ in changes}

    for area in areas:
        expert_logins = {e.login for e in area.experts}
        if not expert_logins & moved_logins:
            continue

        # Bus factor doesn't change (knowledge-based, not team-based).
        # But team diversity within the area changes.
        teams_before = {m2c_before.get(e.login) for e in area.experts if e.login in m2c_before}
        teams_after = {m2c_after.get(e.login) for e in area.experts if e.login in m2c_after}
        teams_before.discard(None)
        teams_after.discard(None)

        if len(teams_after) > len(teams_before):
            status = "improved"
        elif len(teams_after) < len(teams_before):
            status = "degraded"
        else:
            status = "unchanged"

        area_impacts.append(ReorgAreaImpact(
            area_path=area.path,
            area_name=area.name,
            teams_before=len(teams_before),
            teams_after=len(teams_after),
            status=status,
        ))

    # Warnings
    warnings: list[str] = []
    if cross_after < cross_before:
        warnings.append(
            f"Cross-team review edges decrease from {cross_before} to {cross_after} — "
            "this reorg may increase silos"
        )
    if silo_after > silo_before + 0.05:
        warnings.append(
            f"Silo score increases from {silo_before:.0%} to {silo_after:.0%} — "
            "teams become more insular"
        )

    # Check if anyone loses all their cross-team connections
    for login, new_idx in changes:
        connections_in_new_team = sum(
            1 for e in edges
            if (e.author.login == login and m2c_after.get(e.reviewer.login) == new_idx)
            or (e.reviewer.login == login and m2c_after.get(e.author.login) == new_idx)
        )
        if connections_in_new_team == 0:
            new_label = team_labels.get(new_idx, f"Team {new_idx}")
            warnings.append(f"@{login} has no existing review connections in {new_label}")

    return ReorgImpact(
        changes=team_changes,
        cross_team_edges_before=cross_before,
        cross_team_edges_after=cross_after,
        silo_score_before=silo_before,
        silo_score_after=silo_after,
        areas_affected=area_impacts,
        warnings=warnings,
    )


def _count_cross_team_edges(edges: list[SocialEdge], m2c: dict[str, int]) -> int:
    """Count edges that cross team boundaries."""
    count = 0
    for e in edges:
        a_team = m2c.get(e.author.login)
        r_team = m2c.get(e.reviewer.login)
        if a_team is not None and r_team is not None and a_team != r_team:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Hire simulation
# ---------------------------------------------------------------------------

def simulate_hire(
    target_area_paths: list[str],
    areas: list[CodeArea],
    edges: list[SocialEdge],
    communities: list[set[str]],
    member_to_community: dict[str, int],
    team_labels: dict[int, str],
) -> HireReport:
    """Recommend team placement and onboarding buddy for a new hire.

    For each target area:
    - Find which team has the most experts (natural home)
    - Find which team would benefit MOST from having an extra person
      (bus factor improvement + cross-team knowledge)
    - Recommend an onboarding buddy: high expertise + high social connectivity
    """
    area_map = {a.path: a for a in areas}

    # Social reach per person (computed once)
    social_reach: dict[str, int] = defaultdict(int)
    for e in edges:
        social_reach[e.author.login] += 1
        social_reach[e.reviewer.login] += 1

    placements: list[HirePlacement] = []
    bf_improvements: list[float] = []

    for path in target_area_paths:
        area = area_map.get(path)
        if not area:
            continue

        # Which teams have experts in this area?
        team_expert_count: dict[int, int] = defaultdict(int)
        team_expert_score: dict[int, float] = defaultdict(float)
        for expert in area.experts:
            team_idx = member_to_community.get(expert.login)
            if team_idx is not None:
                team_expert_count[team_idx] += 1
                team_expert_score[team_idx] += expert.score

        if not team_expert_count:
            continue

        # Place in the team with fewest experts to spread knowledge.
        # If tied, pick the team with highest total expertise (better mentoring).
        sorted_teams = sorted(
            team_expert_count.keys(),
            key=lambda t: (team_expert_count[t], -team_expert_score[t]),
        )
        best_team_idx = sorted_teams[0]
        if team_expert_score[best_team_idx] == 0 and len(sorted_teams) > 1:
            best_team_idx = sorted_teams[-1]

        best_team_label = team_labels.get(best_team_idx, f"Team {best_team_idx}")

        # Find onboarding buddy: expertise in this area + social reach
        buddy_candidates = [
            (expert.login, expert.score, social_reach.get(expert.login, 0))
            for expert in area.experts
            if member_to_community.get(expert.login) == best_team_idx
        ]
        buddy_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)

        buddy = buddy_candidates[0] if buddy_candidates else None
        buddy_login = buddy[0] if buddy else area.experts[0].login if area.experts else "—"
        buddy_exp = buddy[1] if buddy else 0.0

        # Estimate new bus factor: assume the hire ramps to the median
        # expertise of current experts (conservative — not top, not zero).
        bf_current = area.bus_factor
        existing_scores = [(e.login, e.score) for e in area.experts]
        if existing_scores:
            scores = sorted(e.score for e in area.experts)
            mid = len(scores) // 2
            if len(scores) % 2 == 0:
                median_score = (scores[mid - 1] + scores[mid]) / 2
            else:
                median_score = scores[mid]
            projected = existing_scores + [("_new_hire", median_score)]
            bf_new = _recompute_bus_factor(projected)
        else:
            bf_new = 1
        bf_improvements.append(bf_new - bf_current)

        reason_parts = []
        if team_expert_count[best_team_idx] <= 1:
            reason_parts.append(f"only {team_expert_count[best_team_idx]} expert in {best_team_label}")
        if area.bus_factor <= 2:
            reason_parts.append(f"bus factor {area.bus_factor}")

        placements.append(HirePlacement(
            target_area=path,
            target_area_name=area.name,
            recommended_team=best_team_label,
            onboarding_buddy=buddy_login,
            buddy_expertise=buddy_exp,
            bus_factor_improvement=f"{bf_current} → {bf_new}",
            reason="; ".join(reason_parts) if reason_parts else "best team fit",
        ))

    return HireReport(
        target_areas=target_area_paths,
        placements=placements,
        org_bus_factor_improvement=sum(bf_improvements) / len(bf_improvements) if bf_improvements else 0.0,
    )
