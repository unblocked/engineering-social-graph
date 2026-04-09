"""Tests for the what-if simulation engine."""

from social_graph.graph.code_expertise import AreaExpert, CodeArea
from social_graph.graph.simulate import (
    _avg_bus_factor,
    _build_edge_index,
    _count_cross_team_edges,
    _recompute_bus_factor,
    _social_weight_between,
    simulate_departure,
    simulate_hire,
    simulate_reorg,
)
from social_graph.models import SocialEdge, User


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _user(login: str) -> User:
    return User(id=login, login=login)


def _edge(author: str, reviewer: str, weight: float) -> SocialEdge:
    return SocialEdge(author=_user(author), reviewer=_user(reviewer), weight=weight)


def _area(
    path: str, name: str, experts: list[tuple[str, float]], bus_factor: int = 3,
) -> CodeArea:
    return CodeArea(
        path=path,
        name=name,
        experts=[
            AreaExpert(
                login=login, score=score, commits=10, lines=100,
                files_touched=5, last_commit_days_ago=7,
            )
            for login, score in experts
        ],
        bus_factor=bus_factor,
        total_commits=50,
        total_lines=500,
        contributor_count=len(experts),
    )


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_build_edge_index(self):
        edges = [_edge("a", "b", 0.7), _edge("b", "c", 0.3)]
        idx = _build_edge_index(edges)
        assert idx[("a", "b")] == 0.7
        assert idx[("b", "c")] == 0.3

    def test_social_weight_bidirectional(self):
        import math
        idx = {("a", "b"): 0.8, ("b", "a"): 0.4}
        assert math.isclose(_social_weight_between("a", "b", idx), 0.6)

    def test_social_weight_one_way(self):
        idx = {("a", "b"): 0.8}
        assert _social_weight_between("a", "b", idx) == 0.8
        assert _social_weight_between("b", "a", idx) == 0.8

    def test_social_weight_none(self):
        assert _social_weight_between("a", "b", {}) == 0.0

    def test_recompute_bus_factor_normal(self):
        experts = [("alice", 1.0), ("bob", 0.6), ("charlie", 0.2)]
        assert _recompute_bus_factor(experts) == 2  # alice and bob >= 0.5

    def test_recompute_bus_factor_empty(self):
        assert _recompute_bus_factor([]) == 0

    def test_recompute_bus_factor_single(self):
        assert _recompute_bus_factor([("alice", 1.0)]) == 1

    def test_avg_bus_factor(self):
        areas = [
            _area("a", "A", [("x", 1.0)], bus_factor=1),
            _area("b", "B", [("x", 1.0)], bus_factor=3),
        ]
        assert _avg_bus_factor(areas) == 2.0

    def test_avg_bus_factor_empty(self):
        assert _avg_bus_factor([]) == 0.0


# ---------------------------------------------------------------------------
# Departure simulation
# ---------------------------------------------------------------------------

class TestSimulateDeparture:
    def _standard_fixture(self):
        areas = [
            _area("src/api", "API", [("alice", 0.9), ("bob", 0.5), ("charlie", 0.2)], bus_factor=2),
            _area("src/auth", "Auth", [("alice", 0.95)], bus_factor=1),
            _area("src/db", "DB", [("bob", 0.8), ("charlie", 0.6)], bus_factor=2),
        ]
        edges = [
            _edge("alice", "bob", 0.8),
            _edge("bob", "alice", 0.6),
            _edge("alice", "charlie", 0.3),
            _edge("bob", "charlie", 0.5),
        ]
        m2c = {"alice": 0, "bob": 0, "charlie": 1}
        labels = {0: "Frontend", 1: "Backend"}
        return areas, edges, m2c, labels

    def test_alice_departure_identifies_critical(self):
        areas, edges, m2c, labels = self._standard_fixture()
        report = simulate_departure("alice", areas, edges, m2c, labels)

        assert report.login == "alice"
        assert report.critical_areas >= 1  # auth is critical
        assert "src/auth" in report.knowledge_monopolies

    def test_alice_departure_has_area_impacts(self):
        areas, edges, m2c, labels = self._standard_fixture()
        report = simulate_departure("alice", areas, edges, m2c, labels)

        assert len(report.areas_affected) >= 2  # api + auth
        statuses = {a.area_path: a.status for a in report.areas_affected}
        assert statuses["src/auth"] == "critical"

    def test_alice_departure_suggests_backfills(self):
        areas, edges, m2c, labels = self._standard_fixture()
        report = simulate_departure("alice", areas, edges, m2c, labels)

        assert len(report.backfill_candidates) > 0
        # Bob should rank high — has expertise in API + strong social link to alice
        bob_backfills = [c for c in report.backfill_candidates if c.login == "bob"]
        assert len(bob_backfills) > 0

    def test_alice_departure_generates_pairings(self):
        areas, edges, m2c, labels = self._standard_fixture()
        report = simulate_departure("alice", areas, edges, m2c, labels)

        assert len(report.pairing_sessions) > 0
        # API area has bob as next-in-line, so alice→bob pairing should exist
        api_pairings = [p for p in report.pairing_sessions if p.area_path == "src/api"]
        assert len(api_pairings) >= 1
        assert api_pairings[0].mentor == "alice"

    def test_departure_org_bus_factor_decreases(self):
        areas, edges, m2c, labels = self._standard_fixture()
        report = simulate_departure("alice", areas, edges, m2c, labels)

        assert report.org_bus_factor_after <= report.org_bus_factor_before

    def test_bob_departure_no_monopoly(self):
        areas, edges, m2c, labels = self._standard_fixture()
        report = simulate_departure("bob", areas, edges, m2c, labels)

        # bob is not sole expert anywhere
        assert len(report.knowledge_monopolies) == 0

    def test_unknown_person_no_impact(self):
        areas, edges, m2c, labels = self._standard_fixture()
        report = simulate_departure("unknown_user", areas, edges, m2c, labels)

        assert len(report.areas_affected) == 0
        assert report.critical_areas == 0


# ---------------------------------------------------------------------------
# Reorg simulation
# ---------------------------------------------------------------------------

class TestSimulateReorg:
    def _standard_fixture(self):
        areas = [
            _area("src/api", "API", [("alice", 0.9), ("bob", 0.5)], bus_factor=2),
        ]
        edges = [
            _edge("alice", "bob", 0.8),
            _edge("bob", "alice", 0.6),
            _edge("alice", "charlie", 0.3),
        ]
        communities = [{"alice", "bob"}, {"charlie"}]
        m2c = {"alice": 0, "bob": 0, "charlie": 1}
        labels = {0: "Frontend", 1: "Backend"}
        return areas, edges, communities, m2c, labels

    def test_move_bob_to_backend(self):
        areas, edges, communities, m2c, labels = self._standard_fixture()
        report = simulate_reorg(
            changes=[("bob", 1)],
            areas=areas, edges=edges, communities=communities,
            member_to_community=m2c, team_labels=labels,
        )

        assert len(report.changes) == 1
        assert report.changes[0].from_team == "Frontend"
        assert report.changes[0].to_team == "Backend"

    def test_cross_team_edges_increase(self):
        areas, edges, communities, m2c, labels = self._standard_fixture()
        report = simulate_reorg(
            changes=[("bob", 1)],
            areas=areas, edges=edges, communities=communities,
            member_to_community=m2c, team_labels=labels,
        )

        # Moving bob to team 1 means alice↔bob edges become cross-team
        assert report.cross_team_edges_after >= report.cross_team_edges_before

    def test_silo_score_changes(self):
        areas, edges, communities, m2c, labels = self._standard_fixture()
        report = simulate_reorg(
            changes=[("bob", 1)],
            areas=areas, edges=edges, communities=communities,
            member_to_community=m2c, team_labels=labels,
        )

        # With bob in a different team, silo score should decrease
        assert report.silo_score_after <= report.silo_score_before

    def test_no_connection_warning(self):
        areas, edges, communities, m2c, labels = self._standard_fixture()
        # Move alice to team 1 — she has a connection to charlie (team 1)
        # Move bob to team 1 — he has NO direct connection to charlie
        report = simulate_reorg(
            changes=[("bob", 1)],
            areas=areas, edges=edges, communities=communities,
            member_to_community=m2c, team_labels=labels,
        )
        # bob has an edge to alice (team 0 after move), but alice→charlie exists
        # bob has no direct edge to charlie — check if warning fires
        # Note: bob DOES have edges to alice who stays in team 0, so he
        # does have connections in team 1 if alice moves too. Without alice
        # in team 1, bob has no team 1 connections. Let's verify.
        bob_warnings = [w for w in report.warnings if "bob" in w.lower()]
        # bob has edge to alice (who is in team 0) — no team 1 connections
        assert len(bob_warnings) >= 1


# ---------------------------------------------------------------------------
# Hire simulation
# ---------------------------------------------------------------------------

class TestSimulateHire:
    def _standard_fixture(self):
        areas = [
            _area("src/api", "API", [("alice", 0.9), ("bob", 0.5)], bus_factor=2),
            _area("src/auth", "Auth", [("alice", 0.95)], bus_factor=1),
        ]
        edges = [
            _edge("alice", "bob", 0.8),
            _edge("bob", "alice", 0.6),
        ]
        communities = [{"alice"}, {"bob"}]
        m2c = {"alice": 0, "bob": 1}
        labels = {0: "Frontend", 1: "Backend"}
        return areas, edges, communities, m2c, labels

    def test_hire_for_auth(self):
        areas, edges, communities, m2c, labels = self._standard_fixture()
        report = simulate_hire(
            ["src/auth"], areas, edges, communities, m2c, labels,
        )

        assert len(report.placements) == 1
        p = report.placements[0]
        assert p.target_area_name == "Auth"
        assert p.onboarding_buddy == "alice"  # only expert

    def test_hire_recommends_buddy_with_expertise(self):
        areas, edges, communities, m2c, labels = self._standard_fixture()
        report = simulate_hire(
            ["src/api"], areas, edges, communities, m2c, labels,
        )

        p = report.placements[0]
        assert p.buddy_expertise > 0

    def test_hire_bus_factor_improvement(self):
        areas, edges, communities, m2c, labels = self._standard_fixture()
        report = simulate_hire(
            ["src/auth"], areas, edges, communities, m2c, labels,
        )

        p = report.placements[0]
        assert "1 →" in p.bus_factor_improvement

    def test_hire_nonexistent_area(self):
        areas, edges, communities, m2c, labels = self._standard_fixture()
        report = simulate_hire(
            ["src/nonexistent"], areas, edges, communities, m2c, labels,
        )

        assert len(report.placements) == 0

    def test_hire_multiple_areas(self):
        areas, edges, communities, m2c, labels = self._standard_fixture()
        report = simulate_hire(
            ["src/api", "src/auth"], areas, edges, communities, m2c, labels,
        )

        assert len(report.placements) == 2
