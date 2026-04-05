"""Tests for the three research-backed algorithm improvements.

1. Expertise weights (Cury et al., ESEM 2022) — recency is the dominant factor
2. Bus factor (Avelino et al., ICPC 2016) — coverage-based iterative removal
3. Adaptive Louvain resolution (Fortunato & Barthelemy, 2007)
"""

from __future__ import annotations


from social_graph.graph.code_expertise import (
    BUS_FACTOR_AUTHOR_THRESHOLD,
    WEIGHT_BREADTH,
    WEIGHT_COMMITS,
    WEIGHT_LINES,
    WEIGHT_RECENCY,
    _compute_bus_factor,
    _count_knowledgeable_authors,
    _score_experts,
)
from social_graph.models import SocialEdge, User
from social_graph.output.visualization import _adaptive_resolution, detect_communities


# ---------------------------------------------------------------------------
# 1. Expertise weight tests (Cury et al.)
# ---------------------------------------------------------------------------


class TestExpertiseWeights:
    """Verify expertise scoring weights match Cury et al. findings."""

    def test_recency_is_highest_weight(self):
        """Recency should be the dominant factor per Cury et al."""
        assert WEIGHT_RECENCY > WEIGHT_LINES
        assert WEIGHT_RECENCY > WEIGHT_COMMITS
        assert WEIGHT_RECENCY > WEIGHT_BREADTH

    def test_commits_is_lowest_weight(self):
        """Commit count was not in Cury's final DOE model — lowest weight."""
        assert WEIGHT_COMMITS < WEIGHT_LINES
        assert WEIGHT_COMMITS < WEIGHT_RECENCY

    def test_weights_sum_to_one(self):
        """All weights should sum to 1.0."""
        total = WEIGHT_RECENCY + WEIGHT_LINES + WEIGHT_BREADTH + WEIGHT_COMMITS
        assert abs(total - 1.0) < 1e-9

    def test_recent_contributor_outranks_prolific_old_one(self):
        """A recent contributor should beat an older one with more commits."""
        stats = {
            "recent_dev": {
                "commits": 5,
                "lines": 200,
                "files_touched": 3,
                "decayed_lines": 190.0,
                "decayed_commits": 4.8,
                "last_commit_days": 2,
                "_files": {"a.py", "b.py", "c.py"},
            },
            "old_prolific_dev": {
                "commits": 50,
                "lines": 5000,
                "files_touched": 20,
                "decayed_lines": 100.0,  # heavily decayed
                "decayed_commits": 2.0,  # heavily decayed
                "last_commit_days": 150,
                "_files": set(f"f{i}.py" for i in range(20)),
            },
        }
        experts = _score_experts(stats)
        assert experts[0].login == "recent_dev"

    def test_single_contributor_gets_perfect_score(self):
        """A sole contributor should score 1.0 (all normalized dimensions maxed)."""
        stats = {
            "solo": {
                "commits": 10,
                "lines": 500,
                "files_touched": 5,
                "decayed_lines": 400.0,
                "decayed_commits": 9.0,
                "last_commit_days": 1,
                "_files": set(f"f{i}.py" for i in range(5)),
            },
        }
        experts = _score_experts(stats)
        assert len(experts) == 1
        assert experts[0].score == 1.0


# ---------------------------------------------------------------------------
# 2. Bus factor tests (Avelino et al.)
# ---------------------------------------------------------------------------


class TestBusFactor:
    """Verify bus factor computation."""

    def test_single_contributor_bus_factor_is_one(self):
        """If only one person works on the area, bus factor = 1."""
        stats = {
            "alice": {
                "commits": 20, "lines": 1000, "files_touched": 5,
                "decayed_lines": 900.0, "decayed_commits": 18.0,
                "last_commit_days": 1,
                "_files": {"a.py", "b.py", "c.py", "d.py", "e.py"},
            },
        }
        experts = _score_experts(stats)
        assert _count_knowledgeable_authors(experts) == 1

    def test_two_equal_contributors(self):
        """Two similarly-skilled contributors = bus factor 2."""
        stats = {
            "alice": {
                "commits": 20, "lines": 1000, "files_touched": 4,
                "decayed_lines": 900.0, "decayed_commits": 18.0,
                "last_commit_days": 1,
                "_files": {"a.py", "b.py", "c.py", "d.py"},
            },
            "bob": {
                "commits": 18, "lines": 900, "files_touched": 4,
                "decayed_lines": 850.0, "decayed_commits": 17.0,
                "last_commit_days": 2,
                "_files": {"a.py", "b.py", "c.py", "d.py"},
            },
        }
        experts = _score_experts(stats)
        assert _count_knowledgeable_authors(experts) == 2

    def test_many_strong_contributors(self):
        """Area with many active contributors should have high bus factor."""
        stats = {}
        for i in range(6):
            stats[f"dev{i}"] = {
                "commits": 20 - i, "lines": 1000 - i * 50, "files_touched": 8,
                "decayed_lines": 900.0 - i * 40, "decayed_commits": 18.0 - i * 1.5,
                "last_commit_days": 1 + i,
                "_files": set(f"f{j}.py" for j in range(8)),
            }
        experts = _score_experts(stats)
        bf = _count_knowledgeable_authors(experts)
        assert bf >= 4  # Most devs should be above 50% of top

    def test_minor_contributor_excluded(self):
        """Contributors below 50% of top score shouldn't count."""
        stats = {
            "alice": {
                "commits": 50, "lines": 5000, "files_touched": 10,
                "decayed_lines": 4500.0, "decayed_commits": 45.0,
                "last_commit_days": 1,
                "_files": set(f"f{i}.py" for i in range(10)),
            },
            "intern": {
                "commits": 2, "lines": 20, "files_touched": 1,
                "decayed_lines": 18.0, "decayed_commits": 1.8,
                "last_commit_days": 30,
                "_files": {"f0.py"},
            },
        }
        experts = _score_experts(stats)
        assert experts[1].score < experts[0].score * BUS_FACTOR_AUTHOR_THRESHOLD
        assert _count_knowledgeable_authors(experts) == 1

    def test_author_threshold_is_fifty_percent(self):
        """Author threshold should be 50% of top score."""
        assert BUS_FACTOR_AUTHOR_THRESHOLD == 0.50

    def test_empty_experts_returns_zero(self):
        """No experts means bus factor 0."""
        assert _count_knowledgeable_authors([]) == 0

    def test_coverage_based_removal_still_works(self):
        """The file-level iterative removal function should still be available."""
        stats = {
            "alice": {
                "commits": 20, "lines": 1000, "files_touched": 4,
                "decayed_lines": 900.0, "decayed_commits": 18.0,
                "last_commit_days": 1,
                "_files": {"a.py", "b.py", "c.py", "d.py"},
            },
            "bob": {
                "commits": 18, "lines": 900, "files_touched": 4,
                "decayed_lines": 850.0, "decayed_commits": 17.0,
                "last_commit_days": 2,
                "_files": {"a.py", "b.py", "c.py", "d.py"},
            },
        }
        experts = _score_experts(stats)
        bf = _compute_bus_factor(stats, experts)
        assert bf == 2


# ---------------------------------------------------------------------------
# 3. Adaptive Louvain resolution tests (Fortunato & Barthelemy)
# ---------------------------------------------------------------------------


class TestAdaptiveResolution:
    """Verify adaptive resolution parameter computation."""

    def test_small_graph_resolution_is_one(self):
        """Graphs with <= 2 nodes should use default resolution 1.0."""
        import networkx as nx

        g = nx.Graph()
        g.add_edge("a", "b", weight=1.0)
        assert _adaptive_resolution(g) == 1.0

    def test_twenty_node_graph(self):
        """For n=20, target_size=4: gamma = 20/(4*2.5) = 2.0."""
        import networkx as nx

        g = nx.Graph()
        for i in range(20):
            g.add_edge(f"n{i}", f"n{(i+1) % 20}", weight=1.0)
        gamma = _adaptive_resolution(g, target_community_size=4.0)
        assert abs(gamma - 2.0) < 0.01

    def test_ten_node_graph(self):
        """For n=10, target_size=4: gamma = 10/(4*2.5) = 1.0."""
        import networkx as nx

        g = nx.Graph()
        for i in range(10):
            g.add_edge(f"n{i}", f"n{(i+1) % 10}", weight=1.0)
        gamma = _adaptive_resolution(g, target_community_size=4.0)
        assert gamma == 1.0

    def test_thirty_node_graph(self):
        """For n=30, target_size=4: gamma = 30/(4*2.5) = 3.0."""
        import networkx as nx

        g = nx.Graph()
        for i in range(30):
            g.add_edge(f"n{i}", f"n{(i+1) % 30}", weight=1.0)
        gamma = _adaptive_resolution(g, target_community_size=4.0)
        assert abs(gamma - 3.0) < 0.01

    def test_resolution_scales_with_graph_size(self):
        """Larger graphs should get higher resolution."""
        import networkx as nx

        g_small = nx.Graph()
        for i in range(10):
            g_small.add_edge(f"n{i}", f"n{(i+1) % 10}", weight=1.0)

        g_large = nx.Graph()
        for i in range(40):
            g_large.add_edge(f"n{i}", f"n{(i+1) % 40}", weight=1.0)

        assert _adaptive_resolution(g_large) > _adaptive_resolution(g_small)

    def test_detect_communities_uses_adaptive_resolution(self):
        """End-to-end: communities should be detected with adaptive resolution."""
        # Create two tight clusters connected by a weak bridge
        edges = []
        # Team A: alice, bob, charlie review each other
        for a, b in [("alice", "bob"), ("bob", "charlie"), ("alice", "charlie")]:
            edges.append(SocialEdge(
                author=User(id=a, login=a),
                reviewer=User(id=b, login=b),
                weight=0.8,
            ))
            edges.append(SocialEdge(
                author=User(id=b, login=b),
                reviewer=User(id=a, login=a),
                weight=0.7,
            ))
        # Team B: dave, eve review each other
        edges.append(SocialEdge(
            author=User(id="dave", login="dave"),
            reviewer=User(id="eve", login="eve"),
            weight=0.9,
        ))
        edges.append(SocialEdge(
            author=User(id="eve", login="eve"),
            reviewer=User(id="dave", login="dave"),
            weight=0.8,
        ))
        # Weak bridge: alice → dave
        edges.append(SocialEdge(
            author=User(id="alice", login="alice"),
            reviewer=User(id="dave", login="dave"),
            weight=0.2,
        ))

        communities, member_map = detect_communities(edges)
        # Should detect at least 2 communities
        assert len(communities) >= 2
        # alice, bob, charlie should be in the same community
        assert member_map["alice"] == member_map["bob"] == member_map["charlie"]
        # dave and eve should be together
        assert member_map["dave"] == member_map["eve"]
