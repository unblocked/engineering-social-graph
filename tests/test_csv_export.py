"""Tests for CSV export."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from social_graph.graph.code_expertise import AreaExpert, CodeArea
from social_graph.models import SocialEdge, User
from social_graph.output.json_export import export_csv


def _edge(author: str, reviewer: str, weight: float) -> SocialEdge:
    return SocialEdge(
        author=User(id=author, login=author),
        reviewer=User(id=reviewer, login=reviewer),
        weight=weight,
    )


def _area(name: str, experts: list[AreaExpert], bus_factor: int = 1) -> CodeArea:
    return CodeArea(path=f"src/{name}", name=name, experts=experts, bus_factor=bus_factor)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


class TestCSVExport:
    def test_creates_three_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            export_csv([], [], [], {}, {}, out)
            assert (out / "edges.csv").exists()
            assert (out / "experts.csv").exists()
            assert (out / "teams.csv").exists()

    def test_edges_csv_content(self):
        edges = [_edge("alice", "bob", 0.8), _edge("bob", "charlie", 0.3)]
        communities = [{"alice", "bob"}, {"charlie"}]
        member_to_community = {"alice": 0, "bob": 0, "charlie": 1}
        team_labels = {0: "Backend", 1: "Frontend"}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            export_csv(edges, [], communities, team_labels, member_to_community, out)
            rows = _read_csv(out / "edges.csv")

            assert len(rows) == 2
            assert rows[0]["author"] == "alice"
            assert rows[0]["reviewer"] == "bob"
            assert rows[0]["weight"] == "0.8"
            assert rows[0]["team_author"] == "Backend"
            assert rows[0]["team_reviewer"] == "Backend"
            assert rows[1]["team_reviewer"] == "Frontend"

    def test_experts_csv_content(self):
        areas = [
            _area("auth", [
                AreaExpert(login="alice", score=5.0, commits=20, lines=500, files_touched=10, last_commit_days_ago=3),
                AreaExpert(login="bob", score=2.0, commits=8, lines=200, files_touched=5, last_commit_days_ago=14),
            ], bus_factor=2),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            export_csv([], areas, [], {}, {}, out)
            rows = _read_csv(out / "experts.csv")

            assert len(rows) == 2
            assert rows[0]["area"] == "auth"
            assert rows[0]["login"] == "alice"
            assert rows[0]["score"] == "5.0"
            assert rows[0]["bus_factor"] == "2"
            assert rows[1]["login"] == "bob"

    def test_teams_csv_content(self):
        communities = [{"alice", "bob"}, {"charlie"}]
        team_labels = {0: "Backend", 1: "Frontend"}
        areas = [
            _area("api", [
                AreaExpert(login="alice", score=5.0, commits=20, lines=500, files_touched=10, last_commit_days_ago=3),
            ]),
        ]
        member_to_community = {"alice": 0, "bob": 0, "charlie": 1}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            export_csv([], areas, communities, team_labels, member_to_community, out)
            rows = _read_csv(out / "teams.csv")

            assert len(rows) == 2
            assert rows[0]["team_id"] == "0"
            assert rows[0]["label"] == "Backend"
            assert "alice" in rows[0]["members"]
            assert "bob" in rows[0]["members"]
            assert rows[0]["top_areas"] == "api"
            assert rows[1]["label"] == "Frontend"

    def test_empty_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            export_csv([], [], [], {}, {}, out)

            for name in ("edges.csv", "experts.csv", "teams.csv"):
                rows = _read_csv(out / name)
                assert rows == []

    def test_output_dir_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "deep" / "out"
            export_csv([], [], [], {}, {}, out)
            assert out.is_dir()
