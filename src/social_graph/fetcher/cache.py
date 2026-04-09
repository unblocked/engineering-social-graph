"""SQLite cache for GitHub API responses.

Stores PRs and interactions locally so subsequent runs only fetch new data.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from typing import TYPE_CHECKING

from social_graph.models import (
    Commit,
    Interaction,
    InteractionType,
    PullRequest,
    PullRequestState,
    SocialEdge,
    User,
)

if TYPE_CHECKING:
    from social_graph.graph.code_expertise import CodeArea

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pull_requests (
    id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    number INTEGER NOT NULL,
    author_json TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    merged_at TEXT,
    additions INTEGER DEFAULT 0,
    deletions INTEGER DEFAULT 0,
    UNIQUE(repo, number)
);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pr_id TEXT NOT NULL,
    user_json TEXT NOT NULL,
    type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (pr_id) REFERENCES pull_requests(id)
);

CREATE TABLE IF NOT EXISTS sync_meta (
    repo TEXT PRIMARY KEY,
    last_synced_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pr_repo ON pull_requests(repo);
CREATE INDEX IF NOT EXISTS idx_pr_state ON pull_requests(state);
CREATE INDEX IF NOT EXISTS idx_interactions_pr ON interactions(pr_id);

CREATE TABLE IF NOT EXISTS social_edges (
    author_login TEXT NOT NULL,
    reviewer_login TEXT NOT NULL,
    weight REAL NOT NULL,
    author_json TEXT NOT NULL,
    reviewer_json TEXT NOT NULL,
    PRIMARY KEY (author_login, reviewer_login)
);

CREATE TABLE IF NOT EXISTS lines_by_user (
    login TEXT PRIMARY KEY,
    lines INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS code_areas (
    path TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    bus_factor INTEGER NOT NULL,
    total_commits INTEGER NOT NULL,
    total_lines INTEGER NOT NULL,
    contributor_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS area_experts (
    area_path TEXT NOT NULL,
    login TEXT NOT NULL,
    score REAL NOT NULL,
    commits INTEGER NOT NULL,
    lines INTEGER NOT NULL,
    files_touched INTEGER NOT NULL,
    last_commit_days_ago INTEGER NOT NULL,
    PRIMARY KEY (area_path, login)
);

CREATE TABLE IF NOT EXISTS communities (
    login TEXT PRIMARY KEY,
    community_idx INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS team_labels (
    community_idx INTEGER PRIMARY KEY,
    label TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS build_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS commits (
    sha TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    author_json TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    files_json TEXT NOT NULL,
    coauthors_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_commits_repo ON commits(repo);
"""


def _user_to_json(user: User) -> str:
    return json.dumps({
        "id": user.id, "login": user.login, "name": user.name, "is_bot": user.is_bot,
        "avatar_url": user.avatar_url, "html_url": user.html_url,
        "company": user.company, "location": user.location,
    })


def _user_from_json(data: str) -> User:
    d = json.loads(data)
    return User(
        id=d["id"], login=d["login"], name=d.get("name"), is_bot=d.get("is_bot", False),
        avatar_url=d.get("avatar_url"), html_url=d.get("html_url"),
        company=d.get("company"), location=d.get("location"),
    )


def _dt_to_str(dt: datetime) -> str:
    return dt.isoformat()


def _str_to_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class PRCache:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def get_last_synced(self, repo: str) -> datetime | None:
        row = self._conn.execute("SELECT last_synced_at FROM sync_meta WHERE repo = ?", (repo,)).fetchone()
        return _str_to_dt(row[0]) if row else None

    def set_last_synced(self, repo: str, dt: datetime) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sync_meta (repo, last_synced_at) VALUES (?, ?)",
            (repo, _dt_to_str(dt)),
        )
        self._conn.commit()

    def upsert_pr(self, pr: PullRequest) -> None:
        pr_id = f"{pr.repo}#{pr.number}"
        self._conn.execute(
            """INSERT OR REPLACE INTO pull_requests
               (id, repo, number, author_json, state, created_at, updated_at, merged_at, additions, deletions)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pr_id,
                pr.repo,
                pr.number,
                _user_to_json(pr.author),
                pr.state.value,
                _dt_to_str(pr.created_at),
                _dt_to_str(pr.updated_at),
                _dt_to_str(pr.merged_at) if pr.merged_at else None,
                pr.additions,
                pr.deletions,
            ),
        )
        # Replace interactions for this PR
        self._conn.execute("DELETE FROM interactions WHERE pr_id = ?", (pr_id,))
        for interaction in pr.interactions:
            self._conn.execute(
                "INSERT INTO interactions (pr_id, user_json, type, created_at) VALUES (?, ?, ?, ?)",
                (
                    pr_id,
                    _user_to_json(interaction.user),
                    interaction.type.value,
                    _dt_to_str(interaction.created_at),
                ),
            )
        self._conn.commit()

    def get_prs(self, repo: str, state: PullRequestState | None = None) -> list[PullRequest]:
        if state:
            rows = self._conn.execute(
                "SELECT id, repo, number, author_json, state, created_at, updated_at, merged_at, additions, deletions "
                "FROM pull_requests WHERE repo = ? AND state = ?",
                (repo, state.value),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, repo, number, author_json, state, created_at, updated_at, merged_at, additions, deletions "
                "FROM pull_requests WHERE repo = ?",
                (repo,),
            ).fetchall()

        prs = []
        for row in rows:
            pr_id = row[0]
            interactions = self._get_interactions(pr_id)
            prs.append(
                PullRequest(
                    number=row[2],
                    repo=row[1],
                    author=_user_from_json(row[3]),
                    state=PullRequestState(row[4]),
                    created_at=_str_to_dt(row[5]),
                    updated_at=_str_to_dt(row[6]),
                    merged_at=_str_to_dt(row[7]) if row[7] else None,
                    additions=row[8] or 0,
                    deletions=row[9] or 0,
                    interactions=interactions,
                )
            )
        return prs

    def get_all_prs(self, state: PullRequestState | None = None) -> list[PullRequest]:
        if state:
            rows = self._conn.execute(
                "SELECT DISTINCT repo FROM pull_requests WHERE state = ?", (state.value,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT DISTINCT repo FROM pull_requests").fetchall()
        prs = []
        for (repo,) in rows:
            prs.extend(self.get_prs(repo, state))
        return prs

    def _get_interactions(self, pr_id: str) -> list[Interaction]:
        rows = self._conn.execute(
            "SELECT user_json, type, created_at FROM interactions WHERE pr_id = ?",
            (pr_id,),
        ).fetchall()
        return [
            Interaction(
                user=_user_from_json(row[0]),
                type=InteractionType(row[1]),
                created_at=_str_to_dt(row[2]),
            )
            for row in rows
        ]

    def get_merged_pr_counts(self, repo: str) -> dict[User, int]:
        rows = self._conn.execute(
            "SELECT author_json, COUNT(*) FROM pull_requests WHERE repo = ? AND state = ? GROUP BY author_json",
            (repo, PullRequestState.MERGED.value),
        ).fetchall()
        return {_user_from_json(row[0]): row[1] for row in rows}

    # --- Build data save/load methods ---

    def save_social_edges(self, edges: list[SocialEdge]) -> None:
        """Persist social edges to the cache, replacing any existing data."""
        self._conn.execute("DELETE FROM social_edges")
        for e in edges:
            self._conn.execute(
                "INSERT INTO social_edges (author_login, reviewer_login, weight, author_json, reviewer_json) VALUES (?,?,?,?,?)",
                (e.author.login, e.reviewer.login, e.weight, _user_to_json(e.author), _user_to_json(e.reviewer)),
            )
        self._conn.commit()

    def load_social_edges(self) -> list[SocialEdge]:
        """Load all cached social edges."""
        rows = self._conn.execute("SELECT author_json, reviewer_json, weight FROM social_edges").fetchall()
        return [SocialEdge(author=_user_from_json(r[0]), reviewer=_user_from_json(r[1]), weight=r[2]) for r in rows]

    def save_lines_by_user(self, data: dict[str, int]) -> None:
        self._conn.execute("DELETE FROM lines_by_user")
        for login, lines in data.items():
            self._conn.execute("INSERT INTO lines_by_user (login, lines) VALUES (?,?)", (login, lines))
        self._conn.commit()

    def load_lines_by_user(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT login, lines FROM lines_by_user").fetchall()
        return {r[0]: r[1] for r in rows}

    def save_code_areas(self, areas: list[CodeArea]) -> None:
        """Persist code area expertise data, replacing any existing data."""
        self._conn.execute("DELETE FROM code_areas")
        self._conn.execute("DELETE FROM area_experts")
        for area in areas:
            self._conn.execute(
                "INSERT INTO code_areas (path, name, bus_factor, total_commits, total_lines, contributor_count) VALUES (?,?,?,?,?,?)",
                (area.path, area.name, area.bus_factor, area.total_commits, area.total_lines, area.contributor_count),
            )
            for expert in area.experts:
                self._conn.execute(
                    "INSERT INTO area_experts (area_path, login, score, commits, lines, files_touched, last_commit_days_ago) VALUES (?,?,?,?,?,?,?)",
                    (area.path, expert.login, expert.score, expert.commits, expert.lines, expert.files_touched, expert.last_commit_days_ago),
                )
        self._conn.commit()

    def load_code_areas(self) -> list[CodeArea]:
        """Load all cached code areas with their experts."""
        from social_graph.graph.code_expertise import CodeArea, AreaExpert
        area_rows = self._conn.execute("SELECT path, name, bus_factor, total_commits, total_lines, contributor_count FROM code_areas").fetchall()
        areas = []
        for r in area_rows:
            expert_rows = self._conn.execute(
                "SELECT login, score, commits, lines, files_touched, last_commit_days_ago FROM area_experts WHERE area_path = ? ORDER BY score DESC",
                (r[0],),
            ).fetchall()
            experts = [AreaExpert(login=e[0], score=e[1], commits=e[2], lines=e[3], files_touched=e[4], last_commit_days_ago=e[5]) for e in expert_rows]
            areas.append(CodeArea(path=r[0], name=r[1], bus_factor=r[2], total_commits=r[3], total_lines=r[4], contributor_count=r[5], experts=experts))
        return areas

    def save_communities(self, communities: list[set[str]]) -> None:
        self._conn.execute("DELETE FROM communities")
        for idx, members in enumerate(communities):
            for login in members:
                self._conn.execute("INSERT INTO communities (login, community_idx) VALUES (?,?)", (login, idx))
        self._conn.commit()

    def load_communities(self) -> tuple[list[set[str]], dict[str, int]]:
        rows = self._conn.execute("SELECT login, community_idx FROM communities").fetchall()
        member_to_community = {r[0]: r[1] for r in rows}
        communities_dict: dict[int, set[str]] = {}
        for login, idx in member_to_community.items():
            communities_dict.setdefault(idx, set()).add(login)
        communities = [communities_dict[i] for i in sorted(communities_dict.keys())]
        return communities, member_to_community

    def save_team_labels(self, labels: dict[int, str]) -> None:
        self._conn.execute("DELETE FROM team_labels")
        for idx, label in labels.items():
            self._conn.execute("INSERT INTO team_labels (community_idx, label) VALUES (?,?)", (idx, label))
        self._conn.commit()

    def load_team_labels(self) -> dict[int, str]:
        rows = self._conn.execute("SELECT community_idx, label FROM team_labels").fetchall()
        return {r[0]: r[1] for r in rows}

    def save_build_meta(self, meta: dict[str, str]) -> None:
        self._conn.execute("DELETE FROM build_meta")
        for key, value in meta.items():
            self._conn.execute("INSERT INTO build_meta (key, value) VALUES (?,?)", (key, value))
        self._conn.commit()

    def load_build_meta(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM build_meta").fetchall()
        return {r[0]: r[1] for r in rows}

    def has_build_data(self) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM build_meta WHERE key = 'built_at' LIMIT 1"
        ).fetchone()
        return row is not None


    def upsert_commit(self, commit: Commit) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO commits
               (sha, repo, author_json, committed_at, files_json, coauthors_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                commit.sha,
                commit.repo,
                _user_to_json(commit.author),
                _dt_to_str(commit.committed_at),
                json.dumps(commit.files),
                json.dumps([_user_to_json(user) for user in commit.coauthors]),
            ),
        )
        self._conn.commit()

    def delete_commits(self, repo: str) -> None:
        self._conn.execute("DELETE FROM commits WHERE repo = ?", (repo,))
        self._conn.commit()

    def get_all_commits(self, repo: str | None = None) -> list[Commit]:
        if repo:
            rows = self._conn.execute(
                "SELECT sha, repo, author_json, committed_at, files_json, coauthors_json FROM commits WHERE repo = ?",
                (repo,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT sha, repo, author_json, committed_at, files_json, coauthors_json FROM commits"
            ).fetchall()

        commits = []
        for sha, repo_name, author_json, committed_at, files_json, coauthors_json in rows:
            coauthors = tuple(_user_from_json(data) for data in json.loads(coauthors_json))
            commits.append(
                Commit(
                    sha=sha,
                    repo=repo_name,
                    author=_user_from_json(author_json),
                    committed_at=_str_to_dt(committed_at),
                    files=tuple(json.loads(files_json)),
                    coauthors=coauthors,
                )
            )
        return commits
