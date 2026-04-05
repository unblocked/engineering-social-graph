"""Code expertise detection — purely procedural, no LLM.

Algorithm:
1. Discover code areas from the repo's directory structure
2. For each area, run git log to get per-author commit data
3. Score each author using a composite of four factors, weighted by
   empirical findings from Cury et al. (ESEM 2022):
   - Recency (days since last commit) — strongest signal after first authorship
   - Lines changed (additions + deletions), time-decayed
   - File breadth (number of distinct files touched)
   - Commit count, time-decayed
4. Rank authors per area, compute bus factor using coverage-based iterative
   removal (Avelino et al., ICPC 2016)

The directory tree IS the topic taxonomy — no embeddings needed.

References:
  - Cury et al., "Identifying Source Code File Experts," ESEM 2022. arXiv:2208.07501
  - Avelino et al., "A Novel Approach for Estimating Truck Factors," ICPC 2016. arXiv:1604.06766
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from social_graph.decay import exponential_time_decay


@dataclass
class AreaExpert:
    login: str
    score: float
    commits: int
    lines: int
    files_touched: int
    last_commit_days_ago: int


@dataclass
class CodeArea:
    path: str
    name: str
    experts: list[AreaExpert] = field(default_factory=list)
    bus_factor: int = 0
    total_commits: int = 0
    total_lines: int = 0
    contributor_count: int = 0


# Weights for the composite expertise score.
# Informed by Cury et al. (ESEM 2022): recency (NumDays) has the highest
# correlation with expertise after first authorship; commit count (NumCommits)
# was not selected in their final DOE model — lines added (Adds) was preferred.
WEIGHT_RECENCY = 0.35
WEIGHT_LINES = 0.30
WEIGHT_BREADTH = 0.20
WEIGHT_COMMITS = 0.15

# Bus factor coverage threshold (Avelino et al., 2016): the project stalls when
# fewer than this fraction of files have at least one knowledgeable author.
BUS_FACTOR_COVERAGE_THRESHOLD = 0.50

# A developer is considered an "author" of a file area if their score is at
# least this fraction of the top contributor's score. Adapted from Avelino et
# al.'s 0.75 DOA threshold — lowered to 0.5 because our composite scores are
# normalized to [0,1] and the gap between #1 and #2 is typically larger than
# in the DOA model's regression-based scores.
BUS_FACTOR_AUTHOR_THRESHOLD = 0.50


def detect_areas(
    repo_path: str | Path,
    max_depth: int = 2,
    min_commits: int = 5,
    since: str = "6 months ago",
) -> list[CodeArea]:
    """Discover meaningful code areas and rank experts for each.

    Walks the directory tree to find areas with enough commit activity,
    then scores contributors per area using a composite metric.
    """
    repo_path = Path(repo_path)
    now = datetime.now(timezone.utc)

    # Find candidate directories
    candidates = _find_candidate_dirs(repo_path, max_depth)

    areas: list[CodeArea] = []
    for dir_path in candidates:
        rel_path = str(dir_path.relative_to(repo_path))
        raw = _git_log_for_area(repo_path, rel_path, since)
        if not raw:
            continue

        # Parse into per-author stats
        author_stats = _parse_git_log(raw, now)
        if not author_stats:
            continue

        total_commits = sum(s["commits"] for s in author_stats.values())
        if total_commits < min_commits:
            continue

        # Score and rank
        experts = _score_experts(author_stats)
        total_lines = sum(s["lines"] for s in author_stats.values())

        # Bus factor: count of contributors with score >= 50% of the top
        # contributor. Adapted from Avelino et al. (2016) — we use author count
        # rather than iterative removal because it better reflects knowledge
        # distribution in area-level (vs file-level) analysis.
        bus_factor = _count_knowledgeable_authors(experts)

        # Derive a readable name from the path
        name = _area_name(rel_path)

        areas.append(CodeArea(
            path=rel_path,
            name=name,
            experts=experts,
            bus_factor=bus_factor,
            total_commits=total_commits,
            total_lines=total_lines,
            contributor_count=len(author_stats),
        ))

    # Sort by total activity
    areas.sort(key=lambda a: a.total_commits, reverse=True)
    return areas


def experts_for_area(
    repo_path: str | Path,
    area_path: str,
    since: str = "6 months ago",
    top_n: int = 10,
) -> list[AreaExpert]:
    """Get ranked experts for a specific directory path."""
    repo_path = Path(repo_path)
    now = datetime.now(timezone.utc)

    raw = _git_log_for_area(repo_path, area_path, since)
    if not raw:
        return []

    author_stats = _parse_git_log(raw, now)
    return _score_experts(author_stats)[:top_n]


def _find_candidate_dirs(repo_path: Path, max_depth: int) -> list[Path]:
    """Find directories that are likely meaningful code areas.

    Skips hidden dirs, build artifacts, and other noise.
    """
    skip = {
        ".git", "node_modules", "__pycache__", ".venv", "venv", ".idea",
        ".vscode", "dist", "build", ".gradle", ".next", "target",
        "vendor", ".cache", "coverage", ".pytest_cache",
    }

    candidates: list[Path] = []

    def walk(current: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(current.iterdir())
        except PermissionError:
            return
        dirs = [c for c in children if c.is_dir() and c.name not in skip and not c.name.startswith(".")]
        for d in dirs:
            candidates.append(d)
            walk(d, depth + 1)

    walk(repo_path, 0)
    return candidates


def _git_log_for_area(repo_path: Path, area_path: str, since: str) -> str:
    """Run git log --numstat for a specific directory."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), "log",
         "--numstat", "--format=COMMIT\t%aE\t%aI",
         f"--since={since}", "--", area_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


@dataclass
class _AuthorAreaStats:
    commits: int = 0
    lines: int = 0
    files: set = field(default_factory=set)
    decayed_lines: float = 0.0
    decayed_commits: float = 0.0
    last_commit_days: int = 999


def _parse_git_log(raw: str, now: datetime) -> dict[str, dict]:
    """Parse git log output into per-author stats with time decay."""
    stats: dict[str, _AuthorAreaStats] = defaultdict(_AuthorAreaStats)
    current_author = ""
    current_date = now

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("COMMIT\t"):
            parts = line.split("\t", 2)
            if len(parts) == 3:
                email = parts[1]
                # Normalize: strip domain, merge GitHub noreply aliases
                login = email.split("@")[0]
                if "+" in login:
                    login = login.split("+", 1)[1]
                current_author = login

                date_str = parts[2]
                try:
                    current_date = datetime.fromisoformat(date_str)
                    if current_date.tzinfo is None:
                        current_date = current_date.replace(tzinfo=timezone.utc)
                except ValueError:
                    current_date = now

                delta_days = (now - current_date).days
                decay = exponential_time_decay(delta_days)

                s = stats[current_author]
                s.commits += 1
                s.decayed_commits += decay
                s.last_commit_days = min(s.last_commit_days, delta_days)
        else:
            parts = line.split("\t")
            if len(parts) == 3 and current_author:
                added_str, deleted_str, file_path = parts
                if added_str != "-" and deleted_str != "-":
                    added = int(added_str)
                    deleted = int(deleted_str)
                    total = added + deleted

                    delta_days = (now - current_date).days
                    decay = exponential_time_decay(delta_days)

                    s = stats[current_author]
                    s.lines += total
                    s.decayed_lines += total * decay
                    s.files.add(file_path)

    # Convert to plain dicts for scoring
    return {
        login: {
            "commits": s.commits,
            "lines": s.lines,
            "files_touched": len(s.files),
            "decayed_lines": s.decayed_lines,
            "decayed_commits": s.decayed_commits,
            "last_commit_days": s.last_commit_days,
            "_files": s.files,  # preserved for bus factor file-level analysis
        }
        for login, s in stats.items()
    }


def _score_experts(author_stats: dict[str, dict]) -> list[AreaExpert]:
    """Compute composite expertise scores and rank.

    Score = weighted combination of (Cury et al., 2022):
      - Recency (35%) — inverse of days since last commit; strongest signal
      - Decayed lines changed (30%) — volume of work, time-weighted
      - File breadth (20%) — how much of the area they've touched
      - Decayed commit count (15%) — frequency of engagement, time-weighted
    """
    if not author_stats:
        return []

    # Normalize each dimension to [0, 1]
    max_lines = max(s["decayed_lines"] for s in author_stats.values()) or 1
    max_commits = max(s["decayed_commits"] for s in author_stats.values()) or 1
    max_files = max(s["files_touched"] for s in author_stats.values()) or 1
    max_recency = max(1.0 / (s["last_commit_days"] + 1) for s in author_stats.values()) or 1

    experts: list[AreaExpert] = []
    for login, s in author_stats.items():
        norm_lines = s["decayed_lines"] / max_lines
        norm_commits = s["decayed_commits"] / max_commits
        norm_files = s["files_touched"] / max_files
        norm_recency = (1.0 / (s["last_commit_days"] + 1)) / max_recency

        score = (
            WEIGHT_LINES * norm_lines
            + WEIGHT_COMMITS * norm_commits
            + WEIGHT_BREADTH * norm_files
            + WEIGHT_RECENCY * norm_recency
        )

        experts.append(AreaExpert(
            login=login,
            score=round(score, 4),
            commits=s["commits"],
            lines=s["lines"],
            files_touched=s["files_touched"],
            last_commit_days_ago=s["last_commit_days"],
        ))

    experts.sort(key=lambda e: e.score, reverse=True)
    return experts


def _count_knowledgeable_authors(experts: list[AreaExpert]) -> int:
    """Count contributors with score >= 50% of top, as a bus factor estimate.

    A simpler alternative to file-level iterative removal that works well for
    area-level analysis where the question is "how many people know this area?"
    """
    if not experts:
        return 0
    top_score = experts[0].score
    if top_score == 0:
        return 0
    threshold = top_score * BUS_FACTOR_AUTHOR_THRESHOLD
    return len([e for e in experts if e.score >= threshold])


def _compute_bus_factor(
    author_stats: dict[str, dict],
    experts: list[AreaExpert],
) -> int:
    """Compute bus factor using coverage-based iterative removal.

    Based on Avelino et al. (ICPC 2016): iteratively remove the top contributor
    and check how many files still have at least one knowledgeable author.
    The bus factor is the number of removals before coverage drops below 50%.

    A developer is considered an "author" of the area's files if their score
    is >= 75% of the top contributor's score (following the DOA threshold
    from the paper).
    """
    if not experts or not author_stats:
        return 0

    # Determine which developers are "authors" (score >= 75% of top)
    top_score = experts[0].score
    if top_score == 0:
        return 0
    author_threshold = top_score * BUS_FACTOR_AUTHOR_THRESHOLD

    # Build a mapping: file → set of authors who are knowledgeable about it
    file_authors: dict[str, set[str]] = defaultdict(set)
    for login, stats in author_stats.items():
        # Check if this developer qualifies as an "author"
        expert = next((e for e in experts if e.login == login), None)
        if expert and expert.score >= author_threshold:
            for f in stats.get("_files", set()):
                file_authors[f].add(login)

    # If we don't have file-level data, fall back to score-based estimate
    if not file_authors:
        return len([e for e in experts if e.score >= author_threshold])

    total_files = len(file_authors)
    if total_files == 0:
        return 0

    # Iterative removal: remove top contributor by file count, check coverage
    remaining_authors = {e.login for e in experts if e.score >= author_threshold}
    bus_factor = 0

    while remaining_authors:
        # Find author covering the most files
        author_file_counts = {}
        for login in remaining_authors:
            count = sum(1 for authors in file_authors.values() if login in authors)
            author_file_counts[login] = count

        top_author = max(author_file_counts, key=author_file_counts.get)
        remaining_authors.discard(top_author)
        bus_factor += 1

        # Check coverage: how many files still have at least one author?
        covered = sum(1 for authors in file_authors.values() if authors & remaining_authors)
        coverage = covered / total_files

        if coverage < BUS_FACTOR_COVERAGE_THRESHOLD:
            break

    return bus_factor


def _area_name(path: str) -> str:
    """Derive a human-readable area name from a directory path."""
    parts = path.split("/")
    last = parts[-1]
    # Strip common prefixes
    for prefix in ("lib-", "client-", "ingest-", "service-"):
        if last.startswith(prefix):
            last = last[len(prefix):]
    # Convert kebab-case to title
    return last.replace("-", " ").replace("_", " ").title()
