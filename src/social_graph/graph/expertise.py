"""Expertise graph builder — procedural, no embeddings.

Three modes:
1. File/directory expertise via git log
2. Repo expertise via merged PR counts
3. Domain expertise via directory-based mapping + aggregated git log
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from social_graph.config import DEFAULT_MODEL
from social_graph.decay import days_since, exponential_time_decay
from social_graph.models import ExpertRanking, User


def experts_for_path(
    repo_path: str | Path,
    target_path: str,
    top_n: int = 5,
    apply_decay: bool = True,
    now: datetime | None = None,
) -> list[ExpertRanking]:
    """Find top contributors to a file or directory using git log.

    Scores by commit count with optional time decay weighting.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    repo_path = Path(repo_path)
    commits = _git_log_for_path(repo_path, target_path)

    # Score by (optionally time-decayed) commit count per author
    scores: dict[str, float] = defaultdict(float)
    emails: dict[str, str] = {}

    for author_name, author_email, date_str in commits:
        key = author_email.lower()
        emails[key] = author_email
        if apply_decay:
            commit_date = _parse_git_date(date_str)
            delta = days_since(commit_date, now)
            scores[key] += exponential_time_decay(delta)
        else:
            scores[key] += 1.0

    # Sort and take top N
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [
        ExpertRanking(
            user=User(id=email, login=email, name=_find_name(email, commits)),
            score=score,
            domain=target_path,
        )
        for email, score in ranked
    ]


def experts_for_path_by_churn(
    repo_path: str | Path,
    target_path: str,
    top_n: int = 5,
) -> list[ExpertRanking]:
    """Find top contributors by lines changed (insertions + deletions).

    Matches the approach in git_contributors_extractor.py from the existing pipeline.
    """
    repo_path = Path(repo_path)
    churn = _git_numstat_for_path(repo_path, target_path)

    ranked = sorted(churn.items(), key=lambda x: x[1][1], reverse=True)[:top_n]

    return [
        ExpertRanking(
            user=User(id=email, login=email, name=name),
            score=score,
            domain=target_path,
        )
        for email, (name, score) in ranked
    ]


def experts_for_repo(
    merged_pr_counts: dict[User, int],
    repo_name: str,
    top_n: int = 3,
) -> list[ExpertRanking]:
    """Rank contributors by merged PR count. Matches ScmExpertService.expertsForRepo()."""
    ranked = sorted(merged_pr_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        ExpertRanking(user=user, score=float(count), domain=repo_name)
        for user, count in ranked
    ]


def experts_for_domain(
    repo_path: str | Path,
    domain_name: str,
    domain_map: dict[str, list[str]],
    top_n: int = 5,
    now: datetime | None = None,
) -> list[ExpertRanking]:
    """Find experts for a domain by aggregating git log across mapped paths."""
    paths = domain_map.get(domain_name, [])
    if not paths:
        return []

    if now is None:
        now = datetime.now(timezone.utc)

    # Aggregate scores across all paths in this domain
    combined_scores: dict[str, float] = defaultdict(float)
    names: dict[str, str] = {}

    for path in paths:
        rankings = experts_for_path(repo_path, path, top_n=100, now=now)
        for r in rankings:
            combined_scores[r.user.id] += r.score
            if r.user.name:
                names[r.user.id] = r.user.name

    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [
        ExpertRanking(
            user=User(id=email, login=email, name=names.get(email)),
            score=score,
            domain=domain_name,
        )
        for email, score in ranked
    ]


def load_domain_map(path: Path) -> dict[str, list[str]]:
    """Load a domain→paths mapping from a JSON file."""
    with open(path) as f:
        return json.load(f)


def auto_generate_domain_map(
    repo_path: str | Path,
    api_key: str,
) -> dict[str, list[str]]:
    """Use Claude Haiku to auto-label directories into domain areas.

    Sends only the directory tree (names, no file contents) — a single cheap call.
    """
    import anthropic

    repo_path = Path(repo_path)
    tree = _get_directory_tree(repo_path, max_depth=3)

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": (
                    "Given this repository directory structure, identify the main domain areas "
                    "and map each domain to the relevant directory paths. Return ONLY valid JSON "
                    "in this format: {\"domain_name\": [\"path/to/dir1\", \"path/to/dir2\"]}\n\n"
                    "Directory tree:\n" + tree
                ),
            }
        ],
    )
    text = message.content[0].text
    # Extract JSON from response (handle markdown code blocks)
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def _get_directory_tree(repo_path: Path, max_depth: int = 3) -> str:
    """Get a text representation of the directory tree."""
    lines: list[str] = []
    _walk_tree(repo_path, repo_path, lines, depth=0, max_depth=max_depth)
    return "\n".join(lines)


def _walk_tree(root: Path, current: Path, lines: list[str], depth: int, max_depth: int) -> None:
    if depth > max_depth:
        return
    # Skip hidden dirs and common noise
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".vscode", "dist", "build"}
    indent = "  " * depth
    rel = current.relative_to(root)
    lines.append(f"{indent}{rel.name or '.'}/")
    try:
        children = sorted(current.iterdir())
    except PermissionError:
        return
    dirs = [c for c in children if c.is_dir() and c.name not in skip]
    for d in dirs:
        _walk_tree(root, d, lines, depth + 1, max_depth)


def _git_log_for_path(repo_path: Path, target: str) -> list[tuple[str, str, str]]:
    """Run git log and return (author_name, author_email, iso_date) tuples."""
    result = subprocess.run(
        ["git", "log", "--format=%aN\t%aE\t%aI", "--follow", "--", target],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    entries = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            entries.append((parts[0], parts[1], parts[2]))
    return entries


def _git_numstat_for_path(repo_path: Path, target: str) -> dict[str, tuple[str, float]]:
    """Get insertions+deletions per author via git log --numstat."""
    result = subprocess.run(
        ["git", "log", "--format=COMMIT\t%aN\t%aE", "--numstat", "--", target],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}

    scores: dict[str, float] = defaultdict(float)
    names: dict[str, str] = {}
    current_author = ""
    current_email = ""

    for line in result.stdout.strip().splitlines():
        if line.startswith("COMMIT\t"):
            parts = line.split("\t", 2)
            if len(parts) == 3:
                current_author = parts[1]
                current_email = parts[2].lower()
                names[current_email] = current_author
        elif line.strip() and current_email:
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    added = int(parts[0]) if parts[0] != "-" else 0
                    deleted = int(parts[1]) if parts[1] != "-" else 0
                    scores[current_email] += added + deleted
                except ValueError:
                    pass

    return {email: (names.get(email, ""), score) for email, score in scores.items()}


def _parse_git_date(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str)


def _find_name(email: str, commits: list[tuple[str, str, str]]) -> str | None:
    for name, e, _ in commits:
        if e.lower() == email.lower():
            return name
    return None
