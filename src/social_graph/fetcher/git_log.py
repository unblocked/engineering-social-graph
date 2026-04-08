"""Fetch commit data from local git log for commit-based social graph mode."""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from typing import Callable

from social_graph.models import Commit, User

# Matches "Co-authored-by: Name <email>" (case-insensitive)
_CO_AUTHOR_RE = re.compile(r"Co-authored-by:\s*(.+?)\s*<(.+?)>", re.IGNORECASE)

# Delimiter between commits in git log output
_COMMIT_DELIM = "---COMMIT_DELIM---"


def _login_from_email(email: str) -> str:
    """Extract a login-style identifier from an email address.

    Handles GitHub noreply format: "12345+username@users.noreply.github.com"
    """
    prefix = email.split("@")[0]
    if "+" in prefix:
        prefix = prefix.split("+", 1)[1]
    return prefix


def _user_from_git(name: str, email: str) -> User:
    """Create a User from git author name/email."""
    login = _login_from_email(email)
    return User(id=login, login=login, name=name)


def fetch_repo_commits(
    repo_path: str,
    repo_name: str,
    since: str = "6 months ago",
    log: Callable[[str], None] = print,
) -> list[Commit]:
    """Parse local git log into Commit objects.

    Extracts commit metadata, file lists, and Co-authored-by trailers.
    Filters out merge commits (2+ parents).
    """
    # Format: delimiter, hash, author name, author email, ISO date, parent hashes, body
    fmt = f"{_COMMIT_DELIM}%H%n%aN%n%aE%n%aI%n%P%n%B"
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "log", f"--format={fmt}", "--name-only", f"--since={since}"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        log("  [red]git log timed out after 120s[/red]")
        return []
    if result.returncode != 0:
        log(f"  [red]git log failed:[/red] {result.stderr.strip()}")
        return []

    raw_blocks = result.stdout.split(_COMMIT_DELIM)
    commits: list[Commit] = []

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.split("\n")
        if len(lines) < 5:
            continue

        sha = lines[0].strip()
        author_name = lines[1].strip()
        author_email = lines[2].strip()
        date_str = lines[3].strip()
        parents = lines[4].strip()

        # Skip merge commits (2+ parents)
        if len(parents.split()) > 1:
            continue

        # Parse date
        try:
            committed_at = datetime.fromisoformat(date_str)
            if committed_at.tzinfo is None:
                committed_at = committed_at.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        # Remaining lines: body + blank + file names
        rest = lines[5:]

        # Find the last blank line — everything after it is the file list
        last_blank = -1
        for i in range(len(rest) - 1, -1, -1):
            if not rest[i].strip():
                last_blank = i
                break

        if last_blank == -1:
            body_lines = [line.strip() for line in rest if line.strip()]
            file_lines: list[str] = []
        else:
            body_lines = [line.strip() for line in rest[:last_blank] if line.strip()]
            file_lines = [line.strip() for line in rest[last_blank + 1:] if line.strip()]

        # Parse co-authors from body
        body = "\n".join(body_lines)
        coauthors: list[User] = []
        for match in _CO_AUTHOR_RE.finditer(body):
            ca_name, ca_email = match.group(1), match.group(2)
            coauthors.append(_user_from_git(ca_name, ca_email))

        author = _user_from_git(author_name, author_email)

        # Filter out empty commits (no files changed)
        if not file_lines:
            continue

        commits.append(Commit(
            sha=sha,
            repo=repo_name,
            author=author,
            committed_at=committed_at,
            files=tuple(file_lines),
            coauthors=tuple(coauthors),
        ))

    log(f"  Parsed {len(commits)} commits (excluding merges).")
    return commits
