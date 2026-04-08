"""GitHub API client — fetches PRs and interactions using GraphQL.

Uses a single nested GraphQL query to fetch PRs with reviews and comments
in one call per page. This is dramatically faster than REST and uses
far fewer rate limit points.
"""

from __future__ import annotations

from datetime import datetime, timezone

import requests as http_requests
from rich.console import Console

from social_graph.models import (
    Interaction,
    InteractionType,
    PullRequest,
    PullRequestState,
    User,
)

console = Console()

# GraphQL: fetch PRs with nested reviews, review comments, and issue comments.
# GitHub GraphQL rate limit is 5000 points/hour; this query costs ~1 point per node.
# Fetching 50 PRs with up to 50 reviews and 100 comments each ≈ ~5 points per page.
_USER_FRAGMENT = """
fragment userFields on Actor {
  __typename
  login
  avatarUrl
  url
  ... on User {
    databaseId
    name
    company
    location
  }
  ... on Bot {
    databaseId
  }
  ... on EnterpriseUserAccount {
    name
  }
  ... on Mannequin {
    databaseId
  }
}
"""

PR_QUERY = _USER_FRAGMENT + """
query($owner: String!, $repo: String!, $cursor: String, $pageSize: Int!) {
  rateLimit {
    remaining
    cost
    limit
    resetAt
  }
  repository(owner: $owner, name: $repo) {
    pullRequests(
      first: $pageSize,
      after: $cursor,
      states: [MERGED, CLOSED],
      orderBy: {field: UPDATED_AT, direction: DESC}
    ) {
      totalCount
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        number
        title
        state
        createdAt
        updatedAt
        mergedAt
        additions
        deletions
        author { ...userFields }
        reviews(first: 20) {
          nodes {
            author { ...userFields }
            state
            submittedAt
          }
        }
        reviewThreads(first: 30) {
          nodes {
            comments(first: 10) {
              nodes {
                author { ...userFields }
                createdAt
              }
            }
          }
        }
        comments(first: 20) {
          nodes {
            author { ...userFields }
            createdAt
          }
        }
      }
    }
  }
}
"""


def _log(msg: str) -> None:
    console.print(msg)


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_author(author: dict | None) -> User | None:
    if not author:
        return None
    db_id = author.get("databaseId")
    login = author.get("login", "ghost")
    typename = author.get("__typename", "User")
    return User(
        id=str(db_id) if db_id else login,
        login=login,
        name=author.get("name"),
        is_bot=typename == "Bot",
        avatar_url=author.get("avatarUrl"),
        html_url=author.get("url"),
        company=author.get("company"),
        location=author.get("location"),
    )


class GitHubFetcher:
    """Fetches PR data using GitHub's GraphQL API for maximum speed."""

    def __init__(self, token: str) -> None:
        self._session = http_requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })
        self._gql_url = "https://api.github.com/graphql"

    def fetch_repo_prs(
        self,
        repo_full_name: str,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[PullRequest]:
        """Fetch PRs with nested reviews/comments via a single GraphQL query per page."""
        owner, repo = repo_full_name.split("/", 1)
        cursor = None
        all_prs: list[PullRequest] = []
        page = 0
        page_size = min(50, limit) if limit else 50

        while True:
            page += 1
            _log(f"[cyan]Page {page}:[/cyan] Fetching up to {page_size} PRs (cursor: {cursor or 'start'})...")

            resp = self._session.post(self._gql_url, json={
                "query": PR_QUERY,
                "variables": {
                    "owner": owner,
                    "repo": repo,
                    "cursor": cursor,
                    "pageSize": page_size,
                },
            })

            if resp.status_code != 200:
                _log(f"[red]HTTP {resp.status_code}:[/red] {resp.text[:200]}")
                break

            data = resp.json()

            if "errors" in data:
                _log(f"[red]GraphQL errors:[/red] {data['errors']}")
                break

            rate = data.get("data", {}).get("rateLimit", {})
            _log(f"  Rate limit: {rate.get('remaining')}/{rate.get('limit')} "
                 f"(cost: {rate.get('cost')}, resets: {rate.get('resetAt', '?')})")

            pr_connection = data["data"]["repository"]["pullRequests"]
            total = pr_connection["totalCount"]
            nodes = pr_connection["nodes"]
            page_info = pr_connection["pageInfo"]

            if page == 1:
                _log(f"  Total closed/merged PRs in repo: [cyan]{total}[/cyan]")

            if not nodes:
                break

            # Parse PRs and check date bounds
            stopped_early = False
            for node in nodes:
                pr = self._parse_pr_node(node, repo_full_name)
                if not pr:
                    continue

                if since and pr.updated_at <= since:
                    _log(
                        f"  PR #{pr.number} updated {pr.updated_at.isoformat()} is at or before cutoff "
                        f"{since.isoformat()}, stopping."
                    )
                    stopped_early = True
                    break

                all_prs.append(pr)

            oldest_on_page = min((p.updated_at for p in all_prs[-len(nodes):]), default=None) if all_prs else None
            _log(f"  Got {len(nodes)} PRs this page, oldest updated: "
                 f"{oldest_on_page.date() if oldest_on_page else 'N/A'} "
                 f"[{len(all_prs)} total]")

            if stopped_early:
                break

            if limit and len(all_prs) >= limit:
                all_prs = all_prs[:limit]
                _log(f"  Reached limit of {limit} PRs.")
                break

            if not page_info["hasNextPage"]:
                break

            cursor = page_info["endCursor"]

        _log(f"Done. Fetched [green]{len(all_prs)}[/green] PRs with interactions.")
        return all_prs

    def _parse_pr_node(self, node: dict, repo_full_name: str) -> PullRequest | None:
        author = _parse_author(node.get("author"))
        if not author:
            return None

        interactions: list[Interaction] = []

        # Reviews (approvals, request changes, etc.)
        for review in (node.get("reviews") or {}).get("nodes", []):
            user = _parse_author(review.get("author"))
            if user:
                interactions.append(Interaction(
                    user=user,
                    type=InteractionType.REVIEW,
                    created_at=_parse_dt(review.get("submittedAt")) or datetime.now(timezone.utc),
                ))

        # Review thread comments (inline code comments)
        for thread in (node.get("reviewThreads") or {}).get("nodes", []):
            for comment in (thread.get("comments") or {}).get("nodes", []):
                user = _parse_author(comment.get("author"))
                if user:
                    interactions.append(Interaction(
                        user=user,
                        type=InteractionType.REVIEW_COMMENT,
                        created_at=_parse_dt(comment.get("createdAt")) or datetime.now(timezone.utc),
                    ))

        # Issue-style comments (PR conversation)
        for comment in (node.get("comments") or {}).get("nodes", []):
            user = _parse_author(comment.get("author"))
            if user:
                interactions.append(Interaction(
                    user=user,
                    type=InteractionType.ISSUE_COMMENT,
                    created_at=_parse_dt(comment.get("createdAt")) or datetime.now(timezone.utc),
                ))

        merged_at = _parse_dt(node.get("mergedAt"))
        state = PullRequestState.MERGED if merged_at else PullRequestState.CLOSED

        return PullRequest(
            number=node["number"],
            repo=repo_full_name,
            author=author,
            state=state,
            created_at=_parse_dt(node.get("createdAt")) or datetime.now(timezone.utc),
            updated_at=_parse_dt(node.get("updatedAt")) or datetime.now(timezone.utc),
            merged_at=merged_at,
            additions=node.get("additions", 0) or 0,
            deletions=node.get("deletions", 0) or 0,
            interactions=interactions,
        )
