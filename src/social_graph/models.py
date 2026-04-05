"""Data models for the social graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PullRequestState(Enum):
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"


class InteractionType(Enum):
    REVIEW = "review"
    REVIEW_COMMENT = "review_comment"
    ISSUE_COMMENT = "issue_comment"


@dataclass(frozen=True)
class User:
    id: str
    login: str
    name: str | None = None
    is_bot: bool = False
    avatar_url: str | None = None
    html_url: str | None = None
    company: str | None = None
    location: str | None = None

    BOT_LOGINS = frozenset({
        "dependabot", "dependabot[bot]", "renovate", "renovate[bot]",
        "github-actions", "github-actions[bot]", "codecov", "codecov[bot]",
        "snyk-bot", "greenkeeper[bot]", "depfu[bot]", "cancelself",
    })

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, User):
            return NotImplemented
        return self.id == other.id

    @property
    def is_bot_or_service(self) -> bool:
        """Return True if this user is a bot or known service account."""
        return self.is_bot or self.login.lower() in self.BOT_LOGINS or "[bot]" in self.login.lower()

    @property
    def display_name(self) -> str:
        """Return the user's name if available, otherwise their login."""
        return self.name or self.login


@dataclass(frozen=True)
class Interaction:
    user: User
    type: InteractionType
    created_at: datetime


@dataclass
class PullRequest:
    number: int
    repo: str
    author: User
    state: PullRequestState
    created_at: datetime
    updated_at: datetime
    merged_at: datetime | None = None
    additions: int = 0
    deletions: int = 0
    interactions: list[Interaction] = field(default_factory=list)


@dataclass(frozen=True)
class SocialEdge:
    author: User
    reviewer: User
    weight: float


@dataclass(frozen=True)
class ExpertRanking:
    user: User
    score: float
    domain: str
