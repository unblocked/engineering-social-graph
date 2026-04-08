"""Build social graph edges from commit co-authorship and file proximity."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from social_graph.decay import THREE_SIGMA_DAYS, days_since, exponential_time_decay
from social_graph.models import Commit, SocialEdge, User
from social_graph.scoring import MINIMUM_SOCIAL_WEIGHT, InteractionType, INTERACTION_WEIGHTS


def build_commit_graph(
    commits: list[Commit],
    proximity_window_days: int = 7,
    now: datetime | None = None,
) -> list[SocialEdge]:
    """Build weighted social edges from commit history.

    Two signals:
    1. Co-authorship: explicit Co-authored-by trailers → strong collaboration signal
    2. File proximity: two authors editing the same file within a time window → weaker signal

    Returns edges in the same SocialEdge format as the PR-based builder.
    In commit mode, 'reviewer' means 'collaborator'.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    since = now - timedelta(days=THREE_SIGMA_DAYS)

    # Filter to lookback window
    recent = [c for c in commits if c.committed_at >= since]
    if not recent:
        return []

    w_coauthor = INTERACTION_WEIGHTS[InteractionType.CO_AUTHOR]
    w_proximity = INTERACTION_WEIGHTS[InteractionType.FILE_PROXIMITY]

    # Accumulate raw scores: author_login -> {collaborator_login -> score}
    # We also track User objects for final edge construction
    raw: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    user_map: dict[str, User] = {}

    # --- Signal 1: Co-authorship ---
    for c in recent:
        if not c.coauthors:
            continue
        decay = exponential_time_decay(days_since(c.committed_at, now))
        user_map[c.author.login] = c.author
        for ca in c.coauthors:
            if ca.login == c.author.login:
                continue
            if ca.is_bot_or_service or c.author.is_bot_or_service:
                continue
            user_map[ca.login] = ca
            # Bidirectional: both author and co-author get edges to each other
            raw[c.author.login][ca.login] += w_coauthor * decay
            raw[ca.login][c.author.login] += w_coauthor * decay

    # --- Signal 2: File proximity ---
    # Group commits by file path, then find authors who touched the same file
    # within the proximity window
    file_commits: dict[str, list[Commit]] = defaultdict(list)
    for c in recent:
        if c.author.is_bot_or_service:
            continue
        user_map[c.author.login] = c.author
        for f in c.files:
            file_commits[f].append(c)

    for filepath, fc_list in file_commits.items():
        if len(fc_list) < 2:
            continue
        # Sort by date for sliding window
        fc_list.sort(key=lambda c: c.committed_at)

        # Sliding window: for each commit, look ahead within the proximity window
        # Cap interactions per author-pair per file to avoid noise from hot files
        pair_count: dict[tuple[str, str], int] = defaultdict(int)
        max_per_pair = 5

        for i, c1 in enumerate(fc_list):
            cutoff = c1.committed_at + timedelta(days=proximity_window_days)
            for j in range(i + 1, len(fc_list)):
                c2 = fc_list[j]
                if c2.committed_at > cutoff:
                    break
                a1, a2 = c1.author.login, c2.author.login
                if a1 == a2:
                    continue
                pair_key = (min(a1, a2), max(a1, a2))
                if pair_count[pair_key] >= max_per_pair:
                    continue
                pair_count[pair_key] += 1

                # Decay based on the earlier commit's age
                mid_delta = days_since(c1.committed_at, now)
                decay = exponential_time_decay(mid_delta)
                raw[a1][a2] += w_proximity * decay
                raw[a2][a1] += w_proximity * decay

    # --- Normalize per author and filter ---
    edges: list[SocialEdge] = []
    for author_login, collaborators in raw.items():
        if not collaborators:
            continue
        max_total = max(collaborators.values())
        if max_total == 0:
            continue
        author = user_map[author_login]
        for collab_login, total in collaborators.items():
            weight = total / max_total
            if weight >= MINIMUM_SOCIAL_WEIGHT:
                collab = user_map[collab_login]
                edges.append(SocialEdge(author=author, reviewer=collab, weight=weight))

    return edges
