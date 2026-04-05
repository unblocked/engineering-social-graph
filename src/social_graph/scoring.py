"""Scoring primitives — interaction weights, log dampening, normalization."""

from __future__ import annotations

import math
from collections import defaultdict

from social_graph.models import Interaction, InteractionType, User

# Default interaction type weights
INTERACTION_WEIGHTS: dict[InteractionType, float] = {
    InteractionType.REVIEW: 1.0,
    InteractionType.REVIEW_COMMENT: 0.7,
    InteractionType.ISSUE_COMMENT: 0.4,
}

# Edges below this threshold are discarded
MINIMUM_SOCIAL_WEIGHT: float = 0.1


def logarithmic_weight(count: int) -> float:
    """Diminish large counts logarithmically.

    - logarithmic_weight(1)   == 1.0
    - logarithmic_weight(10)  == 2.0
    - logarithmic_weight(100) == 3.0

    Provides diminishing returns for high interaction counts.
    """
    return 1.0 + math.log10(count)


def score_pr_interactions(
    interactions: list[Interaction],
    author: User,
) -> dict[User, float]:
    """Score reviewers for a single PR's interactions.

    Groups by reviewer, applies log dampening * average interaction weight,
    then normalizes so the top reviewer in this PR scores 1.0.

    Returns empty dict if no valid interactions.
    """
    # Filter out bots, service accounts, and self-interactions
    valid = [i for i in interactions if not i.user.is_bot_or_service and i.user != author]
    if not valid:
        return {}

    # Group by reviewer
    by_reviewer: dict[User, list[float]] = defaultdict(list)
    for interaction in valid:
        weight = INTERACTION_WEIGHTS.get(interaction.type, 0.0)
        by_reviewer[interaction.user].append(weight)

    # Per reviewer: log_weight(count) * avg(interaction_weights)
    raw_scores: dict[User, float] = {}
    for reviewer, weights in by_reviewer.items():
        avg_weight = sum(weights) / len(weights)
        raw_scores[reviewer] = logarithmic_weight(len(weights)) * avg_weight

    # Normalize within this PR
    max_raw = max(raw_scores.values()) if raw_scores else 1.0
    if max_raw == 0:
        return {}
    return {reviewer: score / max_raw for reviewer, score in raw_scores.items()}
