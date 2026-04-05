"""Tests for scoring primitives."""

import math
from datetime import datetime, timezone

from social_graph.models import Interaction, InteractionType, User
from social_graph.scoring import logarithmic_weight, score_pr_interactions


def test_logarithmic_weight_1():
    assert logarithmic_weight(1) == 1.0


def test_logarithmic_weight_10():
    assert math.isclose(logarithmic_weight(10), 2.0)


def test_logarithmic_weight_100():
    assert math.isclose(logarithmic_weight(100), 3.0)


def test_score_pr_interactions_empty():
    author = User(id="1", login="author")
    assert score_pr_interactions([], author) == {}


def test_score_pr_interactions_filters_bots():
    author = User(id="1", login="author")
    bot = User(id="2", login="bot", is_bot=True)
    interactions = [
        Interaction(user=bot, type=InteractionType.REVIEW, created_at=datetime.now(timezone.utc)),
    ]
    assert score_pr_interactions(interactions, author) == {}


def test_score_pr_interactions_filters_self():
    author = User(id="1", login="author")
    interactions = [
        Interaction(user=author, type=InteractionType.REVIEW, created_at=datetime.now(timezone.utc)),
    ]
    assert score_pr_interactions(interactions, author) == {}


def test_score_pr_interactions_single_reviewer():
    """Single reviewer with a review gets normalized to 1.0."""
    author = User(id="1", login="author")
    reviewer = User(id="2", login="reviewer")
    interactions = [
        Interaction(user=reviewer, type=InteractionType.REVIEW, created_at=datetime.now(timezone.utc)),
    ]
    scores = score_pr_interactions(interactions, author)
    assert math.isclose(scores[reviewer], 1.0)


def test_score_pr_interactions_golden_test():
    """Golden test for PR interaction scoring.

    reviewer1: 1 review (weight 1.0) + 1 issue comment (weight 0.4) = 2 interactions
    reviewer2: 1 review comment (weight 0.7) = 1 interaction

    reviewer1 raw = log_weight(2) * avg(1.0, 0.4) = 1.301 * 0.7 = 0.9107
    reviewer2 raw = log_weight(1) * avg(0.7) = 1.0 * 0.7 = 0.7
    reviewer1 normalized = 0.9107 / 0.9107 = 1.0
    reviewer2 normalized = 0.7 / 0.9107 ≈ 0.769
    """
    author = User(id="123", login="author")
    reviewer1 = User(id="456", login="reviewer1")
    reviewer2 = User(id="789", login="reviewer2")
    now = datetime.now(timezone.utc)

    interactions = [
        Interaction(user=reviewer1, type=InteractionType.REVIEW, created_at=now),
        Interaction(user=reviewer2, type=InteractionType.REVIEW_COMMENT, created_at=now),
        Interaction(user=reviewer1, type=InteractionType.ISSUE_COMMENT, created_at=now),
    ]

    scores = score_pr_interactions(interactions, author)

    assert math.isclose(scores[reviewer1], 1.0, rel_tol=1e-6)
    assert 0.6 < scores[reviewer2] < 0.8
    # More precisely:
    expected_r2 = 0.7 / ((1.0 + math.log10(2)) * 0.7)
    assert math.isclose(scores[reviewer2], expected_r2, rel_tol=1e-6)
