"""Gaussian time decay function.

The decay constant models the relevancy of recent discussions. Developers are
quite interested in discussions within this time frame but dramatically less
interested after it.

Visualize: https://www.wolframalpha.com/input?i=plot%5Be%5E%7B-t%5E2%2F112%5E2%7D%2C+%7Bt%2C0%2C336%7D%5D
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

# 16 weeks — exponential Gaussian decay constant
DECAY_CONSTANT_DAYS: int = 7 * 16  # 112

# After 3 sigma the decay is reduced to ~0.01% — not worth measuring
THREE_SIGMA_DAYS: int = DECAY_CONSTANT_DAYS * 3  # 336


def exponential_time_decay(delta_days: int) -> float:
    """Gaussian decay: exp(-(t^2) / (C^2)) where C = 112 days.

    Uses integer day truncation for consistent whole-day granularity.
    """
    return math.exp(-(delta_days**2) / (DECAY_CONSTANT_DAYS**2))


def days_since(dt: datetime, now: datetime | None = None) -> int:
    """Whole-day difference between *now* and *dt*."""
    if now is None:
        now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - dt).days
