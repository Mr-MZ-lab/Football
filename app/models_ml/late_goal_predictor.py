"""
Late Goal Predictor — Survival / Hazard Function Approach.

Models the probability of goals occurring in specific time windows:
  - After minute 75
  - In stoppage time (90+)

Uses a piece-wise constant hazard function fitted to the empirical distribution
of Premier League goal times. Goals are not uniformly distributed — there is a
well-documented spike in goals late in each half (44', 45+', 89', 90+').

Reference: Brillinger (2007) "A Potential Function Approach to the Flow of Play
in Soccer"; Dixon & Robinson (1998).
"""
import math
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Empirical goal-rate multipliers by time window (relative to mean rate)
# Source: Premier League 2010-2023 goal-time distribution analysis
HAZARD_SEGMENTS = [
    (0,  15, 0.90),
    (15, 30, 0.95),
    (30, 45, 1.05),
    (45, 60, 0.85),   # slow start second half
    (60, 75, 1.05),
    (75, 90, 1.20),   # late-game pressure
    (90, 130, 1.45),  # stoppage time — highest rate per minute
]


def _segment_rate(minute_start: int, minute_end: int, base_rate_per_min: float) -> float:
    """Compute expected goals in [minute_start, minute_end) using the hazard."""
    total = 0.0
    for seg_start, seg_end, multiplier in HAZARD_SEGMENTS:
        overlap_start = max(minute_start, seg_start)
        overlap_end = min(minute_end, seg_end)
        if overlap_end > overlap_start:
            total += base_rate_per_min * multiplier * (overlap_end - overlap_start)
    return total


def _p_at_least_one(rate: float) -> float:
    return 1.0 - math.exp(-max(0.0, rate))


class LateGoalPredictor:
    """
    Estimates late-goal probabilities for home, away, and either team.

    Uses:
      - Full-match expected goals from the ensemble (or Poisson)
      - Current score & minute (for live updates)
      - Hazard function to weight goals by time window
    """

    def predict(
        self,
        expected_home_goals: float,
        expected_away_goals: float,
        current_minute: int = 0,
        home_goals_so_far: int = 0,
        away_goals_so_far: int = 0,
    ) -> Dict[str, Any]:
        """
        Returns probabilities for goals in two late windows:
          - after_75: remaining match from minute max(current_minute, 75) to 90
          - after_90: stoppage time 90-95
        """
        # Remaining expected goals (full match) — already conditioned on current minute
        remaining_factor = max(0.0, (90 - current_minute) / 90.0)
        exp_h_remaining = max(0.0, expected_home_goals * remaining_factor)
        exp_a_remaining = max(0.0, expected_away_goals * remaining_factor)

        base_rate_h = exp_h_remaining / max(1, 90 - current_minute)
        base_rate_a = exp_a_remaining / max(1, 90 - current_minute)

        # Window 1: minute 75 → 90 (or current_minute if later)
        w1_start = max(current_minute, 75)
        w1_end = 90
        exp_h_75 = _segment_rate(w1_start, w1_end, base_rate_h)
        exp_a_75 = _segment_rate(w1_start, w1_end, base_rate_a)
        exp_both_75 = exp_h_75 + exp_a_75

        # Window 2: stoppage time 90 → 95
        w2_start = max(current_minute, 90)
        w2_end = 95
        exp_h_90 = _segment_rate(w2_start, w2_end, base_rate_h)
        exp_a_90 = _segment_rate(w2_start, w2_end, base_rate_a)
        exp_both_90 = exp_h_90 + exp_a_90

        return {
            "after_75": round(_p_at_least_one(exp_both_75), 4),
            "after_90": round(_p_at_least_one(exp_both_90), 4),
            "home_late_goal": round(_p_at_least_one(exp_h_75 + exp_h_90), 4),
            "away_late_goal": round(_p_at_least_one(exp_a_75 + exp_a_90), 4),
            "expected_goals_75_90": round(exp_both_75, 3),
            "expected_goals_90plus": round(exp_both_90, 3),
        }
