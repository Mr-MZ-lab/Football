"""
Markov Chain model for live match state transitions.

Models the match as a sequence of discrete states:
  state = (home_goals, away_goals, minute_bucket, home_men, away_men)

Transition probabilities are estimated from the current scoring rates
(adjusted for red cards and time remaining) rather than a pre-computed matrix,
making the model fully dynamic.
"""
import logging
from typing import Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Minutes remaining buckets used for piecewise goal-rate adjustment
# Goals tend to be scored at slightly higher rates late in matches.
LATE_GAME_MULTIPLIER = {
    (0, 15): 1.0,
    (15, 30): 1.0,
    (30, 45): 1.05,
    (45, 60): 0.95,  # slower after HT restart
    (60, 75): 1.05,
    (75, 90): 1.15,
    (90, 130): 1.25,
}


def _time_multiplier(minute: int) -> float:
    for (lo, hi), m in LATE_GAME_MULTIPLIER.items():
        if lo <= minute < hi:
            return m
    return 1.2


class MarkovChainModel:
    """
    Simulation-based Markov model.

    Runs N Monte Carlo simulations of the remaining match minutes and
    returns empirical win/draw/loss probabilities and expected remaining goals.
    """

    N_SIMS = 5000

    def predict(self, features: Dict[str, float], live_state: Dict[str, Any]) -> Dict[str, Any]:
        minute = live_state.get("current_minute", 0)
        home_goals = live_state.get("home_goals", 0)
        away_goals = live_state.get("away_goals", 0)
        home_red = live_state.get("home_red_cards", 0)
        away_red = live_state.get("away_red_cards", 0)

        remaining = max(1, 93 - minute)  # include ~3 min stoppage

        # Base scoring rates (per 90 min) from pre-match features
        rate_h = features.get("lambda_home", 1.4) / 90.0
        rate_a = features.get("lambda_away", 1.1) / 90.0

        # Adjust for red cards: each missing player reduces rate by ~15%
        rate_h *= max(0.4, 1.0 - 0.15 * home_red)
        rate_a *= max(0.4, 1.0 - 0.15 * away_red)

        # Apply time-of-game multiplier
        tm = _time_multiplier(minute)
        rate_h *= tm
        rate_a *= tm

        # Monte Carlo simulation
        rng = np.random.default_rng()
        final_h = home_goals + rng.poisson(rate_h * remaining, self.N_SIMS)
        final_a = away_goals + rng.poisson(rate_a * remaining, self.N_SIMS)

        hw = float(np.mean(final_h > final_a))
        dr = float(np.mean(final_h == final_a))
        aw = float(np.mean(final_h < final_a))

        exp_rem_h = float(np.mean(final_h - home_goals))
        exp_rem_a = float(np.mean(final_a - away_goals))

        return {
            "model": "markov_chain",
            "home_win": round(hw, 4),
            "draw": round(dr, 4),
            "away_win": round(aw, 4),
            "expected_remaining_home_goals": round(exp_rem_h, 3),
            "expected_remaining_away_goals": round(exp_rem_a, 3),
        }
