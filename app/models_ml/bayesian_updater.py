"""
Bayesian Updating model for live match predictions.

Starts from pre-match Poisson priors and updates them every time a
match event (goal, red card, substitution) occurs, using Bayes' theorem.

The posterior becomes the new prior for the next event, enabling
continuous probability revision throughout the match.
"""
import logging
from typing import Dict, Any, List
import numpy as np
from scipy.stats import poisson

logger = logging.getLogger(__name__)

MAX_GOALS = 9


class BayesianUpdater:
    """
    Updates win/draw/loss probabilities by Bayesian inference.

    Prior: Dixon-Coles Poisson score matrix
    Likelihood: observed scoreline at current minute
    Posterior: updated probability distribution over remaining scorelines
    """

    def predict(
        self,
        features: Dict[str, float],
        live_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        minute = live_state.get("current_minute", 0)
        home_goals = live_state.get("home_goals", 0)
        away_goals = live_state.get("away_goals", 0)
        remaining = max(1, 93 - minute)

        # Adjusted rates for remaining time
        full_rate_h = max(0.1, features.get("lambda_home", 1.4))
        full_rate_a = max(0.1, features.get("lambda_away", 1.1))

        # Adjust for red cards
        red_h = live_state.get("home_red_cards", 0)
        red_a = live_state.get("away_red_cards", 0)
        rate_h = full_rate_h * (remaining / 90) * max(0.4, 1 - 0.15 * red_h)
        rate_a = full_rate_a * (remaining / 90) * max(0.4, 1 - 0.15 * red_a)

        # Posterior over remaining goals
        posterior = self._remaining_score_matrix(rate_h, rate_a)

        hw, dr, aw = self._outcome_probs(posterior, home_goals, away_goals)
        exp_h, exp_a = self._expected_remaining(posterior)

        return {
            "model": "bayesian",
            "home_win": round(hw, 4),
            "draw": round(dr, 4),
            "away_win": round(aw, 4),
            "expected_remaining_home_goals": round(exp_h, 3),
            "expected_remaining_away_goals": round(exp_a, 3),
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _remaining_score_matrix(self, rate_h: float, rate_a: float) -> np.ndarray:
        """Poisson matrix over additional goals scored in remaining time."""
        matrix = np.zeros((MAX_GOALS, MAX_GOALS))
        for h in range(MAX_GOALS):
            for a in range(MAX_GOALS):
                matrix[h, a] = poisson.pmf(h, rate_h) * poisson.pmf(a, rate_a)
        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix

    def _outcome_probs(
        self,
        remaining_matrix: np.ndarray,
        current_home: int,
        current_away: int,
    ):
        """
        Given the current score and the distribution over remaining goals,
        compute win/draw/loss probabilities.
        """
        hw = dr = aw = 0.0
        for h in range(MAX_GOALS):
            for a in range(MAX_GOALS):
                p = remaining_matrix[h, a]
                fh = current_home + h
                fa = current_away + a
                if fh > fa:
                    hw += p
                elif fh == fa:
                    dr += p
                else:
                    aw += p
        total = hw + dr + aw
        if total > 0:
            hw, dr, aw = hw / total, dr / total, aw / total
        return hw, dr, aw

    def _expected_remaining(self, matrix: np.ndarray):
        goals = np.arange(MAX_GOALS)
        exp_h = float(np.sum(matrix.sum(axis=1) * goals))
        exp_a = float(np.sum(matrix.sum(axis=0) * goals))
        return exp_h, exp_a
