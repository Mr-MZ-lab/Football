"""
Dixon-Coles Poisson model for goal-count prediction.

Computes the probability of every scoreline (0-0 through 8-8) using
independent Poisson distributions for home and away goals, with a
low-score correction factor (Dixon & Coles, 1997).
"""
import math
import logging
from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import poisson

logger = logging.getLogger(__name__)

MAX_GOALS = 9  # Consider scorelines 0-0 … (MAX_GOALS-1)-(MAX_GOALS-1)


def _tau(home_goals: int, away_goals: int, lam_h: float, lam_a: float, rho: float) -> float:
    """
    Dixon-Coles correction for under/over-representation of low scores.
    Only applied when both teams score 0 or 1 goal.
    """
    if home_goals == 0 and away_goals == 0:
        return 1 - lam_h * lam_a * rho
    if home_goals == 1 and away_goals == 0:
        return 1 + lam_a * rho
    if home_goals == 0 and away_goals == 1:
        return 1 + lam_h * rho
    if home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


class PoissonModel:
    """
    Pre-match Poisson model.

    Inputs: lambda_home, lambda_away (expected goals from feature engineering)
    Outputs: score probability matrix → win/draw/loss probabilities + expected goals
    """

    RHO = 0.10  # Dixon-Coles low-score correction coefficient

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        lam_h = max(0.1, features.get("lambda_home", 1.4))
        lam_a = max(0.1, features.get("lambda_away", 1.1))

        score_matrix = self._score_matrix(lam_h, lam_a)
        home_win, draw, away_win = self._outcome_probs(score_matrix)
        exp_home, exp_away = self._expected_goals(score_matrix)

        return {
            "model": "poisson",
            "home_win": round(home_win, 4),
            "draw": round(draw, 4),
            "away_win": round(away_win, 4),
            "expected_home_goals": round(exp_home, 3),
            "expected_away_goals": round(exp_away, 3),
            "lambda_home": round(lam_h, 3),
            "lambda_away": round(lam_a, 3),
        }

    # ── internal ──────────────────────────────────────────────────────────────

    def _score_matrix(self, lam_h: float, lam_a: float) -> np.ndarray:
        """Returns MAX_GOALS × MAX_GOALS matrix of scoreline probabilities."""
        matrix = np.zeros((MAX_GOALS, MAX_GOALS))
        for h in range(MAX_GOALS):
            for a in range(MAX_GOALS):
                p = (
                    poisson.pmf(h, lam_h)
                    * poisson.pmf(a, lam_a)
                    * _tau(h, a, lam_h, lam_a, self.RHO)
                )
                matrix[h, a] = p

        # Renormalise due to truncation at MAX_GOALS
        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix

    def _outcome_probs(self, matrix: np.ndarray) -> Tuple[float, float, float]:
        home_win = float(np.tril(matrix, -1).sum())  # home > away → lower-left triangle
        away_win = float(np.triu(matrix, 1).sum())
        draw = float(np.diag(matrix).sum())
        return home_win, draw, away_win

    def _expected_goals(self, matrix: np.ndarray) -> Tuple[float, float]:
        goals = np.arange(MAX_GOALS)
        exp_h = float(np.sum(matrix.sum(axis=1) * goals))
        exp_a = float(np.sum(matrix.sum(axis=0) * goals))
        return exp_h, exp_a
