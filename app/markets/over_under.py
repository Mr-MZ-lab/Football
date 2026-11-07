"""
Over/Under goal market predictions.

Uses the Poisson score probability matrix to compute the cumulative
probability of total goals crossing common thresholds (1.5, 2.5, 3.5, 4.5).

P(Over N.5) = sum of P(scoreline) where home_goals + away_goals > N.5
"""
import math
from typing import Dict, Any
import numpy as np
from scipy.stats import poisson

MAX_GOALS = 12  # consider scorelines up to 11-11 for accuracy


class OverUnderModel:

    THRESHOLDS = [0.5, 1.5, 2.5, 3.5, 4.5]

    def predict(self, lambda_home: float, lambda_away: float) -> Dict[str, Any]:
        """
        Args:
            lambda_home: expected goals for home team
            lambda_away: expected goals for away team

        Returns dict with over/under probabilities for each threshold.
        """
        lambda_home = max(0.1, lambda_home)
        lambda_away = max(0.1, lambda_away)

        # Build score matrix
        matrix = self._score_matrix(lambda_home, lambda_away)

        result: Dict[str, float] = {}
        for threshold in self.THRESHOLDS:
            key = f"{threshold:.1f}".replace(".", "_")
            p_over = float(
                sum(
                    matrix[h, a]
                    for h in range(MAX_GOALS)
                    for a in range(MAX_GOALS)
                    if h + a > threshold
                )
            )
            result[f"over_{key}"]  = round(min(0.99, p_over), 4)
            result[f"under_{key}"] = round(max(0.01, 1 - p_over), 4)

        result["expected_total_goals"] = round(lambda_home + lambda_away, 3)
        return result

    def _score_matrix(self, lam_h: float, lam_a: float) -> np.ndarray:
        matrix = np.zeros((MAX_GOALS, MAX_GOALS))
        for h in range(MAX_GOALS):
            for a in range(MAX_GOALS):
                matrix[h, a] = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix
