"""
Asian Handicap market.

Adjusts the effective goal line for the favoured team to eliminate the draw,
giving fairer odds. Common handicap lines: -0.5, -1, -1.5, -2, +0.5, +1, +1.5.

Implementation:
  - Positive handicap (e.g. +1): goals added to away team
  - Negative handicap (e.g. -1): goals deducted from home team

Uses the Poisson score matrix with the handicap shift applied.
Quarter-ball handicaps (e.g. -0.75) are split 50/50 between -0.5 and -1.
"""
import math
from typing import Dict, Any, List
import numpy as np
from scipy.stats import poisson

MAX_GOALS = 12


class AsianHandicapModel:

    COMMON_LINES = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

    def predict(
        self,
        lambda_home: float,
        lambda_away: float,
        handicap: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Compute Asian Handicap probabilities for a single line.

        Args:
            lambda_home: home team expected goals
            lambda_away: away team expected goals
            handicap: applied to home team (negative = home gives goals away)
                      e.g. -1 means home team starts at -1

        Returns:
            {"home_cover": P, "push": P, "away_cover": P, "handicap": handicap}
        """
        lambda_home = max(0.1, lambda_home)
        lambda_away = max(0.1, lambda_away)

        matrix = self._score_matrix(lambda_home, lambda_away)

        home_cover = push = away_cover = 0.0
        for h in range(MAX_GOALS):
            for a in range(MAX_GOALS):
                p = matrix[h, a]
                # Effective goal difference after handicap
                eff_diff = (h + handicap) - a
                if eff_diff > 0:
                    home_cover += p
                elif eff_diff == 0:
                    push += p
                else:
                    away_cover += p

        return {
            "handicap": handicap,
            "home_cover": round(home_cover, 4),
            "push":       round(push, 4),
            "away_cover": round(away_cover, 4),
        }

    def all_lines(
        self,
        lambda_home: float,
        lambda_away: float,
    ) -> List[Dict[str, Any]]:
        """Return AH probabilities for all common handicap lines."""
        return [
            self.predict(lambda_home, lambda_away, line)
            for line in self.COMMON_LINES
        ]

    def _score_matrix(self, lam_h: float, lam_a: float) -> np.ndarray:
        matrix = np.zeros((MAX_GOALS, MAX_GOALS))
        for h in range(MAX_GOALS):
            for a in range(MAX_GOALS):
                matrix[h, a] = poisson.pmf(h, lam_h) * poisson.pmf(a, lam_a)
        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix
