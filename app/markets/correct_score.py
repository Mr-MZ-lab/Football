"""
Correct Score market.

Returns the top N most likely exact scorelines and their probabilities.
Useful for comparing against bookmaker correct-score odds.

Uses Dixon-Coles Poisson with low-score correction (same as main model).
"""
from typing import Dict, Any, List
import numpy as np
from scipy.stats import poisson

MAX_GOALS = 8   # consider 0-0 through 7-7


def _tau(h: int, a: int, lam_h: float, lam_a: float, rho: float = 0.10) -> float:
    """Dixon-Coles correction for low-score bias."""
    if h == 0 and a == 0:
        return 1 - lam_h * lam_a * rho
    if h == 1 and a == 0:
        return 1 + lam_a * rho
    if h == 0 and a == 1:
        return 1 + lam_h * rho
    if h == 1 and a == 1:
        return 1 - rho
    return 1.0


class CorrectScoreModel:

    def predict(
        self,
        lambda_home: float,
        lambda_away: float,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Returns the top_n most likely exact scorelines.

        Each entry:
          { "score": "2-1", "home_goals": 2, "away_goals": 1, "probability": 0.121 }
        """
        lambda_home = max(0.1, lambda_home)
        lambda_away = max(0.1, lambda_away)

        scores = []
        for h in range(MAX_GOALS):
            for a in range(MAX_GOALS):
                p = (
                    poisson.pmf(h, lambda_home)
                    * poisson.pmf(a, lambda_away)
                    * _tau(h, a, lambda_home, lambda_away)
                )
                scores.append({
                    "score": f"{h}-{a}",
                    "home_goals": h,
                    "away_goals": a,
                    "probability": round(float(p), 5),
                })

        # Normalise
        total = sum(s["probability"] for s in scores)
        if total > 0:
            for s in scores:
                s["probability"] = round(s["probability"] / total, 5)

        scores.sort(key=lambda s: s["probability"], reverse=True)
        return scores[:top_n]
