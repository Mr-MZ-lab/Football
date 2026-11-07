"""
Both Teams to Score (BTTS) market.

P(home scores) = 1 - P(home goals = 0) = 1 - e^(-lambda_home)
P(away scores) = 1 - P(away goals = 0) = 1 - e^(-lambda_away)

P(BTTS Yes) = P(home scores) × P(away scores)
P(BTTS No)  = 1 - P(BTTS Yes)

This is the exact analytical solution from independent Poisson processes.
"""
import math
from typing import Dict, Any


class BTTSModel:

    def predict(self, lambda_home: float, lambda_away: float) -> Dict[str, Any]:
        """
        Args:
            lambda_home: expected goals for home team
            lambda_away: expected goals for away team
        """
        lambda_home = max(0.05, lambda_home)
        lambda_away = max(0.05, lambda_away)

        p_home_scores = 1.0 - math.exp(-lambda_home)
        p_away_scores = 1.0 - math.exp(-lambda_away)
        p_btts_yes    = p_home_scores * p_away_scores
        p_btts_no     = 1.0 - p_btts_yes

        return {
            "btts_yes": round(p_btts_yes, 4),
            "btts_no":  round(p_btts_no, 4),
            "home_scores_probability": round(p_home_scores, 4),
            "away_scores_probability": round(p_away_scores, 4),
        }
