"""
Player scoring probability model.

Estimates the probability each player scores at least one goal in the match,
using xG-based Poisson arrival rates adjusted for:
  - Expected minutes played (position, lineup status)
  - Penalty taking responsibility
  - Injury / suspension status
"""
import math
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Expected minutes per 90 by position (starters vs subs)
POSITION_MINUTES = {
    "FWD": 82,
    "MID": 78,
    "DEF": 75,
    "GK": 90,
}


def _poisson_at_least_one(rate: float) -> float:
    """P(X >= 1) = 1 - e^(-rate)  where X ~ Poisson(rate)."""
    return 1.0 - math.exp(-max(0.0, rate))


class PlayerScoringModel:
    """
    For each available player, computes:
      - expected_xg: xG contribution to this match
      - scoring_probability: P(player scores >= 1 goal)
    """

    def predict(
        self,
        players: List[Dict[str, Any]],
        team_expected_goals: float,
        is_home: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            players: list of player stat dicts from MockDataProvider
            team_expected_goals: team-level xG from Poisson/ensemble model
            is_home: whether this is the home team (minor effect on xG scale)

        Returns:
            Sorted list of player dicts with scoring_probability added.
        """
        available = [p for p in players if not p.get("is_injured") and not p.get("is_suspended")]
        if not available:
            return []

        home_factor = 1.05 if is_home else 0.97

        # Raw xg rates per 90 for available players
        raw_rates = []
        for p in available:
            xg90 = p.get("xg_per_90", 0.05)
            minutes = POSITION_MINUTES.get(p.get("position", "MID"), 78)
            match_xg = xg90 * (minutes / 90.0) * home_factor
            raw_rates.append(match_xg)

        # Scale so team total xG matches model output
        total_raw = sum(raw_rates) or 1.0
        scale = team_expected_goals / total_raw

        results = []
        for player, raw_xg in zip(available, raw_rates):
            scaled_xg = raw_xg * scale

            # Add penalty contribution
            pen_taken = player.get("penalties_taken", 0)
            pen_scored = player.get("penalties_scored", 0)
            pen_rate = (pen_scored / pen_taken * 0.1) if pen_taken > 0 else 0.0  # ~0.1 pens per match if taker
            is_penalty_taker = pen_taken > 0

            total_xg = scaled_xg + pen_rate
            prob = _poisson_at_least_one(total_xg)

            results.append({
                "player_id": player.get("id", 0),
                "player_name": player["name"],
                "team": player["team"],
                "position": player.get("position", "MID"),
                "scoring_probability": round(min(0.99, prob), 4),
                "xg_contribution": round(total_xg, 4),
                "is_penalty_taker": is_penalty_taker,
            })

        # Sort by scoring probability descending
        results.sort(key=lambda x: x["scoring_probability"], reverse=True)
        return results
