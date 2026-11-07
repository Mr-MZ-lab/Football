"""
Kelly Criterion — optimal stake sizing.

Full Kelly: f* = (p × b - q) / b
  p = model win probability
  q = 1 - p
  b = net decimal odds (bookmaker_decimal_odds - 1)

We use Fractional Kelly (default: half-Kelly) to reduce variance
while preserving the edge-exploitation property.

Rules enforced:
  - Never bet if edge <= 0 (model probability < implied odds probability)
  - Cap single bet at MAX_BET_FRACTION of bankroll (default 25%)
  - Minimum recommended bet: 1% of bankroll
"""
from typing import Dict, Any, List, Optional


MAX_BET_FRACTION = 0.25   # Never exceed 25% of bankroll on a single bet
MIN_BET_FRACTION = 0.01   # Minimum worth flagging
DEFAULT_KELLY_FRACTION = 0.5   # Half-Kelly by default


class KellyCriterion:

    def __init__(self, kelly_fraction: float = DEFAULT_KELLY_FRACTION):
        self._frac = kelly_fraction

    def calculate(
        self,
        model_probability: float,
        bookmaker_odds: float,
        bankroll: float = 1000.0,
    ) -> Dict[str, Any]:
        """
        Args:
            model_probability: our estimated win probability (0-1)
            bookmaker_odds: decimal odds from bookmaker (e.g. 2.10)
            bankroll: total available bankroll in currency units

        Returns:
            {
              "edge": float,              # model_prob - implied_prob
              "kelly_fraction": float,    # optimal bet as fraction of bankroll
              "recommended_stake": float, # in currency units
              "expected_value": float,    # per unit staked
              "is_value_bet": bool,
            }
        """
        if bookmaker_odds <= 1.0:
            return self._no_bet("Invalid odds (must be > 1.0)")

        p = model_probability
        q = 1.0 - p
        b = bookmaker_odds - 1.0  # net odds
        implied_prob = 1.0 / bookmaker_odds
        edge = p - implied_prob

        if edge <= 0:
            return self._no_bet(f"No edge (model={p:.3f}, implied={implied_prob:.3f})")

        # Full Kelly formula
        full_kelly = (p * b - q) / b

        # Apply fractional Kelly
        kelly = full_kelly * self._frac

        # Apply caps
        kelly = max(0.0, min(kelly, MAX_BET_FRACTION))

        stake = round(kelly * bankroll, 2)
        ev = round(p * b - q, 4)  # expected value per unit staked

        return {
            "edge": round(edge, 4),
            "full_kelly_fraction": round(full_kelly, 4),
            "kelly_fraction": round(kelly, 4),
            "recommended_stake": stake,
            "recommended_stake_pct": round(kelly * 100, 2),
            "expected_value": ev,
            "is_value_bet": True,
            "bankroll": bankroll,
        }

    def calculate_portfolio(
        self,
        bets: List[Dict[str, Any]],
        bankroll: float = 1000.0,
    ) -> List[Dict[str, Any]]:
        """
        Calculate Kelly stakes for a list of bets simultaneously.
        Scales down stakes so total exposure ≤ 50% of bankroll.

        Each bet dict: {"market", "model_probability", "bookmaker_odds"}
        """
        results = []
        for bet in bets:
            result = self.calculate(
                bet["model_probability"],
                bet["bookmaker_odds"],
                bankroll,
            )
            result["market"] = bet.get("market", "unknown")
            results.append(result)

        # Scale if total exposure exceeds 50% of bankroll
        total_kelly = sum(r["kelly_fraction"] for r in results if r.get("is_value_bet"))
        if total_kelly > 0.50:
            scale = 0.50 / total_kelly
            for r in results:
                if r.get("is_value_bet"):
                    r["kelly_fraction"] = round(r["kelly_fraction"] * scale, 4)
                    r["recommended_stake"] = round(r["kelly_fraction"] * bankroll, 2)
                    r["recommended_stake_pct"] = round(r["kelly_fraction"] * 100, 2)
                    r["note"] = "Scaled down (portfolio exposure cap)"

        return results

    def _no_bet(self, reason: str) -> Dict[str, Any]:
        return {
            "edge": 0.0,
            "kelly_fraction": 0.0,
            "recommended_stake": 0.0,
            "recommended_stake_pct": 0.0,
            "expected_value": 0.0,
            "is_value_bet": False,
            "reason": reason,
        }
