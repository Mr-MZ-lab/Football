"""
Arbitrage Betting — guaranteed profit across multiple bookmakers.

An arbitrage opportunity exists when the sum of implied probabilities
across all outcomes (1/odds_home + 1/odds_draw + 1/odds_away) is < 1.0.
The gap represents the guaranteed profit margin.

Example:
  Site A: home 2.20 → implied 45.5%
  Site B: draw 3.80 → implied 26.3%
  Site C: away 4.10 → implied 24.4%
  Sum = 96.2%  →  3.8% profit margin (arb exists!)

Stake formula for guaranteed return R on total stake S:
  stake_i = S × (1 / odds_i) / sum(1 / odds_j for all j)
"""
from __future__ import annotations

from typing import Dict, List, Optional


OUTCOMES = ("home", "draw", "away")


class ArbitrageDetector:
    """
    Detects arbitrage opportunities and computes optimal stakes.

    Accepts odds per outcome, each from the best available bookmaker.
    Works for 1X2 (3-way) and 2-way markets (Over/Under, BTTS).
    """

    def detect(
        self,
        odds: Dict[str, float],
        total_stake: float = 1000.0,
    ) -> Dict:
        """
        Check for an arbitrage opportunity and calculate stakes.

        Parameters
        ----------
        odds : dict
            Decimal odds per outcome keyed by outcome name.
            Example: {"home": 2.20, "draw": 3.80, "away": 4.10}
            Each value should come from the *best* bookmaker for that outcome.
        total_stake : float
            Total capital to allocate across all bets (default €1 000).

        Returns
        -------
        dict with:
            is_arb          — True if an arb exists
            margin          — profit margin (e.g. 0.038 = 3.8%)
            guaranteed_profit — absolute profit on total_stake
            stakes          — per-outcome stake amounts
            returns         — per-outcome gross return (always equal if calculated correctly)
            implied_sum     — sum of implied probabilities (< 1 → arb)
        """
        valid = {k: v for k, v in odds.items() if v and v > 1.0}
        if len(valid) < 2:
            return self._no_arb(odds, total_stake, reason="insufficient_odds")

        implied = {k: 1.0 / v for k, v in valid.items()}
        implied_sum = sum(implied.values())

        is_arb = implied_sum < 1.0
        margin = round(1.0 - implied_sum, 6) if is_arb else 0.0

        stakes = {}
        returns = {}
        if is_arb:
            for outcome, imp in implied.items():
                stakes[outcome] = round(total_stake * (imp / implied_sum), 2)
                returns[outcome] = round(stakes[outcome] * valid[outcome], 2)
        else:
            for outcome in valid:
                stakes[outcome] = 0.0
                returns[outcome] = 0.0

        guaranteed_profit = round(margin * total_stake, 2) if is_arb else 0.0

        return {
            "is_arb":             is_arb,
            "margin":             round(margin, 6),
            "margin_pct":         round(margin * 100, 4),
            "guaranteed_profit":  guaranteed_profit,
            "total_stake":        total_stake,
            "implied_sum":        round(implied_sum, 6),
            "implied_sum_pct":    round(implied_sum * 100, 4),
            "stakes":             stakes,
            "returns":            returns,
            "odds_used":          valid,
        }

    def scan_markets(
        self,
        bookmaker_odds_matrix: Dict[str, Dict[str, float]],
        total_stake: float = 1000.0,
    ) -> List[Dict]:
        """
        Scan multiple markets for arb opportunities.

        Parameters
        ----------
        bookmaker_odds_matrix : dict
            {bookmaker_name: {outcome: decimal_odds}}
            Example:
              {
                "tipico":  {"home": 2.10, "draw": 3.60, "away": 3.80},
                "bwin":    {"home": 2.20, "draw": 3.50, "away": 4.10},
                "betano":  {"home": 2.05, "draw": 3.90, "away": 3.70},
              }
        total_stake : float
            Capital to allocate if an arb is found.
        """
        if not bookmaker_odds_matrix:
            return []

        all_outcomes = set()
        for bm_odds in bookmaker_odds_matrix.values():
            all_outcomes.update(bm_odds.keys())

        best_odds: Dict[str, float] = {}
        best_book: Dict[str, str] = {}
        for outcome in all_outcomes:
            for bookmaker, bm_odds in bookmaker_odds_matrix.items():
                o = bm_odds.get(outcome, 0.0)
                if o > best_odds.get(outcome, 0.0):
                    best_odds[outcome] = o
                    best_book[outcome] = bookmaker

        result = self.detect(best_odds, total_stake)
        result["best_odds_source"] = best_book
        result["bookmakers_checked"] = list(bookmaker_odds_matrix.keys())
        return [result]

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _no_arb(odds: Dict, total_stake: float, reason: str = "") -> Dict:
        implied_sum = sum(1.0 / v for v in odds.values() if v and v > 1.0)
        return {
            "is_arb":             False,
            "margin":             0.0,
            "margin_pct":         0.0,
            "guaranteed_profit":  0.0,
            "total_stake":        total_stake,
            "implied_sum":        round(implied_sum, 6),
            "implied_sum_pct":    round(implied_sum * 100, 4),
            "stakes":             {},
            "returns":            {},
            "odds_used":          odds,
            "note":               reason,
        }
