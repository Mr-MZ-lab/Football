"""
Hedging — lock in profit or cap loss with a second bet.

Two hedge modes:

1. Green-book (lock profit):
   You placed a pre-match bet and your selection is now winning.
   Bet on the opposite outcome(s) to guarantee a profit regardless of result.

2. Loss-limit (reduce exposure):
   You placed a bet and it's not going well.
   Hedge to limit your maximum loss to a defined amount.

Formulae:
  For a 2-way market (e.g. Over/Under):
    hedge_stake = (original_stake × original_odds) / hedge_odds

  For a 3-way market (e.g. 1X2), hedge across draw + away:
    hedge the "field" (all outcomes except your selection).
    Optimal: split proportional to 1/odds so returns are equalised.
"""
from __future__ import annotations

from typing import Dict, Optional


class HedgeCalculator:
    """
    Calculates hedge stakes to lock in guaranteed profit or cap losses.
    """

    # ── Public interface ──────────────────────────────────────────────────────

    def green_book(
        self,
        original_stake: float,
        original_odds: float,
        hedge_odds: Dict[str, float],
        commission: float = 0.0,
    ) -> Dict:
        """
        Calculate hedge stakes to guarantee a profit on any outcome.

        Parameters
        ----------
        original_stake : float
            Amount staked on the original selection (e.g. €100 on home win).
        original_odds : float
            Decimal odds at which the original bet was placed (e.g. 2.50).
        hedge_odds : dict
            Current odds for outcomes you want to hedge on.
            Example: {"draw": 3.20, "away": 4.50}
            For a 2-way market: {"away": 2.10}
        commission : float
            Exchange commission rate (0.0 for bookmakers, ~0.05 for Betfair).

        Returns
        -------
        dict with:
            guaranteed_profit — profit on any outcome after hedging
            hedge_stakes      — how much to bet on each hedge outcome
            hedge_returns     — gross return per outcome
            net_returns       — profit/loss per outcome (after all stakes)
            total_invested    — original + all hedge stakes
            roi               — return on investment %
        """
        original_return = original_stake * original_odds

        if not hedge_odds:
            return self._result(
                original_stake, original_odds, hedge_odds={},
                hedge_stakes={}, note="no hedge odds provided"
            )

        # Each hedge outcome must return the same gross amount as the original bet.
        # K = original_return; stake_i = K / odds_i ensures every scenario pays K.
        valid = {k: v for k, v in hedge_odds.items() if v > 1.0}
        if not valid:
            return self._result(original_stake, original_odds, hedge_odds,
                                hedge_stakes={}, note="invalid hedge odds")

        K = original_return

        hedge_stakes: Dict[str, float] = {}
        hedge_returns: Dict[str, float] = {}
        for outcome, odds in valid.items():
            stake = K / odds
            hedge_stakes[outcome] = round(stake, 2)
            hedge_returns[outcome] = round(stake * odds * (1.0 - commission), 2)

        total_hedge_cost = sum(hedge_stakes.values())
        total_invested = round(original_stake + total_hedge_cost, 2)

        guaranteed_gross = round(K * (1.0 - commission), 2)
        # Profit = what every outcome pays back, minus everything staked
        guaranteed_profit = round(guaranteed_gross - total_invested, 2)

        # Net P&L per scenario
        net_returns: Dict[str, float] = {}
        # Scenario: original bet wins
        net_returns["original_wins"] = round(
            original_return - original_stake - total_hedge_cost, 2
        )
        # Scenario: each hedge wins
        for outcome, ret in hedge_returns.items():
            net_returns[f"{outcome}_wins"] = round(
                ret - total_invested, 2
            )

        roi = round(guaranteed_profit / total_invested * 100, 2) if total_invested > 0 else 0.0

        return {
            "type":               "green_book",
            "guaranteed_profit":  guaranteed_profit,
            "guaranteed_gross":   guaranteed_gross,
            "total_invested":     total_invested,
            "original_stake":     original_stake,
            "original_odds":      original_odds,
            "original_return":    round(original_return, 2),
            "hedge_stakes":       hedge_stakes,
            "hedge_returns":      hedge_returns,
            "net_returns":        net_returns,
            "roi_pct":            roi,
            "commission":         commission,
        }

    def loss_limit(
        self,
        original_stake: float,
        original_odds: float,
        hedge_odds: Dict[str, float],
        max_loss: float,
        commission: float = 0.0,
    ) -> Dict:
        """
        Calculate minimum hedge stake to cap your loss at max_loss.

        If your original bet loses, your total loss ≤ max_loss.
        If it wins, you still profit (less than without hedging).

        Parameters
        ----------
        max_loss : float
            Maximum you're willing to lose in total (e.g. €30 from a €100 stake).
        """
        if max_loss >= original_stake:
            return {"note": "max_loss >= original_stake — no hedge needed",
                    "original_stake": original_stake,
                    "max_loss": max_loss}

        # We need: total_hedge_payout - total_invested ≥ -max_loss (when original loses)
        # i.e. hedge_return ≥ total_invested - max_loss
        # For simplest 2-way hedge (single hedge outcome):
        hedge_items = [(k, v) for k, v in hedge_odds.items() if v > 1.0]
        if not hedge_items:
            return {"note": "no valid hedge odds", "original_stake": original_stake}

        # Use the best (highest) hedge odds to minimise required stake
        hedge_outcome, hedge_odd = max(hedge_items, key=lambda x: x[1])

        # hedge_stake × hedge_odd × (1-comm) ≥ original_stake + hedge_stake - max_loss
        # hedge_stake × (hedge_odd × (1-comm) - 1) ≥ original_stake - max_loss
        net_odd = hedge_odd * (1.0 - commission) - 1.0
        if net_odd <= 0:
            return {"note": "hedge odds too low to cover loss limit"}

        min_hedge_stake = (original_stake - max_loss) / net_odd
        min_hedge_stake = round(min_hedge_stake, 2)

        total_invested = round(original_stake + min_hedge_stake, 2)
        hedge_return   = round(min_hedge_stake * hedge_odd * (1.0 - commission), 2)

        return {
            "type":                "loss_limit",
            "original_stake":      original_stake,
            "original_odds":       original_odds,
            "hedge_outcome":       hedge_outcome,
            "hedge_odds":          hedge_odd,
            "hedge_stake":         min_hedge_stake,
            "total_invested":      total_invested,
            "max_loss_guaranteed": round(total_invested - hedge_return, 2),
            "profit_if_original_wins": round(
                original_stake * original_odds - total_invested, 2
            ),
            "commission":          commission,
        }

    def breakeven(
        self,
        original_stake: float,
        original_odds: float,
        hedge_odds: float,
        commission: float = 0.0,
    ) -> Dict:
        """
        Find the hedge stake that results in exactly zero profit/loss.

        Useful when you just want to get your money back safely.
        """
        # breakeven: original_return == total_invested == original + hedge
        # original_stake × original_odds = original_stake + hedge_stake + hedge_cost
        # hedge_return = hedge_stake × hedge_odds × (1-comm) = original_return - original_stake
        # hedge_stake = (original_return - original_stake) / (hedge_odds × (1-comm))
        original_return = original_stake * original_odds
        net_profit_target = original_return - original_stake  # pre-hedge profit at risk
        hedge_stake = net_profit_target / (hedge_odds * (1.0 - commission))
        hedge_stake = round(hedge_stake, 2)

        return {
            "type":              "breakeven",
            "original_stake":    original_stake,
            "original_odds":     original_odds,
            "hedge_stake":       hedge_stake,
            "hedge_odds":        hedge_odds,
            "total_invested":    round(original_stake + hedge_stake, 2),
            "profit_either_way": 0.0,
            "commission":        commission,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _result(original_stake, original_odds, hedge_odds, hedge_stakes, **extra) -> Dict:
        return {
            "type":            "green_book",
            "original_stake":  original_stake,
            "original_odds":   original_odds,
            "hedge_odds":      hedge_odds,
            "hedge_stakes":    hedge_stakes,
            **extra,
        }
