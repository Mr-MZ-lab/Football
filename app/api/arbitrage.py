"""
POST /api/v1/arbitrage — detect cross-bookmaker arbitrage opportunities.
POST /api/v1/hedge     — calculate hedge stakes to lock profit or cap loss.
"""
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.markets.arbitrage import ArbitrageDetector
from app.markets.hedging import HedgeCalculator

logger = logging.getLogger(__name__)
router = APIRouter()

_arb    = ArbitrageDetector()
_hedger = HedgeCalculator()


# ── Arbitrage ─────────────────────────────────────────────────────────────────

class ArbitrageRequest(BaseModel):
    """
    Pass the best available odds per outcome from each bookmaker you have access to.
    The engine picks the best odds for each outcome and checks for an arb.
    """
    bookmaker_odds: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Odds matrix: {bookmaker: {outcome: decimal_odds}}",
        json_schema_extra={"example": {
            "tipico": {"home": 2.10, "draw": 3.60, "away": 3.80},
            "bwin":   {"home": 2.20, "draw": 3.50, "away": 4.10},
            "betano": {"home": 2.05, "draw": 3.90, "away": 3.70},
        }},
    )
    total_stake: float = Field(
        1000.0, gt=0,
        description="Total capital to allocate if an arb is found (€)",
    )


@router.post("/arbitrage", summary="Detect cross-bookmaker arbitrage opportunities")
def detect_arbitrage(req: ArbitrageRequest) -> Dict:
    """
    Scans odds across multiple bookmakers for guaranteed-profit opportunities.

    An arbitrage exists when the sum of implied probabilities (1/odds) for all
    outcomes is below 1.0. Returns optimal stakes per outcome to guarantee profit
    regardless of match result.

    Typical margins: 0.5%–5% depending on market efficiency.
    Note: bookmakers limit/ban accounts that consistently exploit arbitrage.
    """
    try:
        results = _arb.scan_markets(req.bookmaker_odds, req.total_stake)
        result = results[0] if results else {}

        # Also expose best-odds summary
        best_odds: Dict[str, float] = {}
        best_book: Dict[str, str] = {}
        for bookmaker, odds in req.bookmaker_odds.items():
            for outcome, odd in odds.items():
                if odd > best_odds.get(outcome, 0.0):
                    best_odds[outcome] = odd
                    best_book[outcome] = bookmaker

        return {
            **result,
            "best_odds_per_outcome": best_odds,
            "best_book_per_outcome": best_book,
            "bookmakers_compared":   list(req.bookmaker_odds.keys()),
            "advice": (
                "Arb found — act fast, odds change within seconds."
                if result.get("is_arb")
                else "No arb currently. Overround = "
                     f"{result.get('implied_sum_pct', 0):.2f}% "
                     "(bookmaker margin keeps you from profiting)."
            ),
        }
    except Exception as e:
        logger.exception("Arbitrage detection failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Hedging ───────────────────────────────────────────────────────────────────

class HedgeRequest(BaseModel):
    original_stake: float = Field(..., gt=0, description="Amount staked on original bet (€)")
    original_odds:  float = Field(..., gt=1.0, description="Decimal odds of original bet")
    hedge_odds:     Dict[str, float] = Field(
        ...,
        description="Current odds for outcomes you want to hedge on",
        json_schema_extra={"example": {"draw": 3.20, "away": 4.50}},
    )
    mode: str = Field(
        "green_book",
        description="Hedge mode: 'green_book' (lock profit) | 'loss_limit' | 'breakeven'",
    )
    max_loss: Optional[float] = Field(
        None, description="Max loss in € (only for loss_limit mode)"
    )
    commission: float = Field(
        0.0, ge=0.0, lt=1.0,
        description="Exchange commission rate (0.0 for bookmakers, 0.05 for Betfair)",
    )


@router.post("/hedge", summary="Calculate hedge stakes to lock profit or cap loss")
def calculate_hedge(req: HedgeRequest) -> Dict:
    """
    Calculates a second bet to either:

    - **green_book**: guarantee the same profit regardless of outcome (lock in winnings)
    - **loss_limit**: cap your maximum loss at a defined amount
    - **breakeven**: get your money back regardless of outcome (zero profit/loss)

    Practical example:
    You bet €100 on Bayern to win at 2.50 pre-match. At half-time Bayern lead 2-0
    and the draw/away odds have shortened. Use green_book to lock in profit now.
    """
    try:
        mode = req.mode.lower().replace("-", "_")

        if mode == "green_book":
            result = _hedger.green_book(
                req.original_stake, req.original_odds,
                req.hedge_odds, req.commission,
            )
        elif mode == "loss_limit":
            if req.max_loss is None:
                raise HTTPException(
                    status_code=422,
                    detail="max_loss is required for loss_limit mode",
                )
            result = _hedger.loss_limit(
                req.original_stake, req.original_odds,
                req.hedge_odds, req.max_loss, req.commission,
            )
        elif mode == "breakeven":
            if len(req.hedge_odds) != 1:
                raise HTTPException(
                    status_code=422,
                    detail="breakeven mode requires exactly one hedge outcome",
                )
            hedge_outcome, hedge_odd = next(iter(req.hedge_odds.items()))
            result = _hedger.breakeven(
                req.original_stake, req.original_odds,
                hedge_odd, req.commission,
            )
            result["hedge_outcome"] = hedge_outcome
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown mode '{req.mode}'. Use: green_book, loss_limit, breakeven",
            )

        # Human-readable summary
        gp = result.get("guaranteed_profit", 0)
        ti = result.get("total_invested", req.original_stake)
        result["summary"] = (
            f"Hedge with these stakes to guarantee "
            f"€{gp:.2f} profit on a total investment of €{ti:.2f}."
            if gp > 0
            else f"Hedge to minimise loss (total invested €{ti:.2f})."
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Hedge calculation failed")
        raise HTTPException(status_code=500, detail=str(e))
