"""
POST /markets — all betting markets for a given fixture.

Returns the complete picture for a bettor:
  - 1X2 (win/draw/loss)
  - Over/Under (0.5 through 4.5)
  - BTTS (Both Teams to Score)
  - Double Chance (1X / X2 / 12)
  - Correct Score (top 10)
  - Asian Handicap (common lines)
  - Value bets vs bookmaker odds
  - Kelly Criterion stake recommendations
"""
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ingestion.data_loader import get_team_features
from app.processing.feature_engineer import FeatureEngineer
from app.models_ml.poisson_model import PoissonModel
from app.models_ml.logistic_model import LogisticModel
from app.models_ml.xgboost_model import XGBoostModel
from app.models_ml.late_goal_predictor import LateGoalPredictor
from app.ensemble.ensemble import EnsembleEngine
from app.markets import (
    OverUnderModel, BTTSModel, DoubleChanceModel,
    CorrectScoreModel, AsianHandicapModel, KellyCriterion,
)
from app.markets.calibration import BetaCalibrator

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Singletons ────────────────────────────────────────────────────────────────
_fe        = FeatureEngineer()
_poisson   = PoissonModel()
_logistic  = LogisticModel()
_xgboost   = XGBoostModel()
_late_goal = LateGoalPredictor()
_ensemble  = EnsembleEngine()
_ou        = OverUnderModel()
_btts      = BTTSModel()
_dc        = DoubleChanceModel()
_cs        = CorrectScoreModel()
_ah        = AsianHandicapModel()
_calibrator= BetaCalibrator()


# ── Request schema ────────────────────────────────────────────────────────────

class MarketsRequest(BaseModel):
    home_team: str = Field(..., json_schema_extra={"example": "Bayern München"})
    away_team: str = Field(..., json_schema_extra={"example": "Borussia Dortmund"})
    competition: Optional[str] = Field(None, json_schema_extra={"example": "Bundesliga"})
    bankroll: float = Field(1000.0, gt=0, description="Bankroll for Kelly calculation")
    kelly_fraction: float = Field(0.5, gt=0, le=1.0, description="Kelly fraction (0.5 = half-Kelly)")
    bookmaker_odds: Optional[Dict[str, float]] = Field(
        None,
        description="Decimal odds per market. Keys: home, draw, away, over_2_5, under_2_5, btts_yes, btts_no",
        json_schema_extra={"example": {
            "home": 1.65, "draw": 3.80, "away": 5.50,
            "over_2_5": 1.72, "under_2_5": 2.08,
            "btts_yes": 1.80, "btts_no": 1.90,
        }},
    )


@router.post("/markets", summary="All betting markets for a fixture")
def all_markets(req: MarketsRequest) -> Dict[str, Any]:
    """
    Full market breakdown for a pre-match fixture.

    Includes every major betting market available on German platforms
    (Tipico, Lotto ODDSET, Bwin, Betano) with value-bet detection
    and Kelly Criterion stake recommendations.
    """
    try:
        kelly = KellyCriterion(kelly_fraction=req.kelly_fraction)

        # ── Features + models ─────────────────────────────────────────────────
        raw      = get_team_features(req.home_team, req.away_team)
        features = _fe.build_prematch_features(raw)

        poisson_out  = _poisson.predict(features)
        logistic_out = _logistic.predict(features)
        xgb_out      = _xgboost.predict(features)

        combined   = _ensemble.combine_prematch([poisson_out, logistic_out, xgb_out])
        confidence = _ensemble.confidence_score([poisson_out, logistic_out, xgb_out])
        calibrated = _calibrator.calibrate(combined)

        exp_h = poisson_out["expected_home_goals"]
        exp_a = poisson_out["expected_away_goals"]

        # ── All markets ───────────────────────────────────────────────────────
        ou_out   = _ou.predict(exp_h, exp_a)
        btts_out = _btts.predict(exp_h, exp_a)
        dc_out   = _dc.predict(calibrated["home_win"], calibrated["draw"], calibrated["away_win"])
        cs_out   = _cs.predict(exp_h, exp_a, top_n=10)
        ah_out   = _ah.all_lines(exp_h, exp_a)
        late_out = _late_goal.predict(exp_h, exp_a)

        # ── Value bets + Kelly ────────────────────────────────────────────────
        odds = req.bookmaker_odds or {}
        model_probs = {
            "home":      calibrated["home_win"],
            "draw":      calibrated["draw"],
            "away":      calibrated["away_win"],
            "over_2_5":  ou_out.get("over_2_5", 0),
            "under_2_5": ou_out.get("under_2_5", 0),
            "btts_yes":  btts_out.get("btts_yes", 0),
            "btts_no":   btts_out.get("btts_no", 0),
            "1X":        dc_out.get("1X", 0),
            "X2":        dc_out.get("X2", 0),
            "12":        dc_out.get("12", 0),
        }

        value_bets = []
        kelly_bets = []
        for market, book_odds in odds.items():
            if book_odds <= 1.0:
                continue
            model_prob = model_probs.get(market)
            if model_prob is None:
                continue
            implied = 1.0 / book_odds
            edge    = model_prob - implied
            vb = {
                "market":                market,
                "model_probability":     round(model_prob, 4),
                "bookmaker_probability": round(implied, 4),
                "edge":                  round(edge, 4),
                "bookmaker_odds":        book_odds,
                "is_value_bet":          edge > 0.03,
            }
            value_bets.append(vb)

            if edge > 0.03:
                k = kelly.calculate(model_prob, book_odds, req.bankroll)
                k["market"] = market
                kelly_bets.append(k)

        value_bets.sort(key=lambda x: x["edge"], reverse=True)
        kelly_bets.sort(key=lambda x: x["edge"], reverse=True)

        # Portfolio Kelly scaling
        if kelly_bets:
            kelly_bets = kelly.calculate_portfolio(
                [{"market": k["market"],
                  "model_probability": model_probs[k["market"]],
                  "bookmaker_odds": odds[k["market"]]}
                 for k in kelly_bets],
                bankroll=req.bankroll,
            )

        return {
            "match":      f"{req.home_team} vs {req.away_team}",
            "home_team":  req.home_team,
            "away_team":  req.away_team,
            "predicted_at": datetime.utcnow().isoformat(),
            "confidence_score": round(confidence, 4),

            "markets": {
                "1x2": {
                    "home": calibrated["home_win"],
                    "draw": calibrated["draw"],
                    "away": calibrated["away_win"],
                },
                "over_under": ou_out,
                "btts": btts_out,
                "double_chance": dc_out,
                "correct_score": cs_out,
                "asian_handicap": ah_out,
                "late_goals": {
                    "after_75":       late_out["after_75"],
                    "after_90":       late_out["after_90"],
                    "home_late_goal": late_out["home_late_goal"],
                    "away_late_goal": late_out["away_late_goal"],
                },
            },

            "expected_goals": {
                "home":  round(exp_h, 3),
                "away":  round(exp_a, 3),
                "total": round(exp_h + exp_a, 3),
            },

            "value_bets":   value_bets,
            "kelly_stakes": kelly_bets,
            "bankroll":     req.bankroll,

            "model_detail": {
                "poisson":  {k: v for k, v in poisson_out.items()  if k != "model"},
                "logistic": {k: v for k, v in logistic_out.items() if k != "model"},
                "xgboost":  {k: v for k, v in xgb_out.items()      if k != "model"},
            },
        }

    except Exception as e:
        logger.exception("Markets endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))
