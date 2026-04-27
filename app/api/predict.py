"""
/predict endpoint — pre-match prediction.

Runs the full pre-match pipeline:
  1. Ingest team/player data (mock or real API)
  2. Build features
  3. Run Poisson + Logistic + XGBoost
  4. Ensemble
  5. Late-goal predictor
  6. Player probabilities (optional)
  7. Value bet detection
"""
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    WinProbability,
    ExpectedGoals,
    LateGoalProbability,
    ModelOutputs,
    ValueBet,
)
from app.ingestion.data_loader import get_team_features, get_player_features
from app.processing.feature_engineer import FeatureEngineer
from app.models_ml.poisson_model import PoissonModel
from app.models_ml.logistic_model import LogisticModel
from app.models_ml.xgboost_model import XGBoostModel
from app.models_ml.player_model import PlayerScoringModel
from app.models_ml.late_goal_predictor import LateGoalPredictor
from app.ensemble.ensemble import EnsembleEngine

logger = logging.getLogger(__name__)
router = APIRouter()

# Singletons — initialised once per worker process
_fe = FeatureEngineer()
_poisson = PoissonModel()
_logistic = LogisticModel()
_xgboost = XGBoostModel()
_player_model = PlayerScoringModel()
_late_goal = LateGoalPredictor()
_ensemble = EnsembleEngine()


@router.post("/predict", response_model=PredictionResponse, summary="Pre-match prediction")
def predict_match(req: PredictionRequest) -> PredictionResponse:
    """
    Generate a pre-match prediction for a given fixture.

    Returns win/draw/loss probabilities, expected goals, late-goal probabilities,
    optional player scoring probabilities, and value bets vs bookmaker odds.
    """
    try:
        # ── Data & features ───────────────────────────────────────────────────
        raw = get_team_features(req.home_team, req.away_team)
        features = _fe.build_prematch_features(raw)

        # ── Run models ────────────────────────────────────────────────────────
        poisson_out = _poisson.predict(features)
        logistic_out = _logistic.predict(features)
        xgb_out = _xgboost.predict(features)

        # ── Ensemble ──────────────────────────────────────────────────────────
        combined = _ensemble.combine_prematch([poisson_out, logistic_out, xgb_out])
        confidence = _ensemble.confidence_score([poisson_out, logistic_out, xgb_out])

        # ── Expected goals ────────────────────────────────────────────────────
        exp_h = poisson_out["expected_home_goals"]
        exp_a = poisson_out["expected_away_goals"]

        # ── Late goal probabilities ───────────────────────────────────────────
        late_goals = _late_goal.predict(exp_h, exp_a)

        # ── Value bets ────────────────────────────────────────────────────────
        value_bets = []
        if req.bookmaker_odds:
            value_bets = _compute_value_bets(combined, req.bookmaker_odds)

        # ── Player probabilities (optional) ──────────────────────────────────
        player_probs = []
        if req.include_player_probs:
            pdata = get_player_features(req.home_team, req.away_team)
            home_pp = _player_model.predict(pdata["home_players"], exp_h, is_home=True)
            away_pp = _player_model.predict(pdata["away_players"], exp_a, is_home=False)
            player_probs = home_pp + away_pp

        return PredictionResponse(
            match=f"{req.home_team} vs {req.away_team}",
            home_team=req.home_team,
            away_team=req.away_team,
            predicted_at=datetime.utcnow(),
            win_probability=WinProbability(
                home=combined["home_win"],
                draw=combined["draw"],
                away=combined["away_win"],
            ),
            expected_goals=ExpectedGoals(
                home=round(exp_h, 3),
                away=round(exp_a, 3),
                total=round(exp_h + exp_a, 3),
            ),
            late_goal_probability=LateGoalProbability(
                after_75=late_goals["after_75"],
                after_90=late_goals["after_90"],
                home_late_goal=late_goals["home_late_goal"],
                away_late_goal=late_goals["away_late_goal"],
            ),
            player_scoring_probabilities=player_probs,
            confidence_score=confidence,
            value_bets=value_bets,
            model_outputs=ModelOutputs(
                poisson={k: v for k, v in poisson_out.items() if k != "model"},
                logistic={k: v for k, v in logistic_out.items() if k != "model"},
                xgboost={k: v for k, v in xgb_out.items() if k != "model"},
            ),
        )

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


def _compute_value_bets(probs, odds) -> list:
    mapping = {"home": "home_win", "draw": "draw", "away": "away_win"}
    bets = []
    for market, prob_key in mapping.items():
        book_odds = odds.get(market, 0)
        if book_odds <= 1.0:
            continue
        model_prob = probs.get(prob_key, 0.0)
        book_prob = 1.0 / book_odds
        edge = model_prob - book_prob
        if edge > 0.03:
            bets.append(ValueBet(
                market=market,
                model_probability=round(model_prob, 4),
                bookmaker_probability=round(book_prob, 4),
                edge=round(edge, 4),
                bookmaker_odds=book_odds,
            ))
    bets.sort(key=lambda b: b.edge, reverse=True)
    return bets
