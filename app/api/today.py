"""
GET /today — today's matches with automatic predictions.

Flow:
  1. Fetch today's fixtures (football-data.org if key available, else mock)
  2. For each match: run full pre-match prediction pipeline
  3. Fetch bookmaker odds (The Odds API if key available)
  4. Compute value bets + Kelly stakes
  5. Return ranked list (best value bets first)
"""
import logging
from datetime import datetime, date
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Query, HTTPException

from app.config import settings
from app.ingestion.data_loader import get_team_features, get_player_features
from app.processing.feature_engineer import FeatureEngineer
from app.models_ml.poisson_model import PoissonModel
from app.models_ml.logistic_model import LogisticModel
from app.models_ml.xgboost_model import XGBoostModel
from app.models_ml.late_goal_predictor import LateGoalPredictor
from app.models_ml.player_model import PlayerScoringModel
from app.ensemble.ensemble import EnsembleEngine
from app.markets import OverUnderModel, BTTSModel, DoubleChanceModel, CorrectScoreModel, KellyCriterion
from app.markets.calibration import BetaCalibrator

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Singletons ────────────────────────────────────────────────────────────────
_fe          = FeatureEngineer()
_poisson     = PoissonModel()
_logistic    = LogisticModel()
_xgboost     = XGBoostModel()
_late_goal   = LateGoalPredictor()
_player_model= PlayerScoringModel()
_ensemble    = EnsembleEngine()
_ou          = OverUnderModel()
_btts        = BTTSModel()
_dc          = DoubleChanceModel()
_cs          = CorrectScoreModel()
_kelly       = KellyCriterion(kelly_fraction=0.5)
_calibrator  = BetaCalibrator()


def _get_schedule_provider(league: str):
    """Return real provider if API key configured, else mock."""
    key = settings.FOOTBALL_DATA_API_KEY
    if key:
        from app.ingestion.football_data_provider import FootballDataProvider
        return FootballDataProvider(key)
    from app.ingestion.mock_provider import MockDataProvider
    return MockDataProvider()


def _get_odds_for_match(home: str, away: str, league: str) -> Dict[str, Any]:
    key = settings.ODDS_API_KEY
    if not key:
        return {}
    try:
        from app.ingestion.odds_provider import OddsProvider
        provider = OddsProvider(key)
        return provider.get_match_odds(home, away, league)
    except Exception as e:
        logger.warning(f"Odds fetch failed: {e}")
        return {}


def _predict_match(home: str, away: str, odds_data: Dict) -> Dict[str, Any]:
    """Run the full prediction pipeline for one match."""
    raw      = get_team_features(home, away)
    features = _fe.build_prematch_features(raw)

    poisson_out  = _poisson.predict(features)
    logistic_out = _logistic.predict(features)
    xgb_out      = _xgboost.predict(features)

    combined   = _ensemble.combine_prematch([poisson_out, logistic_out, xgb_out])
    confidence = _ensemble.confidence_score([poisson_out, logistic_out, xgb_out])

    # Calibrate
    calibrated = _calibrator.calibrate(combined)

    exp_h = poisson_out["expected_home_goals"]
    exp_a = poisson_out["expected_away_goals"]

    ou_out   = _ou.predict(exp_h, exp_a)
    btts_out = _btts.predict(exp_h, exp_a)
    dc_out   = _dc.predict(calibrated["home_win"], calibrated["draw"], calibrated["away_win"])
    cs_top5  = _cs.predict(exp_h, exp_a, top_n=5)
    late_out = _late_goal.predict(exp_h, exp_a)

    # Value bets + Kelly from real odds
    value_bets = []
    best_odds = odds_data.get("best_odds", {})
    kelly_stakes = []

    market_checks = [
        ("home_win", "home",      best_odds.get("home", 0)),
        ("draw",     "draw",      best_odds.get("draw", 0)),
        ("away_win", "away",      best_odds.get("away", 0)),
        ("over_2_5", "over_2_5",  best_odds.get("over_2_5", 0)),
        ("btts_yes", "btts_yes",  best_odds.get("btts_yes", 0)),
    ]

    model_probs = {
        "home_win": calibrated["home_win"],
        "draw":     calibrated["draw"],
        "away_win": calibrated["away_win"],
        "over_2_5": ou_out.get("over_2_5", 0),
        "btts_yes": btts_out.get("btts_yes", 0),
    }

    for prob_key, market_key, book_odds in market_checks:
        if book_odds <= 1.0:
            continue
        model_prob  = model_probs.get(prob_key, 0)
        implied     = 1.0 / book_odds
        edge        = model_prob - implied
        if edge > 0.03:
            value_bets.append({
                "market":             market_key,
                "model_probability":  round(model_prob, 4),
                "bookmaker_probability": round(implied, 4),
                "edge":               round(edge, 4),
                "bookmaker_odds":     book_odds,
            })
            k = _kelly.calculate(model_prob, book_odds)
            k["market"] = market_key
            kelly_stakes.append(k)

    value_bets.sort(key=lambda x: x["edge"], reverse=True)

    return {
        "win_probability": {
            "home": calibrated["home_win"],
            "draw": calibrated["draw"],
            "away": calibrated["away_win"],
        },
        "expected_goals": {
            "home":  round(exp_h, 3),
            "away":  round(exp_a, 3),
            "total": round(exp_h + exp_a, 3),
        },
        "over_under": ou_out,
        "btts":        btts_out,
        "double_chance": dc_out,
        "correct_score_top5": cs_top5,
        "late_goal_probability": {
            "after_75":       late_out["after_75"],
            "after_90":       late_out["after_90"],
            "home_late_goal": late_out["home_late_goal"],
            "away_late_goal": late_out["away_late_goal"],
        },
        "confidence_score": round(confidence, 4),
        "value_bets":   value_bets,
        "kelly_stakes": kelly_stakes,
        "bookmaker_odds": best_odds if best_odds else None,
    }


@router.get("/today", summary="Today's matches with predictions")
def todays_matches(
    league: str = Query("Bundesliga", description="League name (e.g. Bundesliga, Premier League)"),
    bankroll: float = Query(1000.0, gt=0, description="Your bankroll for Kelly stake sizing"),
    include_players: bool = Query(False, description="Include player scoring probabilities"),
) -> Dict[str, Any]:
    """
    Fetch today's matches for the given league, run the full prediction
    pipeline on each one, and return results ranked by best value-bet edge.

    Uses football-data.org for the schedule (falls back to mock fixtures)
    and The Odds API for bookmaker odds (falls back to no odds).
    """
    try:
        provider = _get_schedule_provider(league)
        fixtures = provider.get_todays_matches(league)

        if not fixtures:
            return {
                "date": date.today().isoformat(),
                "league": league,
                "matches": [],
                "message": "No matches scheduled today.",
            }

        results = []
        for fix in fixtures:
            home = fix["home_team"]
            away = fix["away_team"]

            try:
                odds_data = _get_odds_for_match(home, away, league)
                prediction = _predict_match(home, away, odds_data)

                # Kelly uses passed bankroll
                for k in prediction.get("kelly_stakes", []):
                    k["recommended_stake"] = round(k["kelly_fraction"] * bankroll, 2)

                # Optionally include player probs
                player_probs = {}
                if include_players:
                    pdata = get_player_features(home, away)
                    exp_h = prediction["expected_goals"]["home"]
                    exp_a = prediction["expected_goals"]["away"]
                    player_probs = {
                        "home": _player_model.predict(pdata["home_players"], exp_h, is_home=True),
                        "away": _player_model.predict(pdata["away_players"], exp_a, is_home=False),
                    }

                results.append({
                    "match_id":   fix.get("match_id"),
                    "home_team":  home,
                    "away_team":  away,
                    "kickoff":    fix.get("kickoff"),
                    "competition":fix.get("competition", league),
                    "status":     fix.get("status", "scheduled"),
                    "source":     fix.get("source", "mock"),
                    "prediction": prediction,
                    **({"player_probabilities": player_probs} if include_players else {}),
                })
            except Exception as e:
                logger.warning(f"Prediction failed for {home} vs {away}: {e}")
                results.append({
                    "home_team": home,
                    "away_team": away,
                    "kickoff":   fix.get("kickoff"),
                    "error":     str(e),
                })

        # Rank by best value-bet edge (matches with no value bets go last)
        def sort_key(m):
            bets = m.get("prediction", {}).get("value_bets", [])
            return -bets[0]["edge"] if bets else 0

        results.sort(key=sort_key)

        return {
            "date":        date.today().isoformat(),
            "league":      league,
            "generated_at": datetime.utcnow().isoformat(),
            "total_matches": len(results),
            "data_sources": {
                "schedule": "football-data.org" if settings.FOOTBALL_DATA_API_KEY else "mock",
                "odds":     "the-odds-api.com"  if settings.ODDS_API_KEY          else "not configured",
            },
            "matches": results,
        }

    except Exception as e:
        logger.exception("Today's matches endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))
