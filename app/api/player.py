"""
/player-probability endpoint — per-player scoring probability.
"""
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from app.schemas.prediction import PlayerProbabilityRequest, PlayerProbabilityResponse
from app.ingestion.data_loader import get_team_features, get_player_features
from app.processing.feature_engineer import FeatureEngineer
from app.models_ml.poisson_model import PoissonModel
from app.models_ml.player_model import PlayerScoringModel
from app.ensemble.ensemble import EnsembleEngine

logger = logging.getLogger(__name__)
router = APIRouter()

_fe = FeatureEngineer()
_poisson = PoissonModel()
_player_model = PlayerScoringModel()
_ensemble = EnsembleEngine()


@router.post(
    "/player-probability",
    response_model=PlayerProbabilityResponse,
    summary="Per-player scoring probability",
)
def player_probability(req: PlayerProbabilityRequest) -> PlayerProbabilityResponse:
    """
    Estimate the probability of each player scoring in this fixture.

    Uses xG per 90 min, expected match minutes, and the team-level expected goals
    from the Poisson model as a scaling baseline.
    """
    try:
        # Team features → expected goals
        raw = get_team_features(req.home_team, req.away_team)
        features = _fe.build_prematch_features(raw)
        poisson_out = _poisson.predict(features)

        exp_h = poisson_out["expected_home_goals"]
        exp_a = poisson_out["expected_away_goals"]
        confidence = 0.72  # baseline for player model

        # Player probabilities
        pdata = get_player_features(req.home_team, req.away_team)
        home_pp = _player_model.predict(pdata["home_players"], exp_h, is_home=True)
        away_pp = _player_model.predict(pdata["away_players"], exp_a, is_home=False)

        return PlayerProbabilityResponse(
            match=f"{req.home_team} vs {req.away_team}",
            home_team=req.home_team,
            away_team=req.away_team,
            predicted_at=datetime.utcnow(),
            home_players=home_pp,
            away_players=away_pp,
            confidence_score=confidence,
        )

    except Exception as e:
        logger.exception("Player probability failed")
        raise HTTPException(status_code=500, detail=str(e))
