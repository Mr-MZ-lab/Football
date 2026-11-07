"""
/live endpoint — real-time in-play prediction.

Accepts the current match state (score, minute, red cards, shots, possession)
and returns updated probabilities using the live model ensemble.
"""
import logging
from fastapi import APIRouter, HTTPException
from app.schemas.prediction import LivePredictionRequest, LivePredictionResponse
from app.realtime.live_engine import LivePredictionEngine

logger = logging.getLogger(__name__)
router = APIRouter()

_engine = LivePredictionEngine()


@router.post("/live", response_model=LivePredictionResponse, summary="Live in-play prediction")
def live_prediction(req: LivePredictionRequest) -> LivePredictionResponse:
    """
    Update match outcome probabilities in real time.

    Call this endpoint after every significant match event (goal, red card,
    substitution) to get recalculated probabilities reflecting the current
    state of play.
    """
    try:
        return _engine.predict(req.model_dump())
    except Exception as e:
        logger.exception("Live prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
