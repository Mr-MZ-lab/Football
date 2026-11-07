"""
POST /late-scorer — Who is most likely to score after minute 90?

Takes the current live match state (score, minute, subs, stoppage time)
and returns a ranked list of players most likely to score in 90+.

This is the key question for in-play bettors on:
  - "Next Goal Scorer" markets
  - "Anytime Scorer" late markets
  - BTTS / Over 2.5 last-minute decisions
"""
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ingestion.data_loader import get_team_features, get_player_features
from app.processing.feature_engineer import FeatureEngineer
from app.models_ml.poisson_model import PoissonModel
from app.models_ml.late_scorer import LateScorer

logger = logging.getLogger(__name__)
router = APIRouter()

_fe          = FeatureEngineer()
_poisson     = PoissonModel()
_late_scorer = LateScorer()


# ── Request ───────────────────────────────────────────────────────────────────

class LateGoalScorerRequest(BaseModel):
    home_team: str = Field(..., json_schema_extra={"example": "Bayern München"})
    away_team: str = Field(..., json_schema_extra={"example": "Borussia Dortmund"})

    home_goals: int = Field(0, ge=0, description="Goals scored by home team so far")
    away_goals: int = Field(0, ge=0, description="Goals scored by away team so far")
    current_minute: int = Field(88, ge=75, le=130, description="Current match minute")
    stoppage_minutes: int = Field(5, ge=1, le=15, description="Expected stoppage time")

    # Substitutes — players who came on during the match
    home_subs: List[str] = Field(
        default=[],
        description="Names of home team substitutes currently on pitch",
        json_schema_extra={"example": ["Thomas Müller", "Serge Gnabry"]},
    )
    away_subs: List[str] = Field(
        default=[],
        description="Names of away team substitutes currently on pitch",
        json_schema_extra={"example": ["Donyell Malen"]},
    )

    top_n: int = Field(5, ge=1, le=20, description="How many players to return per team")


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/late-scorer", summary="Who scores after minute 90?")
def late_goal_scorer(req: LateGoalScorerRequest) -> Dict[str, Any]:
    """
    Returns the players most likely to score in stoppage time (90+).

    Accounts for:
    - Match state (losing team pushes everyone forward — even defenders)
    - Substitutes (fresh legs get a +20% bonus)
    - Penalty takers (foul probability spikes in 90+)
    - Position-specific desperation factors
    - Hazard-function goal rate (goals are 45% more frequent in 90+)
    """
    try:
        # ── Features + expected goals ─────────────────────────────────────
        raw      = get_team_features(req.home_team, req.away_team)
        features = _fe.build_prematch_features(raw)
        poisson  = _poisson.predict(features)

        lambda_home = poisson["expected_home_goals"]
        lambda_away = poisson["expected_away_goals"]

        home_diff = req.home_goals - req.away_goals  # >0 winning, <0 losing
        away_diff = -home_diff

        # ── Player data ───────────────────────────────────────────────────
        pdata        = get_player_features(req.home_team, req.away_team)
        home_players = pdata["home_players"]
        away_players = pdata["away_players"]

        # ── Team-level 90+ probability ────────────────────────────────────
        home_team_prob = _late_scorer.team_90plus_probability(
            lambda_home, req.stoppage_minutes, home_diff, is_home=True
        )
        away_team_prob = _late_scorer.team_90plus_probability(
            lambda_away, req.stoppage_minutes, away_diff, is_home=False
        )

        # ── Per-player rankings ───────────────────────────────────────────
        home_scorers = _late_scorer.predict(
            players=home_players,
            team_lambda=lambda_home,
            score_diff=home_diff,
            stoppage_mins=req.stoppage_minutes,
            subs=req.home_subs,
            is_home=True,
        )
        away_scorers = _late_scorer.predict(
            players=away_players,
            team_lambda=lambda_away,
            score_diff=away_diff,
            stoppage_mins=req.stoppage_minutes,
            subs=req.away_subs,
            is_home=False,
        )

        # ── Combined ranking across both teams ────────────────────────────
        all_scorers = home_scorers[:req.top_n] + away_scorers[:req.top_n]
        all_scorers.sort(key=lambda x: x["scoring_probability"], reverse=True)

        # ── Match narrative ───────────────────────────────────────────────
        score_str = f"{req.home_goals}-{req.away_goals}"
        narrative = _build_narrative(req, home_diff, home_team_prob, away_team_prob)

        return {
            "match":           f"{req.home_team} vs {req.away_team}",
            "current_score":   score_str,
            "current_minute":  req.current_minute,
            "stoppage_minutes": req.stoppage_minutes,
            "predicted_at":    datetime.utcnow().isoformat(),

            # ── Team-level headlines ──────────────────────────────────────
            "team_90plus": {
                req.home_team: home_team_prob,
                req.away_team: away_team_prob,
            },

            # ── Per-team rankings ─────────────────────────────────────────
            "home_scorers": home_scorers[:req.top_n],
            "away_scorers": away_scorers[:req.top_n],

            # ── Overall most likely scorer (both teams combined) ──────────
            "most_likely_scorer": all_scorers[0] if all_scorers else None,
            "top_scorers_combined": all_scorers[:req.top_n],

            # ── Context ───────────────────────────────────────────────────
            "narrative": narrative,
        }

    except Exception as e:
        logger.exception("Late scorer endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))


def _build_narrative(
    req: LateGoalScorerRequest,
    home_diff: int,
    home_prob: Dict,
    away_prob: Dict,
) -> str:
    """Generate a human-readable match situation summary."""
    parts = []

    if home_diff < 0:
        parts.append(
            f"{req.home_team} are losing — expect them to push everyone forward. "
            "Defenders and midfielders have elevated late-goal probability."
        )
    elif home_diff > 0:
        parts.append(
            f"{req.away_team} are chasing the game — "
            "their attacking players will take more risks."
        )
    else:
        parts.append("The match is level — both teams are motivated to find a winner.")

    if req.home_subs:
        parts.append(f"Home subs on pitch: {', '.join(req.home_subs)} (+20% bonus for freshness).")
    if req.away_subs:
        parts.append(f"Away subs on pitch: {', '.join(req.away_subs)} (+20% bonus for freshness).")

    p_h = home_prob["p_score_90plus"]
    p_a = away_prob["p_score_90plus"]
    parts.append(
        f"P(home scores in 90+) = {p_h:.1%} | "
        f"P(away scores in 90+) = {p_a:.1%}. "
        f"P(at least one 90+ goal) = {1-(1-p_h)*(1-p_a):.1%}."
    )

    return " ".join(parts)
