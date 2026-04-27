from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ── Request schemas ───────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    home_team: str = Field(..., json_schema_extra={"example": "Manchester City"})
    away_team: str = Field(..., json_schema_extra={"example": "Liverpool"})
    competition: Optional[str] = Field(None, json_schema_extra={"example": "Premier League"})
    match_date: Optional[datetime] = None
    include_player_probs: bool = False
    bookmaker_odds: Optional[Dict[str, float]] = Field(
        None, json_schema_extra={"example": {"home": 2.1, "draw": 3.4, "away": 3.8}}
    )


class LivePredictionRequest(BaseModel):
    match_id: Optional[int] = None
    home_team: str
    away_team: str
    current_minute: int = Field(..., ge=0, le=130)
    home_goals: int = Field(0, ge=0)
    away_goals: int = Field(0, ge=0)
    home_red_cards: int = Field(0, ge=0)
    away_red_cards: int = Field(0, ge=0)
    home_shots: int = Field(0, ge=0)
    away_shots: int = Field(0, ge=0)
    home_shots_on_target: int = Field(0, ge=0)
    away_shots_on_target: int = Field(0, ge=0)
    home_possession: float = Field(50.0, ge=0.0, le=100.0)
    bookmaker_odds: Optional[Dict[str, float]] = None


class PlayerProbabilityRequest(BaseModel):
    home_team: str
    away_team: str
    competition: Optional[str] = None


# ── Sub-response schemas ──────────────────────────────────────────────────────

class WinProbability(BaseModel):
    home: float = Field(..., description="Home win probability")
    draw: float = Field(..., description="Draw probability")
    away: float = Field(..., description="Away win probability")


class ExpectedGoals(BaseModel):
    home: float = Field(..., description="Expected goals for home team")
    away: float = Field(..., description="Expected goals for away team")
    total: float = Field(..., description="Total expected goals")


class LateGoalProbability(BaseModel):
    after_75: float = Field(..., description="Probability of at least one goal after minute 75")
    after_90: float = Field(..., description="Probability of at least one goal in 90+ (stoppage)")
    home_late_goal: float = Field(..., description="Home team prob of late goal")
    away_late_goal: float = Field(..., description="Away team prob of late goal")


class ValueBet(BaseModel):
    market: str                  # e.g. "home_win"
    model_probability: float
    bookmaker_probability: float  # 1 / bookmaker_odds
    edge: float                  # model_prob - bookmaker_prob
    bookmaker_odds: float


class ModelOutputs(BaseModel):
    poisson: Dict[str, Any]
    logistic: Dict[str, Any]
    xgboost: Dict[str, Any]


# ── Response schemas ──────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    match: str
    home_team: str
    away_team: str
    predicted_at: datetime

    win_probability: WinProbability
    expected_goals: ExpectedGoals
    late_goal_probability: LateGoalProbability
    player_scoring_probabilities: Optional[List[Dict[str, Any]]] = []
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    value_bets: Optional[List[ValueBet]] = []

    # Per-model breakdown for transparency
    model_outputs: Optional[ModelOutputs] = None


class LivePredictionResponse(BaseModel):
    match: str
    home_team: str
    away_team: str
    current_minute: int
    current_score: str
    predicted_at: datetime

    win_probability: WinProbability
    expected_remaining_goals: ExpectedGoals
    late_goal_probability: LateGoalProbability
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    value_bets: Optional[List[ValueBet]] = []

    # Momentum indicator
    momentum: Optional[Dict[str, float]] = None  # {"home": 0.6, "away": 0.4}


class PlayerProbabilityResponse(BaseModel):
    match: str
    home_team: str
    away_team: str
    predicted_at: datetime
    home_players: List[Dict[str, Any]]
    away_players: List[Dict[str, Any]]
    confidence_score: float
