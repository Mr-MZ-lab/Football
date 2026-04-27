from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Optional, List


class MatchBase(BaseModel):
    home_team: str
    away_team: str
    match_date: datetime
    competition: Optional[str] = None
    season: Optional[str] = None


class MatchCreate(MatchBase):
    pass


class MatchOut(MatchBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    result: Optional[str] = None


class MatchEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    minute: int
    extra_time: int = 0
    event_type: str
    player_name: Optional[str] = None
    team: Optional[str] = None
    xg_value: Optional[float] = None


class LiveMatchState(BaseModel):
    """Current live state passed to the live prediction endpoint."""
    match_id: int
    current_minute: int = Field(..., ge=0, le=130)
    home_goals: int = Field(0, ge=0)
    away_goals: int = Field(0, ge=0)
    home_red_cards: int = Field(0, ge=0, le=11)
    away_red_cards: int = Field(0, ge=0, le=11)
    home_shots: int = Field(0, ge=0)
    away_shots: int = Field(0, ge=0)
    home_shots_on_target: int = Field(0, ge=0)
    away_shots_on_target: int = Field(0, ge=0)
    home_possession: float = Field(50.0, ge=0.0, le=100.0)
    recent_events: Optional[List[MatchEventOut]] = []
    bookmaker_odds: Optional[dict] = None
