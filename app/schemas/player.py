from pydantic import BaseModel, ConfigDict
from typing import Optional


class PlayerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    team: str
    position: Optional[str] = None
    goals: int = 0
    total_xg: float = 0.0
    xg_per_90: float = 0.0
    is_injured: bool = False
    is_suspended: bool = False


class PlayerProbability(BaseModel):
    player_id: int
    player_name: str
    team: str
    position: Optional[str] = None
    scoring_probability: float
    xg_contribution: float
    is_penalty_taker: bool = False
