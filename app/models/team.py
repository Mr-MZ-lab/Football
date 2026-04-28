from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    short_name = Column(String(20))
    league = Column(String(100))
    country = Column(String(100))

    # Aggregate season stats (updated after each match)
    avg_goals_scored = Column(Float, default=0.0)
    avg_goals_conceded = Column(Float, default=0.0)
    avg_xg_scored = Column(Float, default=0.0)
    avg_xg_conceded = Column(Float, default=0.0)
    avg_possession = Column(Float, default=50.0)
    avg_shots = Column(Float, default=0.0)
    avg_shots_on_target = Column(Float, default=0.0)

    # Form (last 5 matches: W=3, D=1, L=0)
    form_points = Column(Float, default=0.0)

    # Home / Away splits
    home_avg_goals_scored = Column(Float, default=0.0)
    away_avg_goals_scored = Column(Float, default=0.0)

    # Attack / Defense strength indices (Dixon-Coles parameters)
    attack_strength = Column(Float, default=1.0)
    defense_weakness = Column(Float, default=1.0)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    players = relationship("Player", back_populates="team")
