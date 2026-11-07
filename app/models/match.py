from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(50), unique=True, nullable=True)

    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    match_date = Column(DateTime, nullable=False)
    competition = Column(String(100))
    season = Column(String(20))
    round = Column(String(50))

    # Result (null until match ends)
    home_goals = Column(Integer, nullable=True)
    away_goals = Column(Integer, nullable=True)
    result = Column(String(1), nullable=True)  # H / D / A

    # xG
    home_xg = Column(Float, nullable=True)
    away_xg = Column(Float, nullable=True)

    # Live state
    is_live = Column(Boolean, default=False)
    current_minute = Column(Integer, default=0)
    status = Column(String(20), default="scheduled")  # scheduled / live / finished

    # Possession & shots (full match)
    home_possession = Column(Float, nullable=True)
    away_possession = Column(Float, nullable=True)
    home_shots = Column(Integer, nullable=True)
    away_shots = Column(Integer, nullable=True)
    home_shots_on_target = Column(Integer, nullable=True)
    away_shots_on_target = Column(Integer, nullable=True)

    # Lineups stored as JSON arrays of player IDs
    home_lineup = Column(JSON, nullable=True)
    away_lineup = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    events = relationship("MatchEvent", back_populates="match", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="match", cascade="all, delete-orphan")


class MatchEvent(Base):
    __tablename__ = "match_events"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)

    minute = Column(Integer, nullable=False)
    extra_time = Column(Integer, default=0)  # stoppage time minutes
    event_type = Column(String(30), nullable=False)  # goal / yellow_card / red_card / substitution / penalty

    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=True)
    player_name = Column(String(100), nullable=True)

    # For goals: xG value of the shot
    xg_value = Column(Float, nullable=True)

    # For substitutions: player coming on
    player_in_id = Column(Integer, ForeignKey("players.id"), nullable=True)
    player_in_name = Column(String(100), nullable=True)

    extra_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    match = relationship("Match", back_populates="events")
