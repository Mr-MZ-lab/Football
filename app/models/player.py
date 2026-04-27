from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    position = Column(String(20))  # GK, DEF, MID, FWD

    # Season aggregates
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    appearances = Column(Integer, default=0)
    minutes_played = Column(Integer, default=0)

    # xG statistics
    total_xg = Column(Float, default=0.0)
    xg_per_90 = Column(Float, default=0.0)
    shots_per_90 = Column(Float, default=0.0)
    shots_on_target_per_90 = Column(Float, default=0.0)

    # Penalty stats
    penalties_taken = Column(Integer, default=0)
    penalties_scored = Column(Integer, default=0)

    # Current status
    is_injured = Column(Boolean, default=False)
    injury_return_date = Column(DateTime, nullable=True)
    is_suspended = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    team = relationship("Team", back_populates="players")
