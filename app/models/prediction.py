from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)

    # When this prediction was made and at what match minute
    predicted_at = Column(DateTime, default=datetime.utcnow)
    match_minute = Column(Integer, default=0)  # 0 = pre-match

    # Win/Draw/Loss probabilities
    home_win_prob = Column(Float, nullable=False)
    draw_prob = Column(Float, nullable=False)
    away_win_prob = Column(Float, nullable=False)

    # Expected goals
    expected_home_goals = Column(Float, nullable=False)
    expected_away_goals = Column(Float, nullable=False)

    # Late goal probabilities
    late_goal_prob_75 = Column(Float, nullable=True)   # P(goal after 75')
    late_goal_prob_90 = Column(Float, nullable=True)   # P(goal in 90+)

    # Player scoring probabilities (stored as JSON list)
    player_probs = Column(JSON, nullable=True)

    # Confidence & betting value
    confidence_score = Column(Float, nullable=False)
    value_bets = Column(JSON, nullable=True)  # [{market, model_prob, bookmaker_prob, edge}]

    # Per-model outputs for audit trail
    model_outputs = Column(JSON, nullable=True)

    # Actual outcome (filled in after match)
    actual_result = Column(String(1), nullable=True)    # H / D / A
    actual_home_goals = Column(Integer, nullable=True)
    actual_away_goals = Column(Integer, nullable=True)

    # Brier score (lower is better, filled post-match)
    brier_score = Column(Float, nullable=True)

    # Relationships
    match = relationship("Match", back_populates="predictions")
