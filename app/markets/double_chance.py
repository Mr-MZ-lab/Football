"""
Double Chance market.

Covers two outcomes at once — popular in German Lotto/Tipico:
  1X  = home win OR draw       (eliminates away win risk)
  X2  = draw OR away win       (eliminates home win risk)
  12  = home win OR away win   (eliminates draw risk)
"""
from typing import Dict, Any


class DoubleChanceModel:

    def predict(
        self,
        home_win: float,
        draw: float,
        away_win: float,
    ) -> Dict[str, Any]:
        """
        Args:
            home_win, draw, away_win: pre-match 1X2 probabilities (must sum to ~1)
        """
        return {
            "1X": round(home_win + draw, 4),   # home win or draw
            "X2": round(draw + away_win, 4),   # draw or away win
            "12": round(home_win + away_win, 4),  # home or away win (no draw)
        }
