from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseDataProvider(ABC):
    """Abstract base class for all data providers (real APIs or mock)."""

    @abstractmethod
    def get_team_stats(self, team_name: str) -> Dict[str, Any]:
        """Return aggregate stats for a team."""

    @abstractmethod
    def get_player_stats(self, team_name: str) -> List[Dict[str, Any]]:
        """Return player stats for all players in a team."""

    @abstractmethod
    def get_historical_matches(
        self,
        team_name: str,
        last_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return last N completed matches for a team."""

    @abstractmethod
    def get_head_to_head(
        self,
        home_team: str,
        away_team: str,
        last_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return head-to-head match history."""

    @abstractmethod
    def get_live_match_events(self, match_id: int) -> List[Dict[str, Any]]:
        """Return events for a live match."""

    @abstractmethod
    def get_lineups(self, match_id: int) -> Dict[str, Any]:
        """Return confirmed lineups for a match."""
