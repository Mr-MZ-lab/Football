"""
football-data.org API provider.

Free tier: 10 req/min, covers Bundesliga (BL1), Premier League (PL),
Champions League (CL), and more.

Get a free key at: https://www.football-data.org/client/register
Set FOOTBALL_DATA_API_KEY in your .env file.

Falls back gracefully to MockDataProvider when key is missing or rate-limited.
"""
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional

import httpx

from app.ingestion.base import BaseDataProvider
from app.ingestion.mock_provider import MockDataProvider, TEAM_PROFILES, DEFAULT_PROFILE

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"

# football-data.org competition codes
LEAGUE_CODES = {
    "Bundesliga":      "BL1",
    "Premier League":  "PL",
    "La Liga":         "PD",
    "Serie A":         "SA",
    "Ligue 1":         "FL1",
    "Champions League":"CL",
}

# Map football-data.org team names → our internal names
TEAM_NAME_MAP = {
    "FC Bayern München":          "Bayern München",
    "Borussia Dortmund":          "Borussia Dortmund",
    "Bayer 04 Leverkusen":        "Bayer Leverkusen",
    "RB Leipzig":                 "RB Leipzig",
    "Eintracht Frankfurt":        "Eintracht Frankfurt",
    "VfB Stuttgart":              "VfB Stuttgart",
    "Sport-Club Freiburg":        "SC Freiburg",
    "Borussia Mönchengladbach":   "Borussia Mönchengladbach",
    "Werder Bremen":              "Werder Bremen",
    "1. FC Union Berlin":         "Union Berlin",
    "TSG 1899 Hoffenheim":        "TSG Hoffenheim",
    "FC Augsburg":                "FC Augsburg",
    "VfL Wolfsburg":              "VfL Wolfsburg",
    "1. FSV Mainz 05":            "FSV Mainz 05",
    "1. FC Heidenheim 1846":      "1. FC Heidenheim",
    "Holstein Kiel":              "Holstein Kiel",
    "VfL Bochum 1848":            "VfL Bochum",
    "FC St. Pauli":               "FC St. Pauli",
    # Premier League
    "Manchester City FC":         "Manchester City",
    "Arsenal FC":                 "Arsenal",
    "Liverpool FC":               "Liverpool",
    "Chelsea FC":                 "Chelsea",
    "Tottenham Hotspur FC":       "Tottenham",
}


def _normalize_team(raw_name: str) -> str:
    return TEAM_NAME_MAP.get(raw_name, raw_name)


class FootballDataProvider(BaseDataProvider):
    """
    Wraps football-data.org v4 API.
    Falls back to MockDataProvider for stat endpoints (API only provides
    match schedules + results, not detailed xG / player stats).
    """

    def __init__(self, api_key: str):
        self._key = api_key
        self._mock = MockDataProvider()
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"X-Auth-Token": api_key},
            timeout=10.0,
        )

    # ── Schedule / results (real API) ─────────────────────────────────────────

    def get_todays_matches(self, league: str = "Bundesliga") -> List[Dict[str, Any]]:
        """Fetch today's fixtures from football-data.org."""
        code = LEAGUE_CODES.get(league, "BL1")
        today = date.today().isoformat()
        try:
            resp = self._client.get(
                f"/competitions/{code}/matches",
                params={"dateFrom": today, "dateTo": today, "status": "SCHEDULED,LIVE"},
            )
            resp.raise_for_status()
            matches = resp.json().get("matches", [])
            return [self._parse_match(m) for m in matches]
        except Exception as e:
            logger.warning(f"football-data.org today's matches failed: {e}. Using mock.")
            return self._mock.get_todays_matches(league)

    def get_team_standings(self, league: str = "Bundesliga") -> List[Dict[str, Any]]:
        """Fetch current league table."""
        code = LEAGUE_CODES.get(league, "BL1")
        try:
            resp = self._client.get(f"/competitions/{code}/standings")
            resp.raise_for_status()
            table = resp.json()["standings"][0]["table"]
            return [
                {
                    "position": r["position"],
                    "team": _normalize_team(r["team"]["name"]),
                    "played": r["playedGames"],
                    "won": r["won"],
                    "drawn": r["draw"],
                    "lost": r["lost"],
                    "goals_for": r["goalsFor"],
                    "goals_against": r["goalsAgainst"],
                    "goal_diff": r["goalDifference"],
                    "points": r["points"],
                }
                for r in table
            ]
        except Exception as e:
            logger.warning(f"Standings fetch failed: {e}")
            return []

    # ── Stat endpoints — delegate to mock (API doesn't expose xG / player stats on free tier) ──

    def get_team_stats(self, team_name: str) -> Dict[str, Any]:
        return self._mock.get_team_stats(team_name)

    def get_player_stats(self, team_name: str) -> List[Dict[str, Any]]:
        return self._mock.get_player_stats(team_name)

    def get_historical_matches(self, team_name: str, last_n: int = 10) -> List[Dict[str, Any]]:
        return self._mock.get_historical_matches(team_name, last_n)

    def get_head_to_head(self, home_team: str, away_team: str, last_n: int = 10) -> List[Dict[str, Any]]:
        return self._mock.get_head_to_head(home_team, away_team, last_n)

    def get_live_match_events(self, match_id: int) -> List[Dict[str, Any]]:
        return self._mock.get_live_match_events(match_id)

    def get_lineups(self, match_id: int) -> Dict[str, Any]:
        try:
            resp = self._client.get(f"/matches/{match_id}")
            resp.raise_for_status()
            data = resp.json()
            lineups = data.get("lineups", {})
            home_lineup = [p["name"] for p in lineups.get("homeTeam", {}).get("startXI", [])]
            away_lineup = [p["name"] for p in lineups.get("awayTeam", {}).get("startXI", [])]
            return {"home": home_lineup, "away": away_lineup}
        except Exception as e:
            logger.warning(f"Lineup fetch failed for match {match_id}: {e}")
            return {"home": [], "away": []}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parse_match(self, m: Dict) -> Dict[str, Any]:
        home = _normalize_team(m["homeTeam"]["name"])
        away = _normalize_team(m["awayTeam"]["name"])
        return {
            "match_id": str(m["id"]),
            "home_team": home,
            "away_team": away,
            "kickoff": m.get("utcDate", ""),
            "competition": m.get("competition", {}).get("name", ""),
            "status": m.get("status", "SCHEDULED"),
            "home_goals": m.get("score", {}).get("fullTime", {}).get("home"),
            "away_goals": m.get("score", {}).get("fullTime", {}).get("away"),
            "source": "football-data.org",
        }
