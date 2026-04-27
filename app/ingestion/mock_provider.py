"""
Mock data provider — returns realistic synthetic football data so the system
works fully without external API keys.
"""
import random
from typing import List, Dict, Any
from datetime import datetime, timedelta
from app.ingestion.base import BaseDataProvider


# Realistic Premier League team parameters (attack / defense indices)
TEAM_PROFILES = {
    "Manchester City": {"attack": 1.85, "defense": 0.65, "avg_xg": 2.3, "avg_conceded_xg": 0.9},
    "Arsenal":         {"attack": 1.70, "defense": 0.72, "avg_xg": 2.0, "avg_conceded_xg": 1.0},
    "Liverpool":       {"attack": 1.75, "defense": 0.75, "avg_xg": 2.1, "avg_conceded_xg": 1.0},
    "Chelsea":         {"attack": 1.50, "defense": 0.90, "avg_xg": 1.7, "avg_conceded_xg": 1.2},
    "Tottenham":       {"attack": 1.55, "defense": 1.05, "avg_xg": 1.8, "avg_conceded_xg": 1.4},
    "Newcastle":       {"attack": 1.45, "defense": 0.85, "avg_xg": 1.6, "avg_conceded_xg": 1.1},
    "Manchester United":{"attack": 1.35, "defense": 1.10, "avg_xg": 1.5, "avg_conceded_xg": 1.5},
    "Aston Villa":     {"attack": 1.40, "defense": 1.00, "avg_xg": 1.5, "avg_conceded_xg": 1.3},
    "West Ham":        {"attack": 1.20, "defense": 1.15, "avg_xg": 1.3, "avg_conceded_xg": 1.5},
    "Brighton":        {"attack": 1.25, "defense": 1.05, "avg_xg": 1.4, "avg_conceded_xg": 1.3},
}

DEFAULT_PROFILE = {"attack": 1.10, "defense": 1.20, "avg_xg": 1.2, "avg_conceded_xg": 1.6}

PLAYER_PROFILES = {
    "Manchester City": [
        {"name": "Erling Haaland", "position": "FWD", "xg_per_90": 0.92, "shots_per_90": 4.2},
        {"name": "Phil Foden",     "position": "MID", "xg_per_90": 0.38, "shots_per_90": 2.0},
        {"name": "Kevin De Bruyne","position": "MID", "xg_per_90": 0.25, "shots_per_90": 1.5},
        {"name": "Bernardo Silva", "position": "MID", "xg_per_90": 0.18, "shots_per_90": 1.2},
        {"name": "Rodri",          "position": "MID", "xg_per_90": 0.08, "shots_per_90": 0.5},
    ],
    "Arsenal": [
        {"name": "Bukayo Saka",    "position": "FWD", "xg_per_90": 0.45, "shots_per_90": 2.5},
        {"name": "Gabriel Martinelli","position": "FWD","xg_per_90": 0.38,"shots_per_90": 2.2},
        {"name": "Leandro Trossard","position":"FWD", "xg_per_90": 0.35, "shots_per_90": 2.0},
        {"name": "Martin Odegaard","position": "MID", "xg_per_90": 0.28, "shots_per_90": 1.8},
        {"name": "Gabriel Jesus",  "position": "FWD", "xg_per_90": 0.40, "shots_per_90": 2.3},
    ],
    "Liverpool": [
        {"name": "Mohamed Salah",  "position": "FWD", "xg_per_90": 0.62, "shots_per_90": 3.1},
        {"name": "Darwin Nunez",   "position": "FWD", "xg_per_90": 0.55, "shots_per_90": 2.8},
        {"name": "Diogo Jota",     "position": "FWD", "xg_per_90": 0.48, "shots_per_90": 2.4},
        {"name": "Luis Diaz",      "position": "FWD", "xg_per_90": 0.35, "shots_per_90": 2.0},
        {"name": "Alexis Mac Allister","position":"MID","xg_per_90": 0.18,"shots_per_90": 1.2},
    ],
}

DEFAULT_PLAYERS = [
    {"name": "Player A", "position": "FWD", "xg_per_90": 0.30, "shots_per_90": 1.8},
    {"name": "Player B", "position": "FWD", "xg_per_90": 0.25, "shots_per_90": 1.5},
    {"name": "Player C", "position": "MID", "xg_per_90": 0.15, "shots_per_90": 1.0},
    {"name": "Player D", "position": "MID", "xg_per_90": 0.10, "shots_per_90": 0.8},
    {"name": "Player E", "position": "MID", "xg_per_90": 0.08, "shots_per_90": 0.6},
]


class MockDataProvider(BaseDataProvider):
    """Returns deterministic + lightly randomised football data for testing."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def _profile(self, team: str) -> Dict:
        return TEAM_PROFILES.get(team, DEFAULT_PROFILE)

    def get_team_stats(self, team_name: str) -> Dict[str, Any]:
        p = self._profile(team_name)
        return {
            "team": team_name,
            "attack_strength": p["attack"],
            "defense_weakness": p["defense"],
            "avg_goals_scored": p["avg_xg"] * self._rng.uniform(0.9, 1.1),
            "avg_goals_conceded": p["avg_conceded_xg"] * self._rng.uniform(0.9, 1.1),
            "avg_xg_scored": p["avg_xg"],
            "avg_xg_conceded": p["avg_conceded_xg"],
            "avg_possession": self._rng.uniform(45, 65),
            "avg_shots": self._rng.uniform(10, 20),
            "avg_shots_on_target": self._rng.uniform(3, 8),
            "form_points": self._rng.uniform(5, 15),
        }

    def get_player_stats(self, team_name: str) -> List[Dict[str, Any]]:
        players = PLAYER_PROFILES.get(team_name, DEFAULT_PLAYERS)
        result = []
        for i, p in enumerate(players):
            minutes = self._rng.randint(1200, 3000)
            result.append({
                "id": hash(f"{team_name}_{p['name']}") % 100000,
                "name": p["name"],
                "team": team_name,
                "position": p["position"],
                "xg_per_90": p["xg_per_90"],
                "shots_per_90": p["shots_per_90"],
                "goals": int(p["xg_per_90"] * minutes / 90 * self._rng.uniform(0.7, 1.3)),
                "minutes_played": minutes,
                "is_injured": self._rng.random() < 0.05,
                "is_suspended": self._rng.random() < 0.03,
                "penalties_taken": self._rng.randint(0, 3) if i == 0 else 0,
                "penalties_scored": self._rng.randint(0, 2) if i == 0 else 0,
            })
        return result

    def get_historical_matches(self, team_name: str, last_n: int = 10) -> List[Dict[str, Any]]:
        p = self._profile(team_name)
        matches = []
        base_date = datetime(2024, 1, 1)
        for i in range(last_n):
            home = self._rng.random() > 0.5
            opp_profile = DEFAULT_PROFILE
            home_goals = max(0, int(self._rng.gauss(p["avg_xg"] if home else p["avg_xg"] * 0.85, 1.0)))
            away_goals = max(0, int(self._rng.gauss(opp_profile["avg_xg"], 1.0)))
            matches.append({
                "date": (base_date + timedelta(weeks=i)).isoformat(),
                "home_team": team_name if home else "Opponent",
                "away_team": "Opponent" if home else team_name,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_xg": round(self._rng.uniform(0.5, 3.0), 2),
                "away_xg": round(self._rng.uniform(0.5, 2.5), 2),
                "home_possession": round(self._rng.uniform(40, 65), 1),
                "home_shots": self._rng.randint(8, 22),
                "away_shots": self._rng.randint(5, 18),
                "competition": "Premier League",
            })
        return matches

    def get_head_to_head(self, home_team: str, away_team: str, last_n: int = 10) -> List[Dict[str, Any]]:
        hp = self._profile(home_team)
        ap = self._profile(away_team)
        matches = []
        base_date = datetime(2022, 1, 1)
        for i in range(last_n):
            hg = max(0, int(self._rng.gauss(hp["avg_xg"], 1.0)))
            ag = max(0, int(self._rng.gauss(ap["avg_xg"], 1.0)))
            matches.append({
                "date": (base_date + timedelta(weeks=i * 12)).isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": hg,
                "away_goals": ag,
                "home_xg": round(self._rng.uniform(0.8, 3.0), 2),
                "away_xg": round(self._rng.uniform(0.8, 2.5), 2),
            })
        return matches

    def get_live_match_events(self, match_id: int) -> List[Dict[str, Any]]:
        return []  # In live mode events are pushed externally

    def get_lineups(self, match_id: int) -> Dict[str, Any]:
        return {"home": [], "away": []}
