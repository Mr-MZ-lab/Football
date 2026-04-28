"""
Mock data provider — returns realistic synthetic football data so the system
works fully without external API keys.

Covers both Premier League and Bundesliga teams.
"""
import random
from typing import List, Dict, Any
from datetime import datetime, timedelta
from app.ingestion.base import BaseDataProvider


# ── Team profiles (attack / defense indices, avg xG) ─────────────────────────

TEAM_PROFILES = {
    # Premier League
    "Manchester City":   {"attack": 1.85, "defense": 0.65, "avg_xg": 2.3,  "avg_conceded_xg": 0.9,  "league": "Premier League"},
    "Arsenal":           {"attack": 1.70, "defense": 0.72, "avg_xg": 2.0,  "avg_conceded_xg": 1.0,  "league": "Premier League"},
    "Liverpool":         {"attack": 1.75, "defense": 0.75, "avg_xg": 2.1,  "avg_conceded_xg": 1.0,  "league": "Premier League"},
    "Chelsea":           {"attack": 1.50, "defense": 0.90, "avg_xg": 1.7,  "avg_conceded_xg": 1.2,  "league": "Premier League"},
    "Tottenham":         {"attack": 1.55, "defense": 1.05, "avg_xg": 1.8,  "avg_conceded_xg": 1.4,  "league": "Premier League"},
    "Newcastle":         {"attack": 1.45, "defense": 0.85, "avg_xg": 1.6,  "avg_conceded_xg": 1.1,  "league": "Premier League"},
    "Manchester United": {"attack": 1.35, "defense": 1.10, "avg_xg": 1.5,  "avg_conceded_xg": 1.5,  "league": "Premier League"},
    "Aston Villa":       {"attack": 1.40, "defense": 1.00, "avg_xg": 1.5,  "avg_conceded_xg": 1.3,  "league": "Premier League"},
    "West Ham":          {"attack": 1.20, "defense": 1.15, "avg_xg": 1.3,  "avg_conceded_xg": 1.5,  "league": "Premier League"},
    "Brighton":          {"attack": 1.25, "defense": 1.05, "avg_xg": 1.4,  "avg_conceded_xg": 1.3,  "league": "Premier League"},

    # Bundesliga
    "Bayern München":    {"attack": 2.00, "defense": 0.60, "avg_xg": 2.6,  "avg_conceded_xg": 0.85, "league": "Bundesliga"},
    "Borussia Dortmund": {"attack": 1.70, "defense": 0.85, "avg_xg": 2.0,  "avg_conceded_xg": 1.2,  "league": "Bundesliga"},
    "Bayer Leverkusen":  {"attack": 1.75, "defense": 0.75, "avg_xg": 2.1,  "avg_conceded_xg": 1.0,  "league": "Bundesliga"},
    "RB Leipzig":        {"attack": 1.60, "defense": 0.80, "avg_xg": 1.9,  "avg_conceded_xg": 1.05, "league": "Bundesliga"},
    "Eintracht Frankfurt":{"attack":1.40, "defense": 1.00, "avg_xg": 1.6,  "avg_conceded_xg": 1.3,  "league": "Bundesliga"},
    "VfB Stuttgart":     {"attack": 1.45, "defense": 0.95, "avg_xg": 1.65, "avg_conceded_xg": 1.2,  "league": "Bundesliga"},
    "SC Freiburg":       {"attack": 1.25, "defense": 1.00, "avg_xg": 1.4,  "avg_conceded_xg": 1.3,  "league": "Bundesliga"},
    "Borussia Mönchengladbach": {"attack": 1.25, "defense": 1.10, "avg_xg": 1.35, "avg_conceded_xg": 1.4, "league": "Bundesliga"},
    "Werder Bremen":     {"attack": 1.20, "defense": 1.10, "avg_xg": 1.3,  "avg_conceded_xg": 1.45, "league": "Bundesliga"},
    "Union Berlin":      {"attack": 1.10, "defense": 1.05, "avg_xg": 1.2,  "avg_conceded_xg": 1.35, "league": "Bundesliga"},
    "TSG Hoffenheim":    {"attack": 1.20, "defense": 1.15, "avg_xg": 1.3,  "avg_conceded_xg": 1.5,  "league": "Bundesliga"},
    "FC Augsburg":       {"attack": 1.05, "defense": 1.15, "avg_xg": 1.15, "avg_conceded_xg": 1.55, "league": "Bundesliga"},
    "VfL Wolfsburg":     {"attack": 1.15, "defense": 1.10, "avg_xg": 1.25, "avg_conceded_xg": 1.4,  "league": "Bundesliga"},
    "FSV Mainz 05":      {"attack": 1.10, "defense": 1.10, "avg_xg": 1.2,  "avg_conceded_xg": 1.4,  "league": "Bundesliga"},
    "1. FC Heidenheim":  {"attack": 1.00, "defense": 1.20, "avg_xg": 1.1,  "avg_conceded_xg": 1.6,  "league": "Bundesliga"},
    "Holstein Kiel":     {"attack": 0.95, "defense": 1.25, "avg_xg": 1.0,  "avg_conceded_xg": 1.7,  "league": "Bundesliga"},
    "VfL Bochum":        {"attack": 0.90, "defense": 1.30, "avg_xg": 0.95, "avg_conceded_xg": 1.8,  "league": "Bundesliga"},
    "FC St. Pauli":      {"attack": 0.95, "defense": 1.20, "avg_xg": 1.0,  "avg_conceded_xg": 1.65, "league": "Bundesliga"},
}

DEFAULT_PROFILE = {"attack": 1.10, "defense": 1.20, "avg_xg": 1.2, "avg_conceded_xg": 1.6, "league": "Unknown"}

# ── Player profiles ───────────────────────────────────────────────────────────

PLAYER_PROFILES = {
    "Manchester City": [
        {"name": "Erling Haaland",    "position": "FWD", "xg_per_90": 0.92, "shots_per_90": 4.2, "penalties_taken": 3},
        {"name": "Phil Foden",        "position": "MID", "xg_per_90": 0.38, "shots_per_90": 2.0, "penalties_taken": 0},
        {"name": "Kevin De Bruyne",   "position": "MID", "xg_per_90": 0.25, "shots_per_90": 1.5, "penalties_taken": 0},
        {"name": "Bernardo Silva",    "position": "MID", "xg_per_90": 0.18, "shots_per_90": 1.2, "penalties_taken": 0},
        {"name": "Rodri",             "position": "MID", "xg_per_90": 0.08, "shots_per_90": 0.5, "penalties_taken": 0},
    ],
    "Arsenal": [
        {"name": "Bukayo Saka",       "position": "FWD", "xg_per_90": 0.45, "shots_per_90": 2.5, "penalties_taken": 2},
        {"name": "Gabriel Martinelli","position": "FWD", "xg_per_90": 0.38, "shots_per_90": 2.2, "penalties_taken": 0},
        {"name": "Leandro Trossard",  "position": "FWD", "xg_per_90": 0.35, "shots_per_90": 2.0, "penalties_taken": 0},
        {"name": "Martin Odegaard",   "position": "MID", "xg_per_90": 0.28, "shots_per_90": 1.8, "penalties_taken": 0},
        {"name": "Gabriel Jesus",     "position": "FWD", "xg_per_90": 0.40, "shots_per_90": 2.3, "penalties_taken": 0},
    ],
    "Liverpool": [
        {"name": "Mohamed Salah",     "position": "FWD", "xg_per_90": 0.62, "shots_per_90": 3.1, "penalties_taken": 2},
        {"name": "Darwin Nunez",      "position": "FWD", "xg_per_90": 0.55, "shots_per_90": 2.8, "penalties_taken": 0},
        {"name": "Diogo Jota",        "position": "FWD", "xg_per_90": 0.48, "shots_per_90": 2.4, "penalties_taken": 0},
        {"name": "Luis Diaz",         "position": "FWD", "xg_per_90": 0.35, "shots_per_90": 2.0, "penalties_taken": 0},
        {"name": "Alexis Mac Allister","position": "MID", "xg_per_90": 0.18, "shots_per_90": 1.2, "penalties_taken": 0},
    ],
    # Bundesliga
    "Bayern München": [
        {"name": "Harry Kane",        "position": "FWD", "xg_per_90": 0.88, "shots_per_90": 4.0, "penalties_taken": 4},
        {"name": "Jamal Musiala",     "position": "MID", "xg_per_90": 0.42, "shots_per_90": 2.3, "penalties_taken": 0},
        {"name": "Leroy Sané",        "position": "FWD", "xg_per_90": 0.35, "shots_per_90": 2.0, "penalties_taken": 0},
        {"name": "Thomas Müller",     "position": "FWD", "xg_per_90": 0.28, "shots_per_90": 1.8, "penalties_taken": 0},
        {"name": "Serge Gnabry",      "position": "FWD", "xg_per_90": 0.32, "shots_per_90": 1.9, "penalties_taken": 0},
    ],
    "Borussia Dortmund": [
        {"name": "Serhou Guirassy",   "position": "FWD", "xg_per_90": 0.72, "shots_per_90": 3.5, "penalties_taken": 3},
        {"name": "Julian Brandt",     "position": "MID", "xg_per_90": 0.30, "shots_per_90": 1.8, "penalties_taken": 0},
        {"name": "Karim Adeyemi",     "position": "FWD", "xg_per_90": 0.38, "shots_per_90": 2.1, "penalties_taken": 0},
        {"name": "Donyell Malen",     "position": "FWD", "xg_per_90": 0.33, "shots_per_90": 1.9, "penalties_taken": 0},
        {"name": "Pascal Groß",       "position": "MID", "xg_per_90": 0.18, "shots_per_90": 1.1, "penalties_taken": 0},
    ],
    "Bayer Leverkusen": [
        {"name": "Florian Wirtz",     "position": "MID", "xg_per_90": 0.45, "shots_per_90": 2.5, "penalties_taken": 1},
        {"name": "Victor Boniface",   "position": "FWD", "xg_per_90": 0.55, "shots_per_90": 2.8, "penalties_taken": 2},
        {"name": "Granit Xhaka",      "position": "MID", "xg_per_90": 0.12, "shots_per_90": 0.9, "penalties_taken": 0},
        {"name": "Jonas Hofmann",     "position": "MID", "xg_per_90": 0.22, "shots_per_90": 1.4, "penalties_taken": 0},
        {"name": "Álex Grimaldo",     "position": "DEF", "xg_per_90": 0.15, "shots_per_90": 1.0, "penalties_taken": 0},
    ],
    "RB Leipzig": [
        {"name": "Lois Openda",       "position": "FWD", "xg_per_90": 0.65, "shots_per_90": 3.2, "penalties_taken": 2},
        {"name": "Benjamin Sesko",    "position": "FWD", "xg_per_90": 0.52, "shots_per_90": 2.6, "penalties_taken": 0},
        {"name": "Xavi Simons",       "position": "MID", "xg_per_90": 0.30, "shots_per_90": 1.8, "penalties_taken": 0},
        {"name": "Christoph Baumgartner", "position": "MID", "xg_per_90": 0.22, "shots_per_90": 1.4, "penalties_taken": 0},
        {"name": "Willi Orban",       "position": "DEF", "xg_per_90": 0.10, "shots_per_90": 0.6, "penalties_taken": 0},
    ],
    "Eintracht Frankfurt": [
        {"name": "Omar Marmoush",     "position": "FWD", "xg_per_90": 0.60, "shots_per_90": 3.0, "penalties_taken": 3},
        {"name": "Hugo Ekitiké",      "position": "FWD", "xg_per_90": 0.40, "shots_per_90": 2.1, "penalties_taken": 0},
        {"name": "Fares Chaibi",      "position": "MID", "xg_per_90": 0.20, "shots_per_90": 1.3, "penalties_taken": 0},
        {"name": "Junior Dina Ebimbe","position": "MID", "xg_per_90": 0.18, "shots_per_90": 1.1, "penalties_taken": 0},
        {"name": "Rasmus Kristensen", "position": "DEF", "xg_per_90": 0.08, "shots_per_90": 0.5, "penalties_taken": 0},
    ],
}

DEFAULT_PLAYERS = [
    {"name": "Stürmer A",  "position": "FWD", "xg_per_90": 0.30, "shots_per_90": 1.8, "penalties_taken": 0},
    {"name": "Stürmer B",  "position": "FWD", "xg_per_90": 0.25, "shots_per_90": 1.5, "penalties_taken": 0},
    {"name": "Mittelfeld A","position": "MID", "xg_per_90": 0.15, "shots_per_90": 1.0, "penalties_taken": 0},
    {"name": "Mittelfeld B","position": "MID", "xg_per_90": 0.10, "shots_per_90": 0.8, "penalties_taken": 0},
    {"name": "Mittelfeld C","position": "MID", "xg_per_90": 0.08, "shots_per_90": 0.6, "penalties_taken": 0},
]

# ── Bundesliga 2024/25 fixture schedule (mock) ────────────────────────────────
# Used by /today endpoint when football-data.org API is unavailable

BUNDESLIGA_TEAMS = [t for t, p in TEAM_PROFILES.items() if p["league"] == "Bundesliga"]

SAMPLE_FIXTURES = [
    ("Bayern München",    "Borussia Dortmund"),
    ("Bayer Leverkusen",  "RB Leipzig"),
    ("Eintracht Frankfurt", "VfB Stuttgart"),
    ("SC Freiburg",       "Werder Bremen"),
    ("Borussia Mönchengladbach", "Union Berlin"),
    ("VfL Wolfsburg",     "TSG Hoffenheim"),
    ("FSV Mainz 05",      "FC Augsburg"),
    ("FC St. Pauli",      "Holstein Kiel"),
    ("VfL Bochum",        "1. FC Heidenheim"),
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
            "league": p.get("league", "Unknown"),
            "attack_strength": p["attack"],
            "defense_weakness": p["defense"],
            "avg_goals_scored":   p["avg_xg"]          * self._rng.uniform(0.9, 1.1),
            "avg_goals_conceded": p["avg_conceded_xg"]  * self._rng.uniform(0.9, 1.1),
            "avg_xg_scored":      p["avg_xg"],
            "avg_xg_conceded":    p["avg_conceded_xg"],
            "avg_possession":     self._rng.uniform(45, 65),
            "avg_shots":          self._rng.uniform(10, 20),
            "avg_shots_on_target":self._rng.uniform(3, 8),
            "form_points":        self._rng.uniform(5, 15),
        }

    def get_player_stats(self, team_name: str) -> List[Dict[str, Any]]:
        players = PLAYER_PROFILES.get(team_name, DEFAULT_PLAYERS)
        result = []
        for i, p in enumerate(players):
            minutes = self._rng.randint(1200, 3000)
            result.append({
                "id": abs(hash(f"{team_name}_{p['name']}")) % 100000,
                "name": p["name"],
                "team": team_name,
                "position": p["position"],
                "xg_per_90": p["xg_per_90"],
                "shots_per_90": p["shots_per_90"],
                "goals": int(p["xg_per_90"] * minutes / 90 * self._rng.uniform(0.7, 1.3)),
                "minutes_played": minutes,
                "is_injured":   self._rng.random() < 0.05,
                "is_suspended": self._rng.random() < 0.03,
                "penalties_taken":  p.get("penalties_taken", 0),
                "penalties_scored": int(p.get("penalties_taken", 0) * 0.78),
            })
        return result

    def get_historical_matches(self, team_name: str, last_n: int = 10) -> List[Dict[str, Any]]:
        p = self._profile(team_name)
        matches = []
        base_date = datetime(2024, 8, 1)
        for i in range(last_n):
            home = self._rng.random() > 0.5
            hg = max(0, int(self._rng.gauss(p["avg_xg"] * (1.1 if home else 0.9), 1.0)))
            ag = max(0, int(self._rng.gauss(p["avg_conceded_xg"], 1.0)))
            matches.append({
                "date": (base_date + timedelta(weeks=i)).isoformat(),
                "home_team": team_name if home else "Opponent",
                "away_team": "Opponent" if home else team_name,
                "home_goals": hg,
                "away_goals": ag,
                "home_xg": round(self._rng.uniform(0.5, 3.0), 2),
                "away_xg": round(self._rng.uniform(0.5, 2.5), 2),
                "home_possession": round(self._rng.uniform(40, 65), 1),
                "home_shots": self._rng.randint(8, 22),
                "away_shots": self._rng.randint(5, 18),
                "competition": p.get("league", "Unknown"),
            })
        return matches

    def get_head_to_head(self, home_team: str, away_team: str, last_n: int = 10) -> List[Dict[str, Any]]:
        hp = self._profile(home_team)
        ap = self._profile(away_team)
        matches = []
        base_date = datetime(2020, 1, 1)
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
        return []

    def get_lineups(self, match_id: int) -> Dict[str, Any]:
        return {"home": [], "away": []}

    def get_todays_matches(self, league: str = "Bundesliga") -> List[Dict[str, Any]]:
        """Return mock fixtures for today (used as fallback without API key)."""
        from datetime import date
        today = date.today().isoformat()
        fixtures = []
        for i, (home, away) in enumerate(SAMPLE_FIXTURES):
            if TEAM_PROFILES.get(home, {}).get("league") == league:
                kickoff_hour = 15 + (i % 3) * 2  # 15:30 / 17:30 / 19:30
                fixtures.append({
                    "match_id": f"mock_{i}",
                    "home_team": home,
                    "away_team": away,
                    "kickoff": f"{today}T{kickoff_hour:02d}:30:00",
                    "competition": league,
                    "status": "scheduled",
                    "source": "mock",
                })
        return fixtures
