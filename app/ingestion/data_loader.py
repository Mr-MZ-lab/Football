"""
Loads data from a provider and returns feature dicts for the ML pipeline.

Provider selection:
  - FOOTBALL_DATA_API_KEY set → FootballDataProvider (real API)
  - Otherwise               → MockDataProvider

Also handles one-time mock data seeding into PostgreSQL.
"""
import logging
from app.config import settings

logger = logging.getLogger(__name__)


def _get_provider():
    """Return the best available data provider."""
    if settings.FOOTBALL_DATA_API_KEY:
        try:
            from app.ingestion.football_data_provider import FootballDataProvider
            return FootballDataProvider(settings.FOOTBALL_DATA_API_KEY)
        except Exception as e:
            logger.warning(f"FootballDataProvider init failed: {e}. Falling back to mock.")
    from app.ingestion.mock_provider import MockDataProvider
    return MockDataProvider()


# Module-level singleton (created once per worker)
_provider = _get_provider()


def get_team_features(home_team: str, away_team: str) -> dict:
    """Fetch and merge features for both teams."""
    home_stats   = _provider.get_team_stats(home_team)
    away_stats   = _provider.get_team_stats(away_team)
    home_history = _provider.get_historical_matches(home_team, last_n=10)
    away_history = _provider.get_historical_matches(away_team, last_n=10)
    h2h          = _provider.get_head_to_head(home_team, away_team, last_n=10)

    return {
        "home_team":    home_team,
        "away_team":    away_team,
        "home_stats":   home_stats,
        "away_stats":   away_stats,
        "home_history": home_history,
        "away_history": away_history,
        "h2h":          h2h,
    }


def get_player_features(home_team: str, away_team: str) -> dict:
    home_players = _provider.get_player_stats(home_team)
    away_players = _provider.get_player_stats(away_team)
    return {"home_players": home_players, "away_players": away_players}


def get_todays_matches(league: str = "Bundesliga"):
    """Return today's fixtures from the configured provider."""
    if hasattr(_provider, "get_todays_matches"):
        return _provider.get_todays_matches(league)
    return []


def seed_mock_data():
    """Seed the DB with mock teams/players on first run."""
    try:
        from app.database import SessionLocal
        from app.models.team import Team
        from app.models.player import Player
        from app.ingestion.mock_provider import TEAM_PROFILES, PLAYER_PROFILES, DEFAULT_PLAYERS

        db = SessionLocal()
        try:
            if db.query(Team).count() > 0:
                logger.info("Mock data already seeded — skipping.")
                return

            for team_name, profile in TEAM_PROFILES.items():
                team = Team(
                    name=team_name,
                    short_name=team_name[:3].upper(),
                    league=profile.get("league", "Unknown"),
                    country="Germany" if profile.get("league") == "Bundesliga" else "England",
                    attack_strength=profile["attack"],
                    defense_weakness=profile["defense"],
                    avg_xg_scored=profile["avg_xg"],
                    avg_xg_conceded=profile["avg_conceded_xg"],
                )
                db.add(team)
                db.flush()

                players_data = PLAYER_PROFILES.get(team_name, DEFAULT_PLAYERS)
                for pd_item in players_data:
                    player = Player(
                        name=pd_item["name"],
                        team_id=team.id,
                        position=pd_item["position"],
                        xg_per_90=pd_item["xg_per_90"],
                        shots_per_90=pd_item["shots_per_90"],
                    )
                    db.add(player)

            db.commit()
            logger.info(f"Seeded {len(TEAM_PROFILES)} teams with players.")
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"seed_mock_data failed (DB likely unavailable): {e}")
