from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Database
    DATABASE_URL: str = "postgresql://football_user:football_pass@localhost:5432/football_db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # External API keys (optional)
    SOFASCORE_API_KEY: str = ""
    OPTA_API_KEY: str = ""

    # Model paths and training config
    MODEL_PATH: str = "./saved_models"
    MIN_TRAINING_SAMPLES: int = 50
    RETRAIN_INTERVAL_HOURS: int = 24

    # Feature engineering
    ROLLING_WINDOW: int = 5  # last N matches for rolling averages

    model_config = {"env_file": ".env"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
