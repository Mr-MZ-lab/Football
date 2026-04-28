"""
AI Betting Brain — FastAPI application entry point.

Modular football prediction platform combining Poisson models, XGBoost,
LSTM time-series, Markov chains, and Bayesian updating into a unified ensemble.
Covers all major betting markets for German bookmakers (Tipico, Lotto, Bwin).
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("Starting AI Betting Brain...")
    os.makedirs(settings.MODEL_PATH, exist_ok=True)

    try:
        from app.database import create_all_tables
        create_all_tables()
        logger.info("Database tables ready.")
    except Exception as e:
        logger.warning(f"DB init skipped (no DB connection?): {e}")

    try:
        from app.ingestion.data_loader import seed_mock_data
        seed_mock_data()
        logger.info("Mock data seeded.")
    except Exception as e:
        logger.warning(f"Mock data seed skipped: {e}")

    yield

    logger.info("AI Betting Brain shutting down.")


app = FastAPI(
    title="AI Betting Brain",
    description=(
        "Intelligent, modular, self-learning football prediction platform. "
        "Covers 1X2, Over/Under, BTTS, Double Chance, Correct Score, "
        "Asian Handicap and Kelly Criterion stake sizing. "
        "Integrates with football-data.org and The Odds API for real data."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────────────
from app.api.predict  import router as predict_router
from app.api.live     import router as live_router
from app.api.player   import router as player_router
from app.api.today    import router as today_router
from app.api.markets  import router as markets_router

app.include_router(predict_router,  prefix="/api/v1", tags=["Pre-Match Prediction"])
app.include_router(live_router,     prefix="/api/v1", tags=["Live Prediction"])
app.include_router(player_router,   prefix="/api/v1", tags=["Player Probability"])
app.include_router(today_router,    prefix="/api/v1", tags=["Today's Matches"])
app.include_router(markets_router,  prefix="/api/v1", tags=["Betting Markets"])


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "AI Betting Brain",
        "version": "2.0.0",
        "endpoints": [
            "GET  /api/v1/today              — today's matches with predictions",
            "POST /api/v1/predict            — pre-match prediction (1X2)",
            "POST /api/v1/markets            — all markets (O/U, BTTS, AH, Kelly)",
            "POST /api/v1/live              — live in-play prediction",
            "POST /api/v1/player-probability — per-player scoring probability",
        ],
        "data_sources": {
            "schedule": "football-data.org (configure FOOTBALL_DATA_API_KEY)",
            "odds":     "the-odds-api.com  (configure ODDS_API_KEY)",
        },
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
