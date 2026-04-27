"""
AI Betting Brain — FastAPI application entry point.

Modular football prediction platform combining Poisson models, XGBoost,
LSTM time-series, Markov chains, and Bayesian updating into a unified ensemble.
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

    yield  # app is running

    # ── Shutdown (nothing to clean up) ────────────────────────────────────────
    logger.info("AI Betting Brain shutting down.")


app = FastAPI(
    title="AI Betting Brain",
    description="Intelligent, modular, self-learning football prediction platform",
    version="1.0.0",
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
from app.api.predict import router as predict_router
from app.api.live import router as live_router
from app.api.player import router as player_router

app.include_router(predict_router, prefix="/api/v1", tags=["Pre-Match Prediction"])
app.include_router(live_router, prefix="/api/v1", tags=["Live Prediction"])
app.include_router(player_router, prefix="/api/v1", tags=["Player Probability"])


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "AI Betting Brain", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
