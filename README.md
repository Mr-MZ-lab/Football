# Football — AI Betting Brain 🧠⚽
پیشبینی بازی های فوتبال

A production-grade, modular, self-learning football prediction platform. Combines multiple ML models into a unified ensemble to predict match outcomes, expected goals, late-goal probabilities, and per-player scoring chances.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI REST API                      │
│   /predict   /live   /player-probability                 │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                  Ensemble Engine                         │
│   Weighted average · Dynamic weight updates              │
└───┬──────────┬────────────┬──────────────┬──────────────┘
    │          │            │              │
┌───▼──┐  ┌───▼──┐  ┌──────▼───┐  ┌──────▼───┐
│Poisson│  │Logis-│  │ XGBoost  │  │ Live     │
│Model  │  │tic   │  │          │  │ Models   │
│(DC)   │  │Regr. │  │          │  │LSTM·MC·B │
└───────┘  └──────┘  └──────────┘  └──────────┘
    │                                     │
┌───▼─────────────────────────────────────▼───────────────┐
│              Feature Engineering Pipeline                │
│  Rolling averages · xG · Possession · Form · H2H         │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              Data Ingestion Layer                        │
│   MockProvider (default) · Opta/SofaScore (optional)     │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Database | PostgreSQL (SQLAlchemy ORM) |
| Cache | Redis |
| Pre-match ML | Poisson (Dixon-Coles), Logistic Regression, XGBoost |
| Live ML | LSTM (PyTorch), Markov Chain (Monte Carlo), Bayesian Updater |
| Data | Pandas, NumPy, SciPy |
| Self-learning | Brier score tracking + periodic retraining |

---

## Quick Start (Docker)

```bash
# 1. Clone and configure
cp .env.example .env

# 2. Start all services
docker-compose up --build

# 3. Open API docs
open http://localhost:8000/docs
```

---

## Quick Start (Local — no Docker)

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# The app works fully without PostgreSQL/Redis using mock data

# 4. Create model directory
mkdir -p saved_models

# 5. Run
uvicorn app.main:app --reload --port 8000

# 6. Open docs
open http://localhost:8000/docs
```

---

## API Reference

### `POST /api/v1/predict` — Pre-match prediction

```json
// Request
{
  "home_team": "Manchester City",
  "away_team": "Liverpool",
  "competition": "Premier League",
  "include_player_probs": true,
  "bookmaker_odds": { "home": 2.1, "draw": 3.4, "away": 3.8 }
}

// Response
{
  "match": "Manchester City vs Liverpool",
  "win_probability": { "home": 0.512, "draw": 0.271, "away": 0.217 },
  "expected_goals": { "home": 1.87, "away": 1.24, "total": 3.11 },
  "late_goal_probability": {
    "after_75": 0.634,
    "after_90": 0.289,
    "home_late_goal": 0.391,
    "away_late_goal": 0.301
  },
  "player_scoring_probabilities": [...],
  "confidence_score": 0.812,
  "value_bets": [
    {
      "market": "home",
      "model_probability": 0.512,
      "bookmaker_probability": 0.476,
      "edge": 0.036,
      "bookmaker_odds": 2.1
    }
  ]
}
```

### `POST /api/v1/live` — Live in-play prediction

```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "current_minute": 65,
  "home_goals": 1,
  "away_goals": 0,
  "home_red_cards": 0,
  "away_red_cards": 1,
  "home_shots": 12,
  "away_shots": 5,
  "home_shots_on_target": 5,
  "away_shots_on_target": 2,
  "home_possession": 61.0
}
```

### `POST /api/v1/player-probability` — Per-player scoring probability

```json
// Request
{ "home_team": "Manchester City", "away_team": "Arsenal" }

// Response
{
  "home_players": [
    {
      "player_name": "Erling Haaland",
      "scoring_probability": 0.712,
      "xg_contribution": 1.24,
      "is_penalty_taker": true
    }
  ]
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Module Structure

```
app/
├── main.py                    # FastAPI entry point
├── config.py                  # Settings (pydantic-settings)
├── database.py                # SQLAlchemy engine + session
├── models/                    # ORM models
│   ├── team.py
│   ├── player.py
│   ├── match.py
│   └── prediction.py
├── schemas/                   # Pydantic request/response schemas
├── api/                       # Route handlers
│   ├── predict.py             → POST /predict
│   ├── live.py                → POST /live
│   └── player.py              → POST /player-probability
├── ingestion/                 # Data providers
│   ├── base.py                # Abstract provider interface
│   ├── mock_provider.py       # Realistic mock data (no API key needed)
│   └── data_loader.py         # Provider → feature dict
├── processing/
│   └── feature_engineer.py    # Feature vector construction
├── models_ml/                 # ML model implementations
│   ├── poisson_model.py       # Dixon-Coles Poisson
│   ├── logistic_model.py      # Multinomial logistic regression
│   ├── xgboost_model.py       # XGBoost gradient boosting
│   ├── lstm_model.py          # PyTorch LSTM (live momentum)
│   ├── markov_chain.py        # Monte Carlo Markov simulation
│   ├── bayesian_updater.py    # Bayesian goal-rate update
│   ├── player_model.py        # xG-based player scoring probability
│   └── late_goal_predictor.py # Hazard function — late goals
├── ensemble/
│   └── ensemble.py            # Weighted average + dynamic weights
├── realtime/
│   └── live_engine.py         # Live prediction orchestrator
└── self_learning/
    └── trainer.py             # Brier score tracking + retraining
```

---

## Self-Learning System

After each completed match, call the trainer to record actual results
and update ensemble weights:

```python
from app.self_learning.trainer import SelfLearningTrainer
trainer = SelfLearningTrainer()
trainer.record_outcome(prediction_id=42, actual_home_goals=2, actual_away_goals=1)
```

---

## Extending with Real Data

Replace `MockDataProvider` with a real API provider:

```python
# app/ingestion/my_real_provider.py
from app.ingestion.base import BaseDataProvider

class SofaScoreProvider(BaseDataProvider):
    def get_team_stats(self, team_name):
        # call SofaScore API ...
```

Then update `app/ingestion/data_loader.py` to use the new provider.

---

## Suggested Improvements

1. **Calibration**: Platt scaling / isotonic regression on raw model outputs
2. **Over-Under / Asian Handicap markets**: Extend value-bet detection
3. **WebSocket streaming**: Real-time live event push to clients
4. **ELO ratings**: Parallel team-strength model alongside Dixon-Coles
5. **Weather & venue data**: Additional XGBoost features
6. **SHAP explanations**: Feature importance per prediction via `/explain`
7. **Kubernetes deployment**: Helm chart for production scaling
