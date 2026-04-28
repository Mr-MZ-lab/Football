"""
XGBoost gradient boosting model.

This is the primary workhorse model. It consumes the full feature vector
(all keys from FeatureEngineer) and outputs win/draw/loss probabilities.
Trained on synthetic data initially; auto-retrained by the self-learning system.
"""
import logging
import os
import pickle
from typing import Dict, Any, List
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from app.config import settings

logger = logging.getLogger(__name__)

MODEL_FILE = os.path.join(settings.MODEL_PATH, "xgboost_model.pkl")
ENCODER_FILE = os.path.join(settings.MODEL_PATH, "xgboost_encoder.pkl")


# All features consumed by XGBoost (must be stable/ordered)
FEATURE_KEYS: List[str] = [
    "home_attack", "away_attack", "home_defense", "away_defense",
    "lambda_home", "lambda_away",
    "home_avg_xg", "away_avg_xg",
    "home_avg_xg_conceded", "away_avg_xg_conceded",
    "home_avg_possession", "away_avg_possession",
    "home_avg_shots", "away_avg_shots",
    "home_form", "away_form",
    "home_rolling_goals_scored", "away_rolling_goals_scored",
    "home_rolling_goals_conceded", "away_rolling_goals_conceded",
    "home_rolling_xg_scored", "away_rolling_xg_scored",
    "home_rolling_xg_conceded", "away_rolling_xg_conceded",
    "home_rolling_clean_sheets", "away_rolling_clean_sheets",
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_avg_goals",
]


def _generate_synthetic_data(n: int = 3000):
    rng = np.random.default_rng(99)
    X, y = [], []

    for _ in range(n):
        atk_h = rng.uniform(0.7, 2.1)
        def_h = rng.uniform(0.55, 1.45)
        atk_a = rng.uniform(0.7, 2.1)
        def_a = rng.uniform(0.55, 1.45)
        lam_h = atk_h * def_a * 1.25
        lam_a = atk_a * def_h

        form_h = rng.uniform(0.2, 1.0)
        form_a = rng.uniform(0.2, 1.0)
        roll_gs_h = rng.uniform(0.5, 3.0)
        roll_gc_h = rng.uniform(0.5, 2.5)
        roll_gs_a = rng.uniform(0.5, 3.0)
        roll_gc_a = rng.uniform(0.5, 2.5)

        row = [
            atk_h, atk_a, def_h, def_a,
            lam_h, lam_a,
            rng.uniform(0.8, 2.5), rng.uniform(0.8, 2.5),  # xg scored
            rng.uniform(0.6, 2.0), rng.uniform(0.6, 2.0),  # xg conceded
            rng.uniform(0.40, 0.65), rng.uniform(0.35, 0.60),  # possession
            rng.uniform(0.4, 0.9), rng.uniform(0.3, 0.8),  # shots
            form_h, form_a,
            roll_gs_h, roll_gs_a,
            roll_gc_h, roll_gc_a,
            rng.uniform(0.5, 2.5), rng.uniform(0.5, 2.5),  # rolling xg scored
            rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0),  # rolling xg conceded
            rng.uniform(0.0, 0.6), rng.uniform(0.0, 0.6),  # clean sheets
            rng.uniform(0.2, 0.7), rng.uniform(0.15, 0.4), rng.uniform(1.5, 4.0),
        ]

        hg = rng.poisson(lam_h)
        ag = rng.poisson(lam_a)
        label = "H" if hg > ag else ("D" if hg == ag else "A")
        X.append(row)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


class XGBoostModel:
    """XGBoost multi-class classifier: P(H), P(D), P(A)."""

    def __init__(self):
        self._model = None
        self._encoder = LabelEncoder()
        self._load_or_train()

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        x = self._extract(features)
        proba = self._model.predict_proba(x.reshape(1, -1))[0]
        classes = self._encoder.classes_  # ['A', 'D', 'H'] sorted alphabetically

        prob_map = {c: float(p) for c, p in zip(classes, proba)}
        return {
            "model": "xgboost",
            "home_win": round(prob_map.get("H", 0.0), 4),
            "draw":     round(prob_map.get("D", 0.0), 4),
            "away_win": round(prob_map.get("A", 0.0), 4),
        }

    def fit(self, X: np.ndarray, y_str: np.ndarray):
        y = self._encoder.fit_transform(y_str)
        self._model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
        )
        self._model.fit(X, y, verbose=False)
        self._save()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _extract(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([features.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32)

    def _load_or_train(self):
        if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
            try:
                with open(MODEL_FILE, "rb") as f:
                    self._model = pickle.load(f)
                with open(ENCODER_FILE, "rb") as f:
                    self._encoder = pickle.load(f)
                logger.info("XGBoost model loaded from disk.")
                return
            except Exception as e:
                logger.warning(f"Could not load XGBoost model: {e}")

        logger.info("Training XGBoost model on synthetic data...")
        X, y = _generate_synthetic_data()
        self.fit(X, y)

    def _save(self):
        os.makedirs(settings.MODEL_PATH, exist_ok=True)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(self._model, f)
        with open(ENCODER_FILE, "wb") as f:
            pickle.dump(self._encoder, f)
