"""
Logistic Regression model for win / draw / loss classification.

Uses scikit-learn's LogisticRegression with synthetic training data
derived from the team profiles so it's functional out-of-the-box.
The model is retrained by the self-learning system when real data accumulates.
"""
import logging
import os
import pickle
from typing import Dict, Any, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.config import settings

logger = logging.getLogger(__name__)

MODEL_FILE = os.path.join(settings.MODEL_PATH, "logistic_model.pkl")
SCALER_FILE = os.path.join(settings.MODEL_PATH, "logistic_scaler.pkl")


def _generate_synthetic_training_data(n: int = 2000):
    """
    Generates synthetic (features, label) pairs using Poisson simulation
    so the logistic model has reasonable priors without real match history.
    """
    rng = np.random.default_rng(42)
    X, y = [], []

    for _ in range(n):
        atk_h = rng.uniform(0.8, 2.0)
        def_h = rng.uniform(0.6, 1.4)
        atk_a = rng.uniform(0.8, 2.0)
        def_a = rng.uniform(0.6, 1.4)
        form_h = rng.uniform(0.3, 1.0)
        form_a = rng.uniform(0.3, 1.0)

        lam_h = atk_h * def_a * 1.25
        lam_a = atk_a * def_h

        hg = rng.poisson(lam_h)
        ag = rng.poisson(lam_a)
        label = 0 if hg > ag else (1 if hg == ag else 2)  # H / D / A

        row = [
            atk_h, def_h, atk_a, def_a,
            lam_h, lam_a,
            form_h, form_a,
        ]
        X.append(row)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y)


FEATURE_KEYS: List[str] = [
    "home_attack", "home_defense", "away_attack", "away_defense",
    "lambda_home", "lambda_away",
    "home_form", "away_form",
]


class LogisticModel:
    """Softmax logistic regression: outputs P(H), P(D), P(A)."""

    def __init__(self):
        self._model = None
        self._scaler = None
        self._load_or_train()

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        x = self._extract(features)
        x_scaled = self._scaler.transform(x.reshape(1, -1))
        probs = self._model.predict_proba(x_scaled)[0]

        # Map class indices to H/D/A (order from training)
        classes = self._model.classes_
        prob_map = dict(zip(classes.tolist(), probs.tolist()))

        return {
            "model": "logistic",
            "home_win": round(prob_map.get(0, 0.0), 4),
            "draw":     round(prob_map.get(1, 0.0), 4),
            "away_win": round(prob_map.get(2, 0.0), 4),
        }

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
        )
        self._model.fit(X_scaled, y)
        self._save()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _extract(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([features.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32)

    def _load_or_train(self):
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            try:
                with open(MODEL_FILE, "rb") as f:
                    self._model = pickle.load(f)
                with open(SCALER_FILE, "rb") as f:
                    self._scaler = pickle.load(f)
                logger.info("Logistic model loaded from disk.")
                return
            except Exception as e:
                logger.warning(f"Could not load logistic model: {e}")

        logger.info("Training logistic model on synthetic data...")
        X, y = _generate_synthetic_training_data()
        self.fit(X, y)

    def _save(self):
        os.makedirs(settings.MODEL_PATH, exist_ok=True)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(self._model, f)
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(self._scaler, f)
