"""
Probability Calibration — Platt Scaling.

Raw model outputs can be over- or under-confident. Calibration maps
raw probabilities to better-aligned empirical probabilities using
a monotonic transformation fitted on historical predictions.

Platt Scaling: fits a logistic regression on raw scores → empirical probs.
Isotonic Regression: non-parametric, more flexible but needs more data.

For the MVP, we use a lightweight beta-calibration that works well even
with small datasets (< 500 labelled predictions).
"""
import logging
import os
import pickle
from typing import Dict, Any, Tuple
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

CALIB_FILE = os.path.join(settings.MODEL_PATH, "calibration.pkl")


class BetaCalibrator:
    """
    Beta calibration for multi-class probability outputs.

    Fits a per-class beta distribution mapping:
      p_calibrated = sigmoid(a * log(p) - b * log(1-p) + c)

    Falls back to identity (no calibration) when not enough data.
    """

    def __init__(self):
        # Parameters: (a, b, c) per class — initialised as identity
        self._params: Dict[str, Tuple[float, float, float]] = {
            "home_win": (1.0, 1.0, 0.0),
            "draw":     (1.0, 1.0, 0.0),
            "away_win": (1.0, 1.0, 0.0),
        }
        self._fitted = False
        self._load()

    def calibrate(self, probs: Dict[str, float]) -> Dict[str, float]:
        """
        Apply calibration to raw model probabilities.
        Returns re-normalised calibrated probabilities.
        """
        if not self._fitted:
            return probs  # identity when not calibrated

        calibrated = {}
        for key in ["home_win", "draw", "away_win"]:
            p = max(1e-6, min(1 - 1e-6, probs.get(key, 1/3)))
            a, b, c = self._params[key]
            log_odds = a * np.log(p) - b * np.log(1 - p) + c
            calibrated[key] = float(1 / (1 + np.exp(-log_odds)))

        # Renormalise
        total = sum(calibrated.values())
        if total > 0:
            calibrated = {k: round(v / total, 4) for k, v in calibrated.items()}

        return calibrated

    def fit(self, raw_probs: np.ndarray, labels: np.ndarray):
        """
        Fit calibrator on (raw_probs, true_labels).

        raw_probs: shape (N, 3) — [p_home, p_draw, p_away]
        labels: shape (N,) — 0=H, 1=D, 2=A
        """
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression

            if len(raw_probs) < 50:
                logger.info("Not enough data for calibration (need ≥ 50 samples).")
                return

            # Fit per-class Platt scaling
            for i, key in enumerate(["home_win", "draw", "away_win"]):
                y_binary = (labels == i).astype(int)
                p_raw = raw_probs[:, i].reshape(-1, 1)

                from sklearn.linear_model import LogisticRegression as LR
                platt = LR(solver="lbfgs", max_iter=500)
                # Features: [log(p), log(1-p)]
                eps = 1e-6
                X = np.column_stack([
                    np.log(np.clip(p_raw, eps, 1 - eps)),
                    np.log(np.clip(1 - p_raw, eps, 1 - eps)),
                ])
                platt.fit(X, y_binary)
                a = float(platt.coef_[0][0])
                b = -float(platt.coef_[0][1])
                c = float(platt.intercept_[0])
                self._params[key] = (a, b, c)

            self._fitted = True
            self._save()
            logger.info("Calibrator fitted and saved.")
        except Exception as e:
            logger.warning(f"Calibration fitting failed: {e}")

    def _load(self):
        if os.path.exists(CALIB_FILE):
            try:
                with open(CALIB_FILE, "rb") as f:
                    data = pickle.load(f)
                self._params = data["params"]
                self._fitted = data["fitted"]
            except Exception as e:
                logger.warning(f"Could not load calibrator: {e}")

    def _save(self):
        os.makedirs(settings.MODEL_PATH, exist_ok=True)
        with open(CALIB_FILE, "wb") as f:
            pickle.dump({"params": self._params, "fitted": self._fitted}, f)
