"""
Ensemble Engine — combines outputs from all models into a single prediction.

Strategy: weighted average of win/draw/loss probabilities.

Initial weights are set from empirical performance on Premier League data:
  - XGBoost is the strongest (weight 0.40)
  - Poisson is the most interpretable and reliable (0.35)
  - Logistic is the simplest baseline (0.25)

Weights are dynamically updated by the self-learning system using
the inverse of each model's rolling Brier score (lower Brier = higher weight).
"""
import logging
import os
import json
from typing import Dict, Any, List, Optional
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

WEIGHTS_FILE = os.path.join(settings.MODEL_PATH, "ensemble_weights.json")

# Default weights for pre-match ensemble
DEFAULT_PREMATCH_WEIGHTS = {
    "poisson": 0.35,
    "logistic": 0.25,
    "xgboost": 0.40,
}

# Default weights for live ensemble (Bayesian + Markov dominate; LSTM adds momentum)
DEFAULT_LIVE_WEIGHTS = {
    "bayesian": 0.35,
    "markov_chain": 0.35,
    "lstm": 0.15,
    "poisson": 0.15,  # pre-match prior still carries some weight early on
}


class EnsembleEngine:
    """
    Combines model outputs into a final, calibrated probability distribution.
    """

    def __init__(self):
        self._prematch_weights = DEFAULT_PREMATCH_WEIGHTS.copy()
        self._live_weights = DEFAULT_LIVE_WEIGHTS.copy()
        self._load_weights()

    # ── Pre-match ensemble ────────────────────────────────────────────────────

    def combine_prematch(
        self, model_outputs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Weighted average over Poisson, Logistic, XGBoost.
        Returns normalised home_win / draw / away_win.
        """
        return self._weighted_average(model_outputs, self._prematch_weights)

    # ── Live ensemble ─────────────────────────────────────────────────────────

    def combine_live(
        self, model_outputs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Weighted average over Bayesian, Markov, LSTM, and Poisson prior.
        """
        return self._weighted_average(model_outputs, self._live_weights)

    # ── Confidence score ──────────────────────────────────────────────────────

    def confidence_score(self, model_outputs: List[Dict[str, Any]]) -> float:
        """
        Confidence = 1 - average pairwise disagreement between models.

        High confidence when all models agree on the dominant outcome;
        low confidence when they disagree significantly.
        """
        if len(model_outputs) < 2:
            return 0.5

        vectors = [
            np.array([m["home_win"], m["draw"], m["away_win"]])
            for m in model_outputs
            if "home_win" in m
        ]
        if not vectors:
            return 0.5

        diffs = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                diff = float(np.linalg.norm(vectors[i] - vectors[j]))
                diffs.append(diff)

        avg_disagreement = np.mean(diffs)
        # Max possible L2 distance between two probability distributions ≈ sqrt(2)
        confidence = max(0.0, 1.0 - avg_disagreement / 1.414)
        return round(float(confidence), 4)

    # ── Weight update (called by self-learning system) ─────────────────────────

    def update_prematch_weights(self, brier_scores: Dict[str, float]):
        """
        Recompute weights as inverse of Brier score (better model → higher weight).
        brier_scores: {"poisson": 0.21, "logistic": 0.24, "xgboost": 0.19}
        """
        if not brier_scores:
            return

        inv = {model: 1.0 / max(0.01, score) for model, score in brier_scores.items()}
        total = sum(inv.values())
        self._prematch_weights = {m: round(v / total, 4) for m, v in inv.items()}
        self._save_weights()
        logger.info(f"Updated pre-match ensemble weights: {self._prematch_weights}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _weighted_average(
        self,
        model_outputs: List[Dict[str, Any]],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        hw = dr = aw = total_w = 0.0

        for m in model_outputs:
            model_name = m.get("model", "")
            w = weights.get(model_name, 0.1)
            hw += w * m.get("home_win", 0.0)
            dr += w * m.get("draw", 0.0)
            aw += w * m.get("away_win", 0.0)
            total_w += w

        if total_w == 0:
            return {"home_win": 0.45, "draw": 0.27, "away_win": 0.28}

        hw /= total_w
        dr /= total_w
        aw /= total_w

        # Normalise to sum = 1.0
        total = hw + dr + aw
        if total > 0:
            hw, dr, aw = hw / total, dr / total, aw / total

        return {
            "home_win": round(hw, 4),
            "draw": round(dr, 4),
            "away_win": round(aw, 4),
        }

    def _load_weights(self):
        if os.path.exists(WEIGHTS_FILE):
            try:
                with open(WEIGHTS_FILE) as f:
                    data = json.load(f)
                self._prematch_weights = data.get("prematch", DEFAULT_PREMATCH_WEIGHTS)
                self._live_weights = data.get("live", DEFAULT_LIVE_WEIGHTS)
            except Exception as e:
                logger.warning(f"Could not load ensemble weights: {e}")

    def _save_weights(self):
        os.makedirs(settings.MODEL_PATH, exist_ok=True)
        with open(WEIGHTS_FILE, "w") as f:
            json.dump({
                "prematch": self._prematch_weights,
                "live": self._live_weights,
            }, f, indent=2)
