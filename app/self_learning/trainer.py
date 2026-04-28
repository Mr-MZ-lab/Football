"""
Self-Learning System.

Responsibilities:
  1. After each match completes, update the Prediction row with actual results
  2. Compute Brier scores for each model's prediction
  3. When enough new data has accumulated (MIN_TRAINING_SAMPLES), retrain models
  4. Update ensemble weights using the latest per-model Brier scores

Brier Score (for multi-class): BS = Σ (p_i - o_i)² for i in {H, D, A}
Range: 0 (perfect) → 2 (worst)
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

BRIER_HISTORY_FILE = os.path.join(settings.MODEL_PATH, "brier_history.json")


def _brier_score(probs: Dict[str, float], actual: str) -> float:
    """
    Multi-class Brier score.
    actual: 'H', 'D', or 'A'
    probs: {"home_win": p1, "draw": p2, "away_win": p3}
    """
    outcome_map = {
        "H": [1, 0, 0],
        "D": [0, 1, 0],
        "A": [0, 0, 1],
    }
    targets = outcome_map.get(actual, [1/3, 1/3, 1/3])
    p_vec = [
        probs.get("home_win", 1/3),
        probs.get("draw", 1/3),
        probs.get("away_win", 1/3),
    ]
    return float(sum((p - t) ** 2 for p, t in zip(p_vec, targets)))


class SelfLearningTrainer:
    """
    Manages outcome tracking and periodic model retraining.
    Can be run as a background task or on-demand after match completion.
    """

    def record_outcome(
        self,
        prediction_id: int,
        actual_home_goals: int,
        actual_away_goals: int,
    ) -> Optional[float]:
        """
        Store actual outcome in DB and compute Brier score.
        Returns the Brier score or None if DB is unavailable.
        """
        actual_result = (
            "H" if actual_home_goals > actual_away_goals
            else "D" if actual_home_goals == actual_away_goals
            else "A"
        )

        try:
            from app.database import SessionLocal
            from app.models.prediction import Prediction

            db = SessionLocal()
            try:
                pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
                if not pred:
                    logger.warning(f"Prediction {prediction_id} not found.")
                    return None

                pred.actual_result = actual_result
                pred.actual_home_goals = actual_home_goals
                pred.actual_away_goals = actual_away_goals

                probs = {
                    "home_win": pred.home_win_prob,
                    "draw": pred.draw_prob,
                    "away_win": pred.away_win_prob,
                }
                brier = _brier_score(probs, actual_result)
                pred.brier_score = brier

                db.commit()
                logger.info(f"Recorded outcome for prediction {prediction_id}: {actual_result}, BS={brier:.4f}")
                return brier
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"record_outcome failed (DB unavailable?): {e}")
            return None

    def should_retrain(self) -> bool:
        """Check whether enough new outcomes have accumulated to warrant retraining."""
        try:
            from app.database import SessionLocal
            from app.models.prediction import Prediction

            db = SessionLocal()
            try:
                cutoff = datetime.utcnow() - timedelta(hours=settings.RETRAIN_INTERVAL_HOURS)
                n_new = (
                    db.query(Prediction)
                    .filter(Prediction.actual_result.isnot(None))
                    .filter(Prediction.predicted_at >= cutoff)
                    .count()
                )
                return n_new >= settings.MIN_TRAINING_SAMPLES
            finally:
                db.close()
        except Exception:
            return False

    def retrain_all(self):
        """
        Retrain Logistic and XGBoost models on all labelled predictions,
        then update ensemble weights.
        """
        rows = self._load_labelled_predictions()
        if len(rows) < settings.MIN_TRAINING_SAMPLES:
            logger.info(f"Not enough data to retrain ({len(rows)} samples).")
            return

        logger.info(f"Retraining on {len(rows)} labelled predictions...")
        self._retrain_logistic(rows)
        self._retrain_xgboost(rows)
        self._update_ensemble_weights(rows)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Return aggregated per-model performance metrics."""
        rows = self._load_labelled_predictions()
        if not rows:
            return {"message": "No labelled predictions yet."}

        brier_scores = [r["brier_score"] for r in rows if r.get("brier_score") is not None]
        results = [r["actual_result"] for r in rows if r.get("actual_result")]

        h_count = results.count("H")
        d_count = results.count("D")
        a_count = results.count("A")
        n = len(results)

        return {
            "total_predictions": n,
            "result_distribution": {
                "home_wins": h_count / n if n else 0,
                "draws": d_count / n if n else 0,
                "away_wins": a_count / n if n else 0,
            },
            "avg_brier_score": round(float(np.mean(brier_scores)), 4) if brier_scores else None,
            "min_brier_score": round(float(np.min(brier_scores)), 4) if brier_scores else None,
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_labelled_predictions(self) -> List[Dict[str, Any]]:
        try:
            from app.database import SessionLocal
            from app.models.prediction import Prediction

            db = SessionLocal()
            try:
                rows = (
                    db.query(Prediction)
                    .filter(Prediction.actual_result.isnot(None))
                    .all()
                )
                return [
                    {
                        "home_win_prob": r.home_win_prob,
                        "draw_prob": r.draw_prob,
                        "away_win_prob": r.away_win_prob,
                        "actual_result": r.actual_result,
                        "brier_score": r.brier_score,
                        "model_outputs": r.model_outputs or {},
                    }
                    for r in rows
                ]
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Could not load labelled predictions: {e}")
            return []

    def _retrain_logistic(self, rows: List[Dict[str, Any]]):
        from app.models_ml.logistic_model import LogisticModel, FEATURE_KEYS
        import numpy as np

        model = LogisticModel()
        X, y = [], []
        label_map = {"H": 0, "D": 1, "A": 2}

        for r in rows:
            outputs = r.get("model_outputs", {})
            poisson_out = outputs.get("poisson", {})
            if not poisson_out:
                continue
            feat_approx = {
                "home_attack": poisson_out.get("lambda_home", 1.4),
                "home_defense": 1.0,
                "away_attack": poisson_out.get("lambda_away", 1.1),
                "away_defense": 1.0,
                "lambda_home": poisson_out.get("lambda_home", 1.4),
                "lambda_away": poisson_out.get("lambda_away", 1.1),
                "home_form": 0.5,
                "away_form": 0.5,
            }
            X.append([feat_approx.get(k, 0.0) for k in FEATURE_KEYS])
            y.append(label_map[r["actual_result"]])

        if len(X) >= settings.MIN_TRAINING_SAMPLES:
            model.fit(np.array(X, dtype=np.float32), np.array(y))
            logger.info("Logistic model retrained.")

    def _retrain_xgboost(self, rows: List[Dict[str, Any]]):
        # Similar to logistic — use stored model_outputs as feature proxies
        logger.info("XGBoost retraining skipped (requires full feature reconstruction).")

    def _update_ensemble_weights(self, rows: List[Dict[str, Any]]):
        from app.ensemble.ensemble import EnsembleEngine

        # Compute per-model rolling Brier scores from stored model_outputs
        model_briers: Dict[str, List[float]] = {"poisson": [], "logistic": [], "xgboost": []}

        for r in rows:
            actual = r["actual_result"]
            outputs = r.get("model_outputs", {})
            for model_name in model_briers:
                m_out = outputs.get(model_name, {})
                if m_out:
                    probs = {
                        "home_win": m_out.get("home_win", 1/3),
                        "draw": m_out.get("draw", 1/3),
                        "away_win": m_out.get("away_win", 1/3),
                    }
                    model_briers[model_name].append(_brier_score(probs, actual))

        avg_briers = {
            m: float(np.mean(scores))
            for m, scores in model_briers.items()
            if scores
        }
        if avg_briers:
            EnsembleEngine().update_prematch_weights(avg_briers)
            logger.info(f"Ensemble weights updated. Avg Brier scores: {avg_briers}")
