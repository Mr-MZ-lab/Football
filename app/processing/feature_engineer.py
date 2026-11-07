"""
Feature engineering pipeline.

Converts raw team/player data and historical match records into a flat
feature vector used by all ML models.
"""
import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

HOME_ADVANTAGE = 0.25  # empirically observed extra goals scored at home


class FeatureEngineer:
    """
    Builds a feature dict from team data fetched by the ingestion layer.
    All values are normalised to roughly [0, 3] to aid gradient-based models.
    """

    def build_prematch_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Main entry point: takes the dict returned by data_loader.get_team_features()
        and returns a flat feature dict ready for ML models.
        """
        home_stats = data["home_stats"]
        away_stats = data["away_stats"]
        home_history = data.get("home_history", [])
        away_history = data.get("away_history", [])
        h2h = data.get("h2h", [])

        features = {}

        # ── Attack / Defence parameters (Dixon-Coles style) ───────────────────
        features["home_attack"] = home_stats.get("attack_strength", 1.0)
        features["away_attack"] = away_stats.get("attack_strength", 1.0)
        features["home_defense"] = home_stats.get("defense_weakness", 1.0)
        features["away_defense"] = away_stats.get("defense_weakness", 1.0)

        # Poisson goal rate: lambda = attack * opponent_defense * home_advantage
        features["lambda_home"] = (
            home_stats.get("attack_strength", 1.0)
            * away_stats.get("defense_weakness", 1.0)
            * (1 + HOME_ADVANTAGE)
        )
        features["lambda_away"] = (
            away_stats.get("attack_strength", 1.0)
            * home_stats.get("defense_weakness", 1.0)
        )

        # ── Rolling averages from last-N match history ────────────────────────
        features.update(self._rolling_stats("home", home_history))
        features.update(self._rolling_stats("away", away_history))

        # ── xG features ───────────────────────────────────────────────────────
        features["home_avg_xg"] = home_stats.get("avg_xg_scored", 1.2)
        features["away_avg_xg"] = away_stats.get("avg_xg_scored", 1.2)
        features["home_avg_xg_conceded"] = home_stats.get("avg_xg_conceded", 1.2)
        features["away_avg_xg_conceded"] = away_stats.get("avg_xg_conceded", 1.2)

        # ── Possession / shots ────────────────────────────────────────────────
        features["home_avg_possession"] = home_stats.get("avg_possession", 50.0) / 100.0
        features["away_avg_possession"] = away_stats.get("avg_possession", 50.0) / 100.0
        features["home_avg_shots"] = home_stats.get("avg_shots", 12.0) / 20.0
        features["away_avg_shots"] = away_stats.get("avg_shots", 12.0) / 20.0

        # ── Form ──────────────────────────────────────────────────────────────
        features["home_form"] = home_stats.get("form_points", 7.5) / 15.0
        features["away_form"] = away_stats.get("form_points", 7.5) / 15.0

        # ── Head-to-head ──────────────────────────────────────────────────────
        if h2h:
            home_wins = sum(1 for m in h2h if m["home_goals"] > m["away_goals"])
            draws = sum(1 for m in h2h if m["home_goals"] == m["away_goals"])
            features["h2h_home_win_rate"] = home_wins / len(h2h)
            features["h2h_draw_rate"] = draws / len(h2h)
            features["h2h_avg_goals"] = np.mean(
                [m["home_goals"] + m["away_goals"] for m in h2h]
            )
        else:
            features["h2h_home_win_rate"] = 0.45
            features["h2h_draw_rate"] = 0.27
            features["h2h_avg_goals"] = 2.5

        return features

    def build_live_features(
        self,
        prematch_features: Dict[str, float],
        live_state: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Augments pre-match features with live match state for in-play models.
        """
        features = prematch_features.copy()
        minute = live_state.get("current_minute", 0)
        remaining = max(1, 90 - minute)

        features["current_minute"] = minute / 90.0
        features["remaining_minutes"] = remaining / 90.0
        features["score_diff"] = (
            live_state.get("home_goals", 0) - live_state.get("away_goals", 0)
        )
        features["total_goals_so_far"] = (
            live_state.get("home_goals", 0) + live_state.get("away_goals", 0)
        )
        features["home_red_cards"] = live_state.get("home_red_cards", 0)
        features["away_red_cards"] = live_state.get("away_red_cards", 0)

        # Shot intensity in current match (normalised per 90 pace)
        elapsed = max(1, minute)
        features["home_shot_pace"] = live_state.get("home_shots", 0) / elapsed * 90 / 20.0
        features["away_shot_pace"] = live_state.get("away_shots", 0) / elapsed * 90 / 20.0
        features["home_sot_pace"] = live_state.get("home_shots_on_target", 0) / elapsed * 90 / 8.0
        features["away_sot_pace"] = live_state.get("away_shots_on_target", 0) / elapsed * 90 / 8.0
        features["home_possession_live"] = live_state.get("home_possession", 50.0) / 100.0

        return features

    # ── helpers ───────────────────────────────────────────────────────────────

    def _rolling_stats(self, prefix: str, history: List[Dict], window: int = 5) -> Dict[str, float]:
        """Compute rolling-window averages from match history."""
        recent = history[-window:] if len(history) >= window else history
        if not recent:
            return {
                f"{prefix}_rolling_goals_scored": 1.2,
                f"{prefix}_rolling_goals_conceded": 1.2,
                f"{prefix}_rolling_xg_scored": 1.2,
                f"{prefix}_rolling_xg_conceded": 1.2,
                f"{prefix}_rolling_clean_sheets": 0.2,
            }

        is_home_team = [m["home_team"] != "Opponent" for m in recent]

        goals_scored = [
            m["home_goals"] if h else m["away_goals"]
            for m, h in zip(recent, is_home_team)
        ]
        goals_conceded = [
            m["away_goals"] if h else m["home_goals"]
            for m, h in zip(recent, is_home_team)
        ]
        xg_scored = [m.get("home_xg", 1.2) if h else m.get("away_xg", 1.2)
                     for m, h in zip(recent, is_home_team)]
        xg_conceded = [m.get("away_xg", 1.2) if h else m.get("home_xg", 1.2)
                       for m, h in zip(recent, is_home_team)]

        return {
            f"{prefix}_rolling_goals_scored":   float(np.mean(goals_scored)),
            f"{prefix}_rolling_goals_conceded":  float(np.mean(goals_conceded)),
            f"{prefix}_rolling_xg_scored":       float(np.mean(xg_scored)),
            f"{prefix}_rolling_xg_conceded":     float(np.mean(xg_conceded)),
            f"{prefix}_rolling_clean_sheets":    float(np.mean([g == 0 for g in goals_conceded])),
        }

    def features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to a fixed-order numpy array for ML models."""
        keys = sorted(features.keys())
        return np.array([features[k] for k in keys], dtype=np.float32)

    def feature_names(self, features: Dict[str, float]) -> List[str]:
        return sorted(features.keys())
