"""
Real-Time Prediction Engine.

Orchestrates all live models (Bayesian, Markov Chain, LSTM) and re-runs
the full prediction pipeline every time a match event is received.

Key responsibilities:
  1. Accept live match state (score, minute, events)
  2. Run all live models in sequence
  3. Ensemble the results
  4. Compute momentum, late-goal probability, and value bets
  5. Return a structured LivePredictionResponse
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from app.models_ml.poisson_model import PoissonModel
from app.models_ml.markov_chain import MarkovChainModel
from app.models_ml.bayesian_updater import BayesianUpdater
from app.models_ml.lstm_model import LSTMModel
from app.models_ml.late_goal_predictor import LateGoalPredictor
from app.ensemble.ensemble import EnsembleEngine
from app.processing.feature_engineer import FeatureEngineer
from app.ingestion.data_loader import get_team_features
from app.schemas.prediction import (
    LivePredictionResponse,
    WinProbability,
    ExpectedGoals,
    LateGoalProbability,
    ValueBet,
)

logger = logging.getLogger(__name__)

# Module-level singletons (loaded once at startup, reused across requests)
_poisson = PoissonModel()
_markov = MarkovChainModel()
_bayesian = BayesianUpdater()
_lstm = LSTMModel()
_late_goal = LateGoalPredictor()
_ensemble = EnsembleEngine()
_fe = FeatureEngineer()


class LivePredictionEngine:

    def predict(self, request: Dict[str, Any]) -> LivePredictionResponse:
        home_team = request["home_team"]
        away_team = request["away_team"]
        minute = request.get("current_minute", 45)
        home_goals = request.get("home_goals", 0)
        away_goals = request.get("away_goals", 0)

        # ── Feature engineering ───────────────────────────────────────────────
        raw = get_team_features(home_team, away_team)
        prematch_features = _fe.build_prematch_features(raw)
        home_subs = request.get("home_subs", [])
        away_subs = request.get("away_subs", [])

        live_state = {
            "current_minute": minute,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_red_cards": request.get("home_red_cards", 0),
            "away_red_cards": request.get("away_red_cards", 0),
            "home_shots": request.get("home_shots", 0),
            "away_shots": request.get("away_shots", 0),
            "home_shots_on_target": request.get("home_shots_on_target", 0),
            "away_shots_on_target": request.get("away_shots_on_target", 0),
            "home_possession": request.get("home_possession", 50.0),
            "home_subs": home_subs,
            "away_subs": away_subs,
        }
        live_features = _fe.build_live_features(prematch_features, live_state)

        # ── Run models ────────────────────────────────────────────────────────
        poisson_out = _poisson.predict(prematch_features)
        bayesian_out = _bayesian.predict(live_features, live_state)
        markov_out = _markov.predict(live_features, live_state)
        lstm_out = _lstm.predict(live_features, live_state)

        # ── Ensemble ──────────────────────────────────────────────────────────
        all_outputs = [poisson_out, bayesian_out, markov_out, lstm_out]
        combined = _ensemble.combine_live(all_outputs)
        confidence = _ensemble.confidence_score(all_outputs)

        # ── Expected remaining goals ──────────────────────────────────────────
        remaining_factor = max(0.0, (90 - minute) / 90.0)
        exp_h = poisson_out["expected_home_goals"] * remaining_factor
        exp_a = poisson_out["expected_away_goals"] * remaining_factor

        # Use Markov/Bayesian estimates if available
        exp_h_rem = (
            bayesian_out.get("expected_remaining_home_goals", 0) * 0.5
            + markov_out.get("expected_remaining_home_goals", 0) * 0.5
        )
        exp_a_rem = (
            bayesian_out.get("expected_remaining_away_goals", 0) * 0.5
            + markov_out.get("expected_remaining_away_goals", 0) * 0.5
        )

        # ── Late goal probability ─────────────────────────────────────────────
        late_goals = _late_goal.predict(
            expected_home_goals=exp_h_rem,
            expected_away_goals=exp_a_rem,
            current_minute=minute,
            home_goals_so_far=home_goals,
            away_goals_so_far=away_goals,
        )

        # ── Momentum (with substitution boost) ───────────────────────────────
        momentum = self._compute_momentum(live_features, live_state)

        # ── Substitution impact ───────────────────────────────────────────────
        sub_impact = self._compute_sub_impact(home_subs, away_subs, minute)
        if sub_impact:
            # Subs boost the subbing team's expected remaining goals
            h_boost = sub_impact["home_boost"]
            a_boost = sub_impact["away_boost"]
            exp_h_rem = round(exp_h_rem * (1.0 + h_boost), 3)
            exp_a_rem = round(exp_a_rem * (1.0 + a_boost), 3)

        # ── Value bets ────────────────────────────────────────────────────────
        value_bets = []
        if odds := request.get("bookmaker_odds"):
            value_bets = _compute_value_bets(combined, odds)

        return LivePredictionResponse(
            match=f"{home_team} vs {away_team}",
            home_team=home_team,
            away_team=away_team,
            current_minute=minute,
            current_score=f"{home_goals}-{away_goals}",
            predicted_at=datetime.utcnow(),
            win_probability=WinProbability(
                home=combined["home_win"],
                draw=combined["draw"],
                away=combined["away_win"],
            ),
            expected_remaining_goals=ExpectedGoals(
                home=round(exp_h_rem, 3),
                away=round(exp_a_rem, 3),
                total=round(exp_h_rem + exp_a_rem, 3),
            ),
            late_goal_probability=LateGoalProbability(
                after_75=late_goals["after_75"],
                after_90=late_goals["after_90"],
                home_late_goal=late_goals["home_late_goal"],
                away_late_goal=late_goals["away_late_goal"],
            ),
            confidence_score=confidence,
            value_bets=value_bets,
            momentum=momentum,
            substitutions=sub_impact,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _compute_sub_impact(
        self,
        home_subs: List[str],
        away_subs: List[str],
        minute: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Estimate the impact of substitutions on remaining expected goals.

        Logic: each sub made after minute 60 injects fresh energy.
        The earlier the sub, the more minutes of impact; the later, the less
        volume but still a spike of intensity (set-piece threat, fresh press).

        Returns None if no subs, otherwise a dict with boost factors.
        """
        if not home_subs and not away_subs:
            return None

        def _boost(subs: List[str]) -> float:
            if not subs:
                return 0.0
            # Each sub contributes +4% to remaining xG, capped at 3 subs (+12%)
            n = min(len(subs), 3)
            per_sub = 0.04
            # Earlier subs are slightly more impactful (more time on pitch)
            time_factor = max(0.5, (90 - minute) / 30.0)
            return round(n * per_sub * time_factor, 4)

        h_boost = _boost(home_subs)
        a_boost = _boost(away_subs)

        return {
            "home_subs":   home_subs,
            "away_subs":   away_subs,
            "home_boost":  h_boost,
            "away_boost":  a_boost,
            "note": (
                f"Home substitutions (+{h_boost*100:.1f}% xG boost), "
                f"Away substitutions (+{a_boost*100:.1f}% xG boost)."
            ),
        }

    def _compute_momentum(
        self, live_features: Dict[str, float], live_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Simple momentum score based on recent shots, possession, and score.
        Returns {"home": 0.0–1.0, "away": 0.0–1.0} summing to 1.0.
        """
        home_shot_pace = live_features.get("home_shot_pace", 0.5)
        away_shot_pace = live_features.get("away_shot_pace", 0.4)
        possession = live_features.get("home_possession_live", 0.5)
        score_diff = live_state.get("home_goals", 0) - live_state.get("away_goals", 0)

        home_raw = home_shot_pace * 0.4 + possession * 0.4 + max(0, score_diff) * 0.2
        away_raw = away_shot_pace * 0.4 + (1 - possession) * 0.4 + max(0, -score_diff) * 0.2

        total = home_raw + away_raw
        if total == 0:
            return {"home": 0.5, "away": 0.5}

        return {
            "home": round(home_raw / total, 3),
            "away": round(away_raw / total, 3),
        }


def _compute_value_bets(
    probs: Dict[str, float], odds: Dict[str, float]
) -> List[ValueBet]:
    """Identify markets where model probability exceeds implied bookmaker probability."""
    mapping = {
        "home": ("home_win", odds.get("home", 0)),
        "draw": ("draw", odds.get("draw", 0)),
        "away": ("away_win", odds.get("away", 0)),
    }
    bets = []
    for market, (prob_key, book_odds) in mapping.items():
        if book_odds <= 1.0:
            continue
        model_prob = probs.get(prob_key, 0.0)
        book_prob = 1.0 / book_odds
        edge = model_prob - book_prob
        if edge > 0.03:  # only flag bets with meaningful edge
            bets.append(ValueBet(
                market=market,
                model_probability=round(model_prob, 4),
                bookmaker_probability=round(book_prob, 4),
                edge=round(edge, 4),
                bookmaker_odds=book_odds,
            ))

    bets.sort(key=lambda b: b.edge, reverse=True)
    return bets
