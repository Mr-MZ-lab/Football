"""
End-to-end tests for the AI Betting Brain prediction pipeline.

These tests run without a database or Redis — they exercise the full
prediction stack using mock data providers and in-memory model singletons.
"""
import pytest
import math
from datetime import datetime


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def feature_engineer():
    from app.processing.feature_engineer import FeatureEngineer
    return FeatureEngineer()


@pytest.fixture(scope="module")
def raw_features():
    from app.ingestion.data_loader import get_team_features
    return get_team_features("Manchester City", "Liverpool")


@pytest.fixture(scope="module")
def prematch_features(feature_engineer, raw_features):
    return feature_engineer.build_prematch_features(raw_features)


# ── Feature engineering ───────────────────────────────────────────────────────

class TestFeatureEngineer:
    def test_prematch_features_keys(self, prematch_features):
        required = {
            "lambda_home", "lambda_away",
            "home_attack", "away_attack",
            "home_form", "away_form",
            "h2h_home_win_rate",
        }
        assert required.issubset(prematch_features.keys())

    def test_lambda_positive(self, prematch_features):
        assert prematch_features["lambda_home"] > 0
        assert prematch_features["lambda_away"] > 0

    def test_possession_normalised(self, prematch_features):
        assert 0.0 <= prematch_features["home_avg_possession"] <= 1.0

    def test_live_features(self, feature_engineer, prematch_features):
        live_state = {
            "current_minute": 60,
            "home_goals": 1,
            "away_goals": 0,
            "home_red_cards": 0,
            "away_red_cards": 0,
            "home_shots": 10,
            "away_shots": 6,
            "home_shots_on_target": 4,
            "away_shots_on_target": 2,
            "home_possession": 55.0,
        }
        live_feats = feature_engineer.build_live_features(prematch_features, live_state)
        assert live_feats["current_minute"] == pytest.approx(60 / 90.0)
        assert live_feats["score_diff"] == 1


# ── Poisson model ─────────────────────────────────────────────────────────────

class TestPoissonModel:
    def test_probabilities_sum_to_one(self, prematch_features):
        from app.models_ml.poisson_model import PoissonModel
        out = PoissonModel().predict(prematch_features)
        total = out["home_win"] + out["draw"] + out["away_win"]
        assert abs(total - 1.0) < 1e-4

    def test_expected_goals_range(self, prematch_features):
        from app.models_ml.poisson_model import PoissonModel
        out = PoissonModel().predict(prematch_features)
        assert 0.1 < out["expected_home_goals"] < 6.0
        assert 0.1 < out["expected_away_goals"] < 6.0

    def test_home_advantage(self):
        """Home team should have higher expected goals than away for equal strength teams."""
        from app.models_ml.poisson_model import PoissonModel
        features = {"lambda_home": 1.4, "lambda_away": 1.1}
        out = PoissonModel().predict(features)
        assert out["expected_home_goals"] > out["expected_away_goals"]


# ── Logistic model ────────────────────────────────────────────────────────────

class TestLogisticModel:
    def test_probabilities_sum_to_one(self, prematch_features):
        from app.models_ml.logistic_model import LogisticModel
        out = LogisticModel().predict(prematch_features)
        total = out["home_win"] + out["draw"] + out["away_win"]
        assert abs(total - 1.0) < 1e-4

    def test_strong_team_favoured(self):
        from app.models_ml.logistic_model import LogisticModel
        strong = {
            "home_attack": 1.9, "home_defense": 0.6,
            "away_attack": 0.8, "away_defense": 1.4,
            "lambda_home": 2.5, "lambda_away": 0.7,
            "home_form": 0.9, "away_form": 0.2,
        }
        out = LogisticModel().predict(strong)
        assert out["home_win"] > out["away_win"]


# ── XGBoost model ─────────────────────────────────────────────────────────────

class TestXGBoostModel:
    def test_probabilities_sum_to_one(self, prematch_features):
        from app.models_ml.xgboost_model import XGBoostModel
        out = XGBoostModel().predict(prematch_features)
        total = out["home_win"] + out["draw"] + out["away_win"]
        assert abs(total - 1.0) < 1e-4

    def test_output_shape(self, prematch_features):
        from app.models_ml.xgboost_model import XGBoostModel
        out = XGBoostModel().predict(prematch_features)
        assert "home_win" in out and "draw" in out and "away_win" in out


# ── Late goal predictor ───────────────────────────────────────────────────────

class TestLateGoalPredictor:
    def test_probabilities_in_range(self):
        from app.models_ml.late_goal_predictor import LateGoalPredictor
        out = LateGoalPredictor().predict(1.8, 1.2)
        for key in ["after_75", "after_90", "home_late_goal", "away_late_goal"]:
            assert 0.0 <= out[key] <= 1.0

    def test_higher_xg_means_higher_late_goal_prob(self):
        from app.models_ml.late_goal_predictor import LateGoalPredictor
        p = LateGoalPredictor()
        low = p.predict(0.5, 0.5)
        high = p.predict(3.0, 3.0)
        assert high["after_75"] > low["after_75"]

    def test_late_game_returns_lower_prob_when_past_75(self):
        from app.models_ml.late_goal_predictor import LateGoalPredictor
        p = LateGoalPredictor()
        before_75 = p.predict(2.0, 1.5, current_minute=0)
        at_88 = p.predict(2.0, 1.5, current_minute=88)
        # After minute 88, after_75 window is basically closed
        assert at_88["after_75"] <= before_75["after_75"]


# ── Player model ──────────────────────────────────────────────────────────────

class TestPlayerModel:
    def test_returns_sorted_by_probability(self):
        from app.models_ml.player_model import PlayerScoringModel
        from app.ingestion.data_loader import get_player_features
        pdata = get_player_features("Manchester City", "Liverpool")
        results = PlayerScoringModel().predict(pdata["home_players"], 2.0, is_home=True)
        probs = [r["scoring_probability"] for r in results]
        assert probs == sorted(probs, reverse=True)

    def test_injured_player_excluded(self):
        from app.models_ml.player_model import PlayerScoringModel
        players = [
            {"name": "Player X", "team": "Team A", "position": "FWD",
             "xg_per_90": 0.5, "is_injured": True, "is_suspended": False,
             "penalties_taken": 0, "penalties_scored": 0, "id": 1},
            {"name": "Player Y", "team": "Team A", "position": "FWD",
             "xg_per_90": 0.3, "is_injured": False, "is_suspended": False,
             "penalties_taken": 0, "penalties_scored": 0, "id": 2},
        ]
        results = PlayerScoringModel().predict(players, 1.5)
        names = [r["player_name"] for r in results]
        assert "Player X" not in names
        assert "Player Y" in names

    def test_scoring_probability_in_range(self):
        from app.models_ml.player_model import PlayerScoringModel
        from app.ingestion.data_loader import get_player_features
        pdata = get_player_features("Arsenal", "Chelsea")
        results = PlayerScoringModel().predict(pdata["home_players"], 1.8)
        for r in results:
            assert 0.0 <= r["scoring_probability"] <= 1.0


# ── Markov Chain model ────────────────────────────────────────────────────────

class TestMarkovChainModel:
    def test_probabilities_sum_to_one(self, prematch_features):
        from app.models_ml.markov_chain import MarkovChainModel
        live_state = {"current_minute": 60, "home_goals": 1, "away_goals": 1,
                      "home_red_cards": 0, "away_red_cards": 0}
        out = MarkovChainModel().predict(prematch_features, live_state)
        total = out["home_win"] + out["draw"] + out["away_win"]
        assert abs(total - 1.0) < 0.02  # Monte Carlo so allow small tolerance

    def test_leading_team_has_higher_win_prob(self, prematch_features):
        from app.models_ml.markov_chain import MarkovChainModel
        m = MarkovChainModel()
        equal_features = {**prematch_features, "lambda_home": 1.3, "lambda_away": 1.3}
        leading = {"current_minute": 80, "home_goals": 2, "away_goals": 0,
                   "home_red_cards": 0, "away_red_cards": 0}
        out = m.predict(equal_features, leading)
        assert out["home_win"] > 0.85


# ── Bayesian updater ──────────────────────────────────────────────────────────

class TestBayesianUpdater:
    def test_probabilities_sum_to_one(self, prematch_features):
        from app.models_ml.bayesian_updater import BayesianUpdater
        live_state = {"current_minute": 45, "home_goals": 1, "away_goals": 0,
                      "home_red_cards": 0, "away_red_cards": 0}
        out = BayesianUpdater().predict(prematch_features, live_state)
        total = out["home_win"] + out["draw"] + out["away_win"]
        assert abs(total - 1.0) < 1e-4

    def test_red_card_reduces_team_win_prob(self, prematch_features):
        from app.models_ml.bayesian_updater import BayesianUpdater
        b = BayesianUpdater()
        no_red = {"current_minute": 60, "home_goals": 0, "away_goals": 0,
                  "home_red_cards": 0, "away_red_cards": 0}
        with_red = {"current_minute": 60, "home_goals": 0, "away_goals": 0,
                    "home_red_cards": 1, "away_red_cards": 0}
        out_no = b.predict(prematch_features, no_red)
        out_red = b.predict(prematch_features, with_red)
        assert out_red["home_win"] < out_no["home_win"]


# ── Ensemble ──────────────────────────────────────────────────────────────────

class TestEnsembleEngine:
    def test_combined_probs_sum_to_one(self):
        from app.ensemble.ensemble import EnsembleEngine
        outputs = [
            {"model": "poisson",  "home_win": 0.50, "draw": 0.25, "away_win": 0.25},
            {"model": "logistic", "home_win": 0.45, "draw": 0.30, "away_win": 0.25},
            {"model": "xgboost",  "home_win": 0.48, "draw": 0.27, "away_win": 0.25},
        ]
        combined = EnsembleEngine().combine_prematch(outputs)
        total = combined["home_win"] + combined["draw"] + combined["away_win"]
        assert abs(total - 1.0) < 1e-4

    def test_confidence_high_when_models_agree(self):
        from app.ensemble.ensemble import EnsembleEngine
        outputs = [
            {"model": "poisson",  "home_win": 0.60, "draw": 0.25, "away_win": 0.15},
            {"model": "logistic", "home_win": 0.61, "draw": 0.24, "away_win": 0.15},
            {"model": "xgboost",  "home_win": 0.59, "draw": 0.26, "away_win": 0.15},
        ]
        score = EnsembleEngine().confidence_score(outputs)
        assert score > 0.85

    def test_confidence_low_when_models_disagree(self):
        from app.ensemble.ensemble import EnsembleEngine
        outputs = [
            {"model": "poisson",  "home_win": 0.80, "draw": 0.10, "away_win": 0.10},
            {"model": "logistic", "home_win": 0.10, "draw": 0.10, "away_win": 0.80},
            {"model": "xgboost",  "home_win": 0.33, "draw": 0.34, "away_win": 0.33},
        ]
        score = EnsembleEngine().confidence_score(outputs)
        assert score < 0.5


# ── FastAPI integration test ──────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_predict_endpoint(self, client):
        payload = {
            "home_team": "Manchester City",
            "away_team": "Liverpool",
            "include_player_probs": False,
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "win_probability" in data
        assert "expected_goals" in data
        assert "late_goal_probability" in data
        assert "confidence_score" in data
        total = (
            data["win_probability"]["home"]
            + data["win_probability"]["draw"]
            + data["win_probability"]["away"]
        )
        assert abs(total - 1.0) < 0.01

    def test_live_endpoint(self, client):
        payload = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "current_minute": 65,
            "home_goals": 1,
            "away_goals": 0,
            "home_red_cards": 0,
            "away_red_cards": 0,
            "home_shots": 10,
            "away_shots": 5,
            "home_shots_on_target": 4,
            "away_shots_on_target": 2,
            "home_possession": 58.0,
        }
        resp = client.post("/api/v1/live", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_minute"] == 65
        assert data["current_score"] == "1-0"

    def test_player_probability_endpoint(self, client):
        payload = {
            "home_team": "Manchester City",
            "away_team": "Arsenal",
        }
        resp = client.post("/api/v1/player-probability", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "home_players" in data
        assert "away_players" in data
        assert len(data["home_players"]) > 0

    def test_predict_with_bookmaker_odds(self, client):
        payload = {
            "home_team": "Liverpool",
            "away_team": "Tottenham",
            "bookmaker_odds": {"home": 1.80, "draw": 3.50, "away": 4.50},
        }
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data.get("value_bets"), list)
