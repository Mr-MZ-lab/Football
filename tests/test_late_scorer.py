"""Tests for the Late Goal Scorer model and endpoint."""
import pytest


class TestLateScorer:

    def _players(self):
        return [
            {"name": "Harry Kane",   "team": "Bayern München", "position": "FWD",
             "xg_per_90": 0.88, "is_injured": False, "is_suspended": False,
             "penalties_taken": 4, "penalties_scored": 3},
            {"name": "Jamal Musiala","team": "Bayern München", "position": "MID",
             "xg_per_90": 0.42, "is_injured": False, "is_suspended": False,
             "penalties_taken": 0, "penalties_scored": 0},
            {"name": "Thomas Müller","team": "Bayern München", "position": "FWD",
             "xg_per_90": 0.28, "is_injured": False, "is_suspended": False,
             "penalties_taken": 0, "penalties_scored": 0},
            {"name": "Min-jae Kim",  "team": "Bayern München", "position": "DEF",
             "xg_per_90": 0.04, "is_injured": False, "is_suspended": False,
             "penalties_taken": 0, "penalties_scored": 0},
        ]

    def test_returns_sorted_by_probability(self):
        from app.models_ml.late_scorer import LateScorer
        results = LateScorer().predict(self._players(), team_lambda=1.8)
        probs = [r["scoring_probability"] for r in results]
        assert probs == sorted(probs, reverse=True)

    def test_penalty_taker_is_top_scorer(self):
        from app.models_ml.late_scorer import LateScorer
        results = LateScorer().predict(self._players(), team_lambda=1.8)
        top = results[0]
        assert top["player_name"] == "Harry Kane"
        assert top["is_penalty_taker"] is True

    def test_substitute_has_bonus(self):
        from app.models_ml.late_scorer import LateScorer
        ls = LateScorer()
        without_sub = ls.predict(self._players(), team_lambda=1.8, subs=[])
        with_sub    = ls.predict(self._players(), team_lambda=1.8, subs=["Jamal Musiala"])
        prob_no_sub = next(r["scoring_probability"] for r in without_sub if r["player_name"] == "Jamal Musiala")
        prob_sub    = next(r["scoring_probability"] for r in with_sub    if r["player_name"] == "Jamal Musiala")
        assert prob_sub > prob_no_sub

    def test_defender_gets_boost_when_losing(self):
        from app.models_ml.late_scorer import LateScorer
        ls = LateScorer()
        winning = ls.predict(self._players(), team_lambda=1.8, score_diff=1)
        losing  = ls.predict(self._players(), team_lambda=1.8, score_diff=-1)
        kim_w = next(r["scoring_probability"] for r in winning if r["player_name"] == "Min-jae Kim")
        kim_l = next(r["scoring_probability"] for r in losing  if r["player_name"] == "Min-jae Kim")
        assert kim_l > kim_w

    def test_injured_player_excluded(self):
        from app.models_ml.late_scorer import LateScorer
        players = self._players()
        players[0]["is_injured"] = True
        results = LateScorer().predict(players, team_lambda=1.8)
        names = [r["player_name"] for r in results]
        assert "Harry Kane" not in names

    def test_all_probabilities_in_range(self):
        from app.models_ml.late_scorer import LateScorer
        results = LateScorer().predict(self._players(), team_lambda=1.8)
        for r in results:
            assert 0.0 <= r["scoring_probability"] <= 1.0

    def test_team_90plus_probability(self):
        from app.models_ml.late_scorer import LateScorer
        out = LateScorer().team_90plus_probability(1.8, stoppage_mins=5)
        assert 0.0 < out["p_score_90plus"] < 1.0
        assert out["stoppage_minutes"] == 5

    def test_losing_team_has_higher_90plus_prob(self):
        from app.models_ml.late_scorer import LateScorer
        ls = LateScorer()
        winning = ls.team_90plus_probability(1.5, score_diff=1)
        losing  = ls.team_90plus_probability(1.5, score_diff=-1)
        assert losing["p_score_90plus"] > winning["p_score_90plus"]

    def test_reasons_tags(self):
        from app.models_ml.late_scorer import LateScorer
        results = LateScorer().predict(
            self._players(), team_lambda=1.8, subs=["Jamal Musiala"]
        )
        musiala = next(r for r in results if r["player_name"] == "Jamal Musiala")
        assert "substitute — fresh legs" in musiala["reasons"]

        kane = next(r for r in results if r["player_name"] == "Harry Kane")
        assert "penalty taker" in kane["reasons"]


class TestLateGoalScorerEndpoint:

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_basic_response(self, client):
        payload = {
            "home_team": "Bayern München",
            "away_team": "Borussia Dortmund",
            "home_goals": 1,
            "away_goals": 2,
            "current_minute": 88,
            "stoppage_minutes": 5,
        }
        resp = client.post("/api/v1/late-scorer", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "home_scorers" in data
        assert "away_scorers" in data
        assert "most_likely_scorer" in data
        assert "team_90plus" in data

    def test_losing_team_has_defenders_in_top5(self, client):
        # Bayern losing 0-2 → defenders push up
        payload = {
            "home_team": "Bayern München",
            "away_team": "Borussia Dortmund",
            "home_goals": 0,
            "away_goals": 2,
            "current_minute": 90,
            "stoppage_minutes": 6,
        }
        resp = client.post("/api/v1/late-scorer", json=payload)
        data = resp.json()
        home_scorers = data["home_scorers"]
        positions = [p["position"] for p in home_scorers]
        # DEF should appear somewhere in the list (desperation factor)
        assert "DEF" in positions

    def test_substitute_appears_with_flag(self, client):
        payload = {
            "home_team": "Bayern München",
            "away_team": "Borussia Dortmund",
            "home_goals": 1,
            "away_goals": 1,
            "current_minute": 89,
            "stoppage_minutes": 4,
            "home_subs": ["Thomas Müller", "Serge Gnabry"],
        }
        resp = client.post("/api/v1/late-scorer", json=payload)
        data = resp.json()
        home = {p["player_name"]: p for p in data["home_scorers"]}
        if "Thomas Müller" in home:
            assert home["Thomas Müller"]["is_substitute"] is True

    def test_narrative_contains_context(self, client):
        payload = {
            "home_team": "RB Leipzig",
            "away_team": "Eintracht Frankfurt",
            "home_goals": 0,
            "away_goals": 1,
            "current_minute": 90,
            "stoppage_minutes": 5,
        }
        resp = client.post("/api/v1/late-scorer", json=payload)
        data = resp.json()
        assert "narrative" in data
        assert len(data["narrative"]) > 20

    def test_most_likely_scorer_is_top_of_combined(self, client):
        payload = {
            "home_team": "Bayern München",
            "away_team": "Borussia Dortmund",
            "home_goals": 1,
            "away_goals": 1,
            "current_minute": 88,
            "stoppage_minutes": 5,
        }
        resp = client.post("/api/v1/late-scorer", json=payload)
        data = resp.json()
        best = data["most_likely_scorer"]
        combined = data["top_scorers_combined"]
        assert best["scoring_probability"] == combined[0]["scoring_probability"]
