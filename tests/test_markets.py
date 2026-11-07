"""
Tests for all new betting markets, Kelly Criterion, calibration,
real provider stubs, and new API endpoints.
"""
import pytest


# ── Over/Under ────────────────────────────────────────────────────────────────

class TestOverUnderModel:
    def test_over_under_keys_present(self):
        from app.markets.over_under import OverUnderModel
        out = OverUnderModel().predict(1.8, 1.2)
        for threshold in ["0_5", "1_5", "2_5", "3_5", "4_5"]:
            assert f"over_{threshold}" in out
            assert f"under_{threshold}" in out

    def test_over_plus_under_equals_one(self):
        from app.markets.over_under import OverUnderModel
        out = OverUnderModel().predict(1.8, 1.2)
        for threshold in ["0_5", "1_5", "2_5", "3_5", "4_5"]:
            total = out[f"over_{threshold}"] + out[f"under_{threshold}"]
            assert abs(total - 1.0) < 0.01

    def test_high_xg_means_higher_over_probability(self):
        from app.markets.over_under import OverUnderModel
        ou = OverUnderModel()
        low  = ou.predict(0.8, 0.6)
        high = ou.predict(2.5, 2.0)
        assert high["over_2_5"] > low["over_2_5"]

    def test_expected_goals_in_output(self):
        from app.markets.over_under import OverUnderModel
        out = OverUnderModel().predict(1.5, 1.0)
        assert abs(out["expected_total_goals"] - 2.5) < 0.1


# ── BTTS ──────────────────────────────────────────────────────────────────────

class TestBTTSModel:
    def test_yes_plus_no_equals_one(self):
        from app.markets.btts import BTTSModel
        out = BTTSModel().predict(1.8, 1.2)
        assert abs(out["btts_yes"] + out["btts_no"] - 1.0) < 1e-4

    def test_higher_lambda_means_higher_btts(self):
        from app.markets.btts import BTTSModel
        b = BTTSModel()
        low  = b.predict(0.5, 0.5)
        high = b.predict(2.5, 2.5)
        assert high["btts_yes"] > low["btts_yes"]

    def test_analytical_correctness(self):
        import math
        from app.markets.btts import BTTSModel
        lh, la = 1.5, 1.2
        expected_yes = (1 - math.exp(-lh)) * (1 - math.exp(-la))
        out = BTTSModel().predict(lh, la)
        assert abs(out["btts_yes"] - expected_yes) < 1e-4


# ── Double Chance ─────────────────────────────────────────────────────────────

class TestDoubleChanceModel:
    def test_1x_equals_home_plus_draw(self):
        from app.markets.double_chance import DoubleChanceModel
        out = DoubleChanceModel().predict(0.50, 0.27, 0.23)
        assert abs(out["1X"] - 0.77) < 1e-4

    def test_x2_equals_draw_plus_away(self):
        from app.markets.double_chance import DoubleChanceModel
        out = DoubleChanceModel().predict(0.50, 0.27, 0.23)
        assert abs(out["X2"] - 0.50) < 1e-4

    def test_12_equals_home_plus_away(self):
        from app.markets.double_chance import DoubleChanceModel
        out = DoubleChanceModel().predict(0.50, 0.27, 0.23)
        assert abs(out["12"] - 0.73) < 1e-4

    def test_all_values_between_0_and_1(self):
        from app.markets.double_chance import DoubleChanceModel
        out = DoubleChanceModel().predict(0.40, 0.30, 0.30)
        for v in out.values():
            assert 0.0 <= v <= 1.0


# ── Correct Score ─────────────────────────────────────────────────────────────

class TestCorrectScoreModel:
    def test_returns_top_n_scores(self):
        from app.markets.correct_score import CorrectScoreModel
        out = CorrectScoreModel().predict(1.8, 1.2, top_n=5)
        assert len(out) == 5

    def test_sorted_by_probability_desc(self):
        from app.markets.correct_score import CorrectScoreModel
        out = CorrectScoreModel().predict(1.8, 1.2, top_n=10)
        probs = [s["probability"] for s in out]
        assert probs == sorted(probs, reverse=True)

    def test_score_format(self):
        from app.markets.correct_score import CorrectScoreModel
        out = CorrectScoreModel().predict(1.5, 1.0, top_n=3)
        for s in out:
            assert "-" in s["score"]
            h, a = s["score"].split("-")
            assert int(h) == s["home_goals"]
            assert int(a) == s["away_goals"]

    def test_high_xg_favours_higher_scores(self):
        from app.markets.correct_score import CorrectScoreModel
        cs = CorrectScoreModel()
        low_xg  = cs.predict(0.8, 0.6, top_n=1)[0]["score"]
        high_xg = cs.predict(3.0, 2.5, top_n=1)[0]["score"]
        # Most likely score should have more total goals at high xG
        hl, al = map(int, low_xg.split("-"))
        hh, ah = map(int, high_xg.split("-"))
        assert (hh + ah) >= (hl + al)


# ── Asian Handicap ────────────────────────────────────────────────────────────

class TestAsianHandicapModel:
    def test_probabilities_sum_to_one(self):
        from app.markets.asian_handicap import AsianHandicapModel
        out = AsianHandicapModel().predict(1.8, 1.2, handicap=0.0)
        total = out["home_cover"] + out["push"] + out["away_cover"]
        assert abs(total - 1.0) < 0.01

    def test_negative_handicap_reduces_home_cover(self):
        from app.markets.asian_handicap import AsianHandicapModel
        ah = AsianHandicapModel()
        even    = ah.predict(1.8, 1.2, handicap=0.0)
        minus_2 = ah.predict(1.8, 1.2, handicap=-2.0)
        assert minus_2["home_cover"] < even["home_cover"]

    def test_all_lines_returns_9_entries(self):
        from app.markets.asian_handicap import AsianHandicapModel
        lines = AsianHandicapModel().all_lines(1.8, 1.2)
        assert len(lines) == 9  # -2, -1.5, -1, -0.5, 0, +0.5, +1, +1.5, +2


# ── Kelly Criterion ───────────────────────────────────────────────────────────

class TestKellyCriterion:
    def test_no_bet_when_no_edge(self):
        from app.markets.kelly import KellyCriterion
        k = KellyCriterion()
        # Model prob = 0.40, implied = 1/2.2 = 0.455 → no edge
        out = k.calculate(0.40, 2.20, bankroll=1000)
        assert not out["is_value_bet"]
        assert out["recommended_stake"] == 0.0

    def test_positive_stake_when_edge_exists(self):
        from app.markets.kelly import KellyCriterion
        k = KellyCriterion()
        out = k.calculate(0.60, 2.10, bankroll=1000)
        assert out["is_value_bet"]
        assert out["recommended_stake"] > 0
        assert out["edge"] > 0

    def test_stake_capped_at_25_pct(self):
        from app.markets.kelly import KellyCriterion
        k = KellyCriterion(kelly_fraction=1.0)  # full Kelly
        out = k.calculate(0.95, 10.0, bankroll=1000)  # massive edge
        assert out["kelly_fraction"] <= 0.25

    def test_half_kelly_is_half_of_full(self):
        from app.markets.kelly import KellyCriterion
        full = KellyCriterion(kelly_fraction=1.0).calculate(0.55, 2.0, bankroll=1000)
        half = KellyCriterion(kelly_fraction=0.5).calculate(0.55, 2.0, bankroll=1000)
        if full["is_value_bet"] and half["is_value_bet"]:
            assert abs(half["kelly_fraction"] - full["kelly_fraction"] * 0.5) < 0.001

    def test_portfolio_scales_down_total_exposure(self):
        from app.markets.kelly import KellyCriterion
        bets = [
            {"market": "home",  "model_probability": 0.65, "bookmaker_odds": 1.8},
            {"market": "btts",  "model_probability": 0.70, "bookmaker_odds": 1.7},
            {"market": "over",  "model_probability": 0.68, "bookmaker_odds": 1.75},
        ]
        k = KellyCriterion(kelly_fraction=0.5)
        portfolio = k.calculate_portfolio(bets, bankroll=1000)
        total_fraction = sum(b["kelly_fraction"] for b in portfolio if b.get("is_value_bet"))
        assert total_fraction <= 0.50 + 1e-4  # max 50% total exposure


# ── Bundesliga mock data ──────────────────────────────────────────────────────

class TestBundesligaMockData:
    def test_bundesliga_teams_in_profiles(self):
        from app.ingestion.mock_provider import TEAM_PROFILES
        bl_teams = [t for t, p in TEAM_PROFILES.items() if p["league"] == "Bundesliga"]
        assert len(bl_teams) >= 18

    def test_bundesliga_players_available(self):
        from app.ingestion.mock_provider import MockDataProvider
        p = MockDataProvider()
        stats = p.get_player_stats("Bayern München")
        assert len(stats) >= 5
        kane = next((s for s in stats if "Kane" in s["name"]), None)
        assert kane is not None

    def test_today_fixtures_returns_bundesliga_games(self):
        from app.ingestion.mock_provider import MockDataProvider
        p = MockDataProvider()
        fixtures = p.get_todays_matches("Bundesliga")
        assert len(fixtures) > 0
        for f in fixtures:
            assert "home_team" in f and "away_team" in f and "kickoff" in f

    def test_bundesliga_team_stats(self):
        from app.ingestion.mock_provider import MockDataProvider
        p = MockDataProvider()
        stats = p.get_team_stats("Bayer Leverkusen")
        assert stats["attack_strength"] > 1.0
        assert stats["defense_weakness"] < 1.0


# ── Calibration ───────────────────────────────────────────────────────────────

class TestCalibration:
    def test_identity_when_not_fitted(self):
        from app.markets.calibration import BetaCalibrator
        calib = BetaCalibrator()
        # When not fitted, should return input unchanged
        probs = {"home_win": 0.50, "draw": 0.27, "away_win": 0.23}
        if not calib._fitted:
            out = calib.calibrate(probs)
            for k in probs:
                assert abs(out[k] - probs[k]) < 1e-6

    def test_calibrated_probs_sum_to_one(self):
        from app.markets.calibration import BetaCalibrator
        calib = BetaCalibrator()
        probs = {"home_win": 0.55, "draw": 0.25, "away_win": 0.20}
        out = calib.calibrate(probs)
        total = out["home_win"] + out["draw"] + out["away_win"]
        assert abs(total - 1.0) < 1e-4


# ── New API endpoints ─────────────────────────────────────────────────────────

class TestNewEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_today_endpoint_returns_matches(self, client):
        resp = client.get("/api/v1/today?league=Bundesliga")
        assert resp.status_code == 200
        data = resp.json()
        assert "matches" in data
        assert "league" in data
        assert data["league"] == "Bundesliga"
        assert len(data["matches"]) > 0

    def test_today_endpoint_has_predictions(self, client):
        resp = client.get("/api/v1/today?league=Bundesliga")
        data = resp.json()
        first = data["matches"][0]
        pred = first.get("prediction", {})
        assert "win_probability" in pred
        assert "over_under" in pred
        assert "btts" in pred
        assert "double_chance" in pred

    def test_markets_endpoint_basic(self, client):
        payload = {
            "home_team": "Bayern München",
            "away_team": "Borussia Dortmund",
            "competition": "Bundesliga",
        }
        resp = client.post("/api/v1/markets", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "markets" in data
        markets = data["markets"]
        assert "1x2" in markets
        assert "over_under" in markets
        assert "btts" in markets
        assert "double_chance" in markets
        assert "correct_score" in markets
        assert "asian_handicap" in markets

    def test_markets_endpoint_with_odds_and_kelly(self, client):
        payload = {
            "home_team": "Bayern München",
            "away_team": "Borussia Dortmund",
            "bankroll": 500.0,
            "kelly_fraction": 0.5,
            "bookmaker_odds": {
                "home": 1.55, "draw": 4.20, "away": 6.00,
                "over_2_5": 1.68, "under_2_5": 2.15,
                "btts_yes": 1.75, "btts_no": 2.00,
            },
        }
        resp = client.post("/api/v1/markets", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data.get("value_bets"), list)
        assert isinstance(data.get("kelly_stakes"), list)

    def test_markets_over_under_complement(self, client):
        payload = {"home_team": "RB Leipzig", "away_team": "Eintracht Frankfurt"}
        resp = client.post("/api/v1/markets", json=payload)
        data = resp.json()
        ou = data["markets"]["over_under"]
        assert abs(ou["over_2_5"] + ou["under_2_5"] - 1.0) < 0.01

    def test_today_with_player_probs(self, client):
        resp = client.get("/api/v1/today?league=Bundesliga&include_players=true")
        assert resp.status_code == 200
        data = resp.json()
        first = data["matches"][0]
        if "player_probabilities" in first:
            assert "home" in first["player_probabilities"]

    def test_root_lists_all_endpoints(self, client):
        resp = client.get("/")
        data = resp.json()
        assert "endpoints" in data
        assert any("today" in e for e in data["endpoints"])
        assert any("markets" in e for e in data["endpoints"])
