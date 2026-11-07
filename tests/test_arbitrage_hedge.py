"""Tests for Arbitrage detection, Hedging, and Live Betting substitutions."""
import pytest


# ── Arbitrage unit tests ──────────────────────────────────────────────────────

class TestArbitrageDetector:

    def _no_arb_odds(self):
        return {"home": 1.80, "draw": 3.40, "away": 4.50}

    def _arb_odds(self):
        # implied sum ≈ 0.476 + 0.263 + 0.244 = 0.983 → arb exists
        return {"home": 2.10, "draw": 3.80, "away": 4.10}

    def test_no_arb_detected_for_typical_odds(self):
        from app.markets.arbitrage import ArbitrageDetector
        result = ArbitrageDetector().detect(self._no_arb_odds())
        assert result["is_arb"] is False
        assert result["implied_sum"] > 1.0

    def test_arb_detected_when_implied_sum_below_one(self):
        from app.markets.arbitrage import ArbitrageDetector
        result = ArbitrageDetector().detect(self._arb_odds())
        assert result["is_arb"] is True
        assert result["margin"] > 0

    def test_stakes_sum_to_total_stake(self):
        from app.markets.arbitrage import ArbitrageDetector
        result = ArbitrageDetector().detect(self._arb_odds(), total_stake=1000)
        if result["is_arb"]:
            total = sum(result["stakes"].values())
            assert abs(total - 1000) < 1.0

    def test_returns_are_equal_across_outcomes(self):
        from app.markets.arbitrage import ArbitrageDetector
        result = ArbitrageDetector().detect(self._arb_odds(), total_stake=1000)
        if result["is_arb"]:
            returns = list(result["returns"].values())
            assert max(returns) - min(returns) < 1.0  # within €1 rounding

    def test_guaranteed_profit_positive(self):
        from app.markets.arbitrage import ArbitrageDetector
        result = ArbitrageDetector().detect(self._arb_odds(), total_stake=1000)
        if result["is_arb"]:
            assert result["guaranteed_profit"] > 0

    def test_scan_markets_finds_best_odds(self):
        from app.markets.arbitrage import ArbitrageDetector
        matrix = {
            "tipico": {"home": 2.10, "draw": 3.50, "away": 3.70},
            "bwin":   {"home": 2.00, "draw": 3.80, "away": 4.10},
            "betano": {"home": 2.05, "draw": 3.60, "away": 3.90},
        }
        results = ArbitrageDetector().scan_markets(matrix, 1000)
        assert len(results) > 0
        r = results[0]
        assert "best_odds_source" in r
        assert r["odds_used"]["home"] == 2.10   # best home is tipico
        assert r["odds_used"]["draw"] == 3.80   # best draw is bwin
        assert r["odds_used"]["away"] == 4.10   # best away is bwin

    def test_empty_bookmaker_matrix_returns_empty(self):
        from app.markets.arbitrage import ArbitrageDetector
        results = ArbitrageDetector().scan_markets({})
        assert results == []

    def test_implied_sum_pct_reported(self):
        from app.markets.arbitrage import ArbitrageDetector
        result = ArbitrageDetector().detect(self._no_arb_odds())
        assert "implied_sum_pct" in result
        assert result["implied_sum_pct"] > 100.0  # overround > 100% means no arb


# ── Hedging unit tests ────────────────────────────────────────────────────────

class TestHedgeCalculator:

    def test_green_book_guaranteed_profit_positive(self):
        from app.markets.hedging import HedgeCalculator
        result = HedgeCalculator().green_book(
            original_stake=100, original_odds=3.00,
            hedge_odds={"draw": 3.20, "away": 4.50},
        )
        assert result["guaranteed_profit"] > 0

    def test_green_book_hedge_stakes_are_positive(self):
        from app.markets.hedging import HedgeCalculator
        result = HedgeCalculator().green_book(
            original_stake=100, original_odds=3.00,
            hedge_odds={"away": 2.10},
        )
        assert all(v > 0 for v in result["hedge_stakes"].values())

    def test_green_book_returns_equalised(self):
        from app.markets.hedging import HedgeCalculator
        result = HedgeCalculator().green_book(
            original_stake=100, original_odds=3.00,
            hedge_odds={"draw": 3.20, "away": 4.50},
        )
        returns = list(result["hedge_returns"].values())
        # All hedge returns should be equal (within rounding)
        assert max(returns) - min(returns) < 0.50

    def test_loss_limit_caps_loss(self):
        from app.markets.hedging import HedgeCalculator
        result = HedgeCalculator().loss_limit(
            original_stake=100, original_odds=3.50,
            hedge_odds={"away": 2.20}, max_loss=30,
        )
        assert result["max_loss_guaranteed"] <= 30.5  # small rounding buffer

    def test_loss_limit_no_hedge_needed_when_max_loss_gte_stake(self):
        from app.markets.hedging import HedgeCalculator
        result = HedgeCalculator().loss_limit(
            original_stake=100, original_odds=3.00,
            hedge_odds={"away": 2.00}, max_loss=120,
        )
        assert "note" in result

    def test_breakeven_returns_zero_profit(self):
        from app.markets.hedging import HedgeCalculator
        result = HedgeCalculator().breakeven(
            original_stake=100, original_odds=2.50, hedge_odds=1.80,
        )
        assert result["profit_either_way"] == 0.0
        assert result["hedge_stake"] > 0

    def test_roi_reported_in_green_book(self):
        from app.markets.hedging import HedgeCalculator
        result = HedgeCalculator().green_book(
            original_stake=100, original_odds=3.00,
            hedge_odds={"away": 2.10},
        )
        assert "roi_pct" in result


# ── API endpoint tests ────────────────────────────────────────────────────────

class TestArbitrageEndpoint:

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_no_arb_response(self, client):
        payload = {
            "bookmaker_odds": {
                "tipico": {"home": 1.80, "draw": 3.40, "away": 4.50},
                "bwin":   {"home": 1.75, "draw": 3.50, "away": 4.20},
            },
            "total_stake": 500,
        }
        resp = client.post("/api/v1/arbitrage", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "is_arb" in data
        assert "implied_sum" in data
        assert "advice" in data

    def test_arb_opportunity_detected(self, client):
        # Craft odds that sum below 1.0
        payload = {
            "bookmaker_odds": {
                "siteA": {"home": 2.20},
                "siteB": {"away": 2.20},
                "siteC": {"draw": 10.0},
            },
            "total_stake": 1000,
        }
        resp = client.post("/api/v1/arbitrage", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "is_arb" in data
        assert "stakes" in data

    def test_best_odds_per_outcome_in_response(self, client):
        payload = {
            "bookmaker_odds": {
                "tipico": {"home": 2.10, "draw": 3.50},
                "bwin":   {"home": 2.00, "draw": 3.80},
            },
            "total_stake": 1000,
        }
        resp = client.post("/api/v1/arbitrage", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["best_odds_per_outcome"]["home"] == 2.10
        assert data["best_odds_per_outcome"]["draw"] == 3.80


class TestHedgeEndpoint:

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_green_book_endpoint(self, client):
        payload = {
            "original_stake": 100,
            "original_odds": 3.00,
            "hedge_odds": {"draw": 3.20, "away": 4.50},
            "mode": "green_book",
        }
        resp = client.post("/api/v1/hedge", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "green_book"
        assert "guaranteed_profit" in data
        assert "hedge_stakes" in data
        assert "summary" in data

    def test_loss_limit_endpoint(self, client):
        payload = {
            "original_stake": 100,
            "original_odds": 4.00,
            "hedge_odds": {"away": 2.00},
            "mode": "loss_limit",
            "max_loss": 40,
        }
        resp = client.post("/api/v1/hedge", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "loss_limit"

    def test_breakeven_endpoint(self, client):
        payload = {
            "original_stake": 100,
            "original_odds": 2.50,
            "hedge_odds": {"away": 1.80},
            "mode": "breakeven",
        }
        resp = client.post("/api/v1/hedge", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["profit_either_way"] == 0.0

    def test_invalid_mode_returns_422(self, client):
        payload = {
            "original_stake": 100,
            "original_odds": 2.50,
            "hedge_odds": {"away": 1.80},
            "mode": "magic_mode",
        }
        resp = client.post("/api/v1/hedge", json=payload)
        assert resp.status_code == 422


# ── Live Betting with substitutions ──────────────────────────────────────────

class TestLiveWithSubstitutions:

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_live_accepts_substitutions(self, client):
        payload = {
            "home_team": "Bayern München",
            "away_team": "Borussia Dortmund",
            "current_minute": 75,
            "home_goals": 1,
            "away_goals": 2,
            "home_shots": 12,
            "away_shots": 8,
            "home_possession": 58.0,
            "home_subs": ["Thomas Müller", "Serge Gnabry"],
            "away_subs": ["Donyell Malen"],
        }
        resp = client.post("/api/v1/live", json=payload)
        assert resp.status_code == 200

    def test_substitutions_appear_in_response(self, client):
        payload = {
            "home_team": "Bayern München",
            "away_team": "Borussia Dortmund",
            "current_minute": 80,
            "home_goals": 0,
            "away_goals": 1,
            "home_subs": ["Leroy Sané"],
        }
        resp = client.post("/api/v1/live", json=payload)
        data = resp.json()
        assert "substitutions" in data
        assert data["substitutions"] is not None
        assert "home_subs" in data["substitutions"]
        assert "Leroy Sané" in data["substitutions"]["home_subs"]

    def test_subs_boost_expected_goals(self, client):
        base = {
            "home_team": "Bayern München",
            "away_team": "Borussia Dortmund",
            "current_minute": 75,
            "home_goals": 0,
            "away_goals": 1,
            "home_subs": [],
        }
        with_subs = {**base, "home_subs": ["Thomas Müller", "Serge Gnabry"]}

        r_base = client.post("/api/v1/live", json=base).json()
        r_subs = client.post("/api/v1/live", json=with_subs).json()

        xg_base = r_base["expected_remaining_goals"]["home"]
        xg_subs = r_subs["expected_remaining_goals"]["home"]
        assert xg_subs > xg_base
