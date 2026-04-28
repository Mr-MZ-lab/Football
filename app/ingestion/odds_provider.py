"""
The Odds API — real bookmaker odds aggregator.

Free tier: 500 requests/month.
Get key at: https://the-odds-api.com/
Set ODDS_API_KEY in your .env file.

Covers German bookmakers: Tipico, Bwin, Betano, Unibet, Bet365, etc.
Also covers ODDSET (Lotto) odds where available.

Falls back to returning empty odds when key is missing.
"""
import logging
from typing import Dict, Any, List, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

# Sport keys for The Odds API
SPORT_KEYS = {
    "Bundesliga":      "soccer_germany_bundesliga",
    "Premier League":  "soccer_epl",
    "La Liga":         "soccer_spain_la_liga",
    "Champions League":"soccer_uefa_champs_league",
}

# German bookmakers available on The Odds API
GERMAN_BOOKMAKERS = [
    "tipico_de",
    "bwin",
    "betano",
    "unibet_de",
    "bet365",
    "betway",
    "pinnacle",
]

# Market keys
MARKET_H2H   = "h2h"        # 1X2
MARKET_TOTALS = "totals"    # Over/Under
MARKET_BTTS   = "btts"      # Both Teams to Score


class OddsProvider:
    """
    Fetches real bookmaker odds from The Odds API.
    Returns structured dicts compatible with our value-bet detection.
    """

    def __init__(self, api_key: str):
        self._key = api_key
        self._client = httpx.Client(
            base_url=BASE_URL,
            params={"apiKey": api_key},
            timeout=10.0,
        )

    def get_match_odds(
        self,
        home_team: str,
        away_team: str,
        league: str = "Bundesliga",
        markets: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Returns odds for a specific match across all available bookmakers.

        Returns:
            {
              "match": "Bayern München vs Borussia Dortmund",
              "bookmakers": {
                "tipico_de": {
                  "1x2": {"home": 1.65, "draw": 4.00, "away": 5.50},
                  "over_under_2_5": {"over": 1.72, "under": 2.08},
                  "btts": {"yes": 1.80, "no": 1.90}
                },
                ...
              },
              "best_odds": {"home": 1.72, "draw": 4.20, "away": 5.80},
              "requests_remaining": 480
            }
        """
        sport = SPORT_KEYS.get(league, "soccer_germany_bundesliga")
        markets_param = ",".join(markets or [MARKET_H2H, MARKET_TOTALS, MARKET_BTTS])

        try:
            resp = self._client.get(
                f"/sports/{sport}/odds",
                params={
                    "regions": "eu",
                    "markets": markets_param,
                    "bookmakers": ",".join(GERMAN_BOOKMAKERS),
                    "oddsFormat": "decimal",
                },
            )
            resp.raise_for_status()
            events = resp.json()
            remaining = int(resp.headers.get("x-requests-remaining", -1))

            # Find the event matching our teams
            match_data = self._find_match(events, home_team, away_team)
            if not match_data:
                logger.info(f"No odds found for {home_team} vs {away_team}")
                return self._empty_odds(home_team, away_team, remaining)

            return self._parse_event(match_data, remaining)

        except Exception as e:
            logger.warning(f"Odds API failed: {e}")
            return self._empty_odds(home_team, away_team)

    def get_all_todays_odds(self, league: str = "Bundesliga") -> List[Dict[str, Any]]:
        """Fetch odds for ALL matches today in the given league."""
        sport = SPORT_KEYS.get(league, "soccer_germany_bundesliga")
        try:
            resp = self._client.get(
                f"/sports/{sport}/odds",
                params={
                    "regions": "eu",
                    "markets": f"{MARKET_H2H},{MARKET_TOTALS},{MARKET_BTTS}",
                    "bookmakers": ",".join(GERMAN_BOOKMAKERS),
                    "oddsFormat": "decimal",
                    "daysFrom": 1,
                },
            )
            resp.raise_for_status()
            events = resp.json()
            return [self._parse_event(e, -1) for e in events]
        except Exception as e:
            logger.warning(f"Bulk odds fetch failed: {e}")
            return []

    # ── helpers ───────────────────────────────────────────────────────────────

    def _find_match(self, events: List[Dict], home: str, away: str) -> Optional[Dict]:
        """Find the event whose team names roughly match our input."""
        home_lower = home.lower().replace("fc ", "").replace("1. ", "")
        away_lower = away.lower().replace("fc ", "").replace("1. ", "")
        for event in events:
            h = event.get("home_team", "").lower()
            a = event.get("away_team", "").lower()
            if home_lower in h and away_lower in a:
                return event
        return None

    def _parse_event(self, event: Dict, remaining: int) -> Dict[str, Any]:
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        bookmakers_raw = event.get("bookmakers", [])

        bookmakers = {}
        best_home = best_draw = best_away = 0.0
        best_over = best_under = best_btts_yes = best_btts_no = 0.0

        for bm in bookmakers_raw:
            bm_key = bm["key"]
            bm_data: Dict[str, Any] = {}

            for market in bm.get("markets", []):
                mk = market["key"]
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}

                if mk == MARKET_H2H:
                    bm_data["1x2"] = {
                        "home": outcomes.get(home_team, 0),
                        "draw": outcomes.get("Draw", 0),
                        "away": outcomes.get(away_team, 0),
                    }
                    best_home = max(best_home, outcomes.get(home_team, 0))
                    best_draw = max(best_draw, outcomes.get("Draw", 0))
                    best_away = max(best_away, outcomes.get(away_team, 0))

                elif mk == MARKET_TOTALS:
                    over = next((o["price"] for o in market["outcomes"] if o["name"] == "Over" and o.get("point", 0) == 2.5), 0)
                    under = next((o["price"] for o in market["outcomes"] if o["name"] == "Under" and o.get("point", 0) == 2.5), 0)
                    bm_data["over_under_2_5"] = {"over": over, "under": under}
                    best_over = max(best_over, over)
                    best_under = max(best_under, under)

                elif mk == MARKET_BTTS:
                    bm_data["btts"] = {
                        "yes": outcomes.get("Yes", 0),
                        "no":  outcomes.get("No", 0),
                    }
                    best_btts_yes = max(best_btts_yes, outcomes.get("Yes", 0))
                    best_btts_no  = max(best_btts_no, outcomes.get("No", 0))

            if bm_data:
                bookmakers[bm_key] = bm_data

        return {
            "match": f"{home_team} vs {away_team}",
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": event.get("commence_time", ""),
            "bookmakers": bookmakers,
            "best_odds": {
                "home": best_home, "draw": best_draw, "away": best_away,
                "over_2_5": best_over, "under_2_5": best_under,
                "btts_yes": best_btts_yes, "btts_no": best_btts_no,
            },
            "requests_remaining": remaining,
        }

    def _empty_odds(self, home: str, away: str, remaining: int = -1) -> Dict[str, Any]:
        return {
            "match": f"{home} vs {away}",
            "home_team": home,
            "away_team": away,
            "bookmakers": {},
            "best_odds": {},
            "requests_remaining": remaining,
            "note": "No odds available — check API key or match not yet listed.",
        }
