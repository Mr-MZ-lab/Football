"""
Late Scorer Model — Who scores after minute 90?

Combines the hazard-function goal rate for the 90+ window with
per-player scoring probability, adjusted for match situation.

Key factors unique to stoppage-time goals:
  1. Match state (losing team pushes everyone forward → defenders score too)
  2. Substitutes (fresh legs from 70-80' often score late)
  3. Penalty probability (fouls spike in 90+ → penalty taker crucial)
  4. Set-piece specialists (corners, free-kicks in 90+)
  5. Fatigue gap (high-energy players like FWDs tire vs fresh subs)

Research basis:
  - Losing teams score ~38% of their 90+ goals through set pieces
  - Substitutes account for ~28% of goals scored after minute 75
  - Penalty rate in 90+ is ~2.5× higher than 0-89' (desperation fouls)
"""
import math
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Hazard multiplier for 90+ relative to per-minute base rate
STOPPAGE_HAZARD = 1.45

# Assumed stoppage time duration (minutes)
STOPPAGE_MINUTES = 5  # default; can be overridden

# Positional base weights in 90+ — defenders push up when losing
POSITION_WEIGHTS = {
    "FWD": 1.00,
    "MID": 0.75,
    "DEF": 0.40,   # rises to 0.70 if team is losing
    "GK":  0.05,
}

# Desperation multipliers: how much a losing team boosts each position
DESPERATION = {
    "FWD": 1.15,
    "MID": 1.25,   # midfielders push further forward
    "DEF": 3.50,   # defenders bomb forward for corners / set pieces in 90+
    "GK":  1.30,   # keeper may come up for corners (extreme case)
}

# P(penalty in 90+) per match (empirical estimate)
P_PENALTY_90_PLUS = 0.10  # ~10% of 90+ situations have a penalty

# Penalty conversion rate
P_PENALTY_SCORED = 0.78


def _p_score_from_xg(rate: float) -> float:
    """P(at least one goal) from a Poisson rate."""
    return 1.0 - math.exp(-max(0.0, rate))


class LateScorer:
    """
    Predicts which player is most likely to score in stoppage time (90+).

    Inputs:
        players         — list of player dicts (from MockDataProvider / real API)
        team_lambda     — team's expected goals per 90 from the Poisson model
        score_diff      — current goal difference from this team's perspective
                          negative = losing, 0 = drawing, positive = winning
        stoppage_mins   — expected stoppage time in minutes (default 5)
        subs            — list of player names who are substitutes (came on during match)
        is_home         — slight home advantage
    """

    def predict(
        self,
        players: List[Dict[str, Any]],
        team_lambda: float,
        score_diff: int = 0,
        stoppage_mins: int = STOPPAGE_MINUTES,
        subs: Optional[List[str]] = None,
        is_home: bool = True,
    ) -> List[Dict[str, Any]]:

        available = [
            p for p in players
            if not p.get("is_injured") and not p.get("is_suspended")
        ]
        if not available:
            return []

        subs = subs or []
        losing = score_diff < 0
        home_factor = 1.05 if is_home else 0.97

        # Base per-minute rate for this team in 90+ window
        base_rate_per_min = (team_lambda / 90.0) * STOPPAGE_HAZARD * home_factor

        results = []
        for player in available:
            pos = player.get("position", "MID")
            xg90 = max(0.0, player.get("xg_per_90", 0.05))
            name = player["name"]
            is_sub = name in subs

            # ── Position weight (adjusted for desperation) ────────────────
            pos_w = POSITION_WEIGHTS.get(pos, 0.5)
            if losing:
                pos_w *= DESPERATION.get(pos, 1.2)

            # ── xG contribution in stoppage window ────────────────────────
            # Player's share of team xG, scaled to stoppage_mins
            match_rate = xg90 * (stoppage_mins / 90.0) * pos_w * STOPPAGE_HAZARD
            xg_90plus = match_rate * home_factor

            # ── Substitute bonus: fresh legs, tactical impact ─────────────
            if is_sub:
                xg_90plus *= 1.20   # +20% for substitute energy

            # ── Penalty component ─────────────────────────────────────────
            pen_taken  = player.get("penalties_taken", 0)
            is_pen_taker = pen_taken > 0
            pen_xg = P_PENALTY_90_PLUS * P_PENALTY_SCORED if is_pen_taker else 0.0

            total_xg = xg_90plus + pen_xg

            # ── Scoring probability ───────────────────────────────────────
            prob = _p_score_from_xg(total_xg)

            # ── Reason tags ───────────────────────────────────────────────
            reasons = []
            if is_pen_taker:
                reasons.append("penalty taker")
            if is_sub:
                reasons.append("substitute — fresh legs")
            if losing and pos in ("DEF", "MID"):
                reasons.append("team pushing forward (losing)")
            if xg90 > 0.50:
                reasons.append("high xG scorer")

            results.append({
                "player_name":         name,
                "team":                player.get("team", ""),
                "position":            pos,
                "scoring_probability": round(min(0.99, prob), 4),
                "xg_90plus":           round(total_xg, 4),
                "is_penalty_taker":    is_pen_taker,
                "is_substitute":       is_sub,
                "reasons":             reasons,
            })

        # Sort by scoring probability
        results.sort(key=lambda x: x["scoring_probability"], reverse=True)

        # Normalise so team total = P(team scores in 90+)
        total_player_xg = sum(r["xg_90plus"] for r in results) or 1.0
        team_90plus_xg  = base_rate_per_min * stoppage_mins

        scale = team_90plus_xg / total_player_xg
        for r in results:
            r["xg_90plus"] = round(r["xg_90plus"] * scale, 4)
            r["scoring_probability"] = round(
                min(0.99, _p_score_from_xg(r["xg_90plus"])), 4
            )

        results.sort(key=lambda x: x["scoring_probability"], reverse=True)
        return results

    def team_90plus_probability(
        self,
        team_lambda: float,
        stoppage_mins: int = STOPPAGE_MINUTES,
        score_diff: int = 0,
        is_home: bool = True,
    ) -> Dict[str, float]:
        """
        Team-level probability of scoring at least one goal in 90+.
        Separate from the player breakdown — used as the headline number.
        """
        home_factor = 1.05 if is_home else 0.97
        desperation_factor = 1.15 if score_diff < 0 else 1.0

        rate = (team_lambda / 90.0) * STOPPAGE_HAZARD * home_factor * desperation_factor
        exp_goals = rate * stoppage_mins

        return {
            "p_score_90plus":       round(_p_score_from_xg(exp_goals), 4),
            "expected_goals_90plus": round(exp_goals, 4),
            "stoppage_minutes":     stoppage_mins,
            "desperation_factor":   round(desperation_factor, 2),
        }
