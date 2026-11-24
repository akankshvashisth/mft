from __future__ import annotations

from typing import Any

from src.common.constants import DEFAULT_RISK_FREE_RATE
from src.scanner.combos import CandidateCombo, CandidateLeg
from src.scanner.verticals import (
    adjacent_verticals,
    candidate_leg,
    leg_vol_edge,
    mean_edge_to_half_spread,
    split_by_right,
    vertical_combo,
)

__all__ = ["CandidateCombo", "CandidateLeg", "SurfaceScanner"]

MIN_VERTICAL_CREDIT = 0.15
MIN_CONDOR_CREDIT = 0.30


class SurfaceScanner:
    """Scans calibrated SVI surface slices for relative-value & VRP option combos."""

    def __init__(
        self,
        min_residual_pts: float = 0.02,
        min_edge_to_half_spread: float = 0.10,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    ) -> None:
        self.min_residual_pts = min_residual_pts
        self.min_edge_to_half_spread = min_edge_to_half_spread
        self.risk_free_rate = risk_free_rate

    def _make_leg(self, raw: dict, side: int, qty: int = 1) -> CandidateLeg:
        return candidate_leg(raw, side, qty)

    def scan_surface(
        self,
        surface: dict,
        regime: dict | None = None,
        max_candidates: int = 6,
    ) -> list[CandidateCombo]:
        """Generates ranked defined-risk & relative-value combos from a calibrated surface."""
        if not surface.get("arb_free", False):
            return []
        # Every structure here sells volatility, so a blocked regime yields nothing
        short_vol_ok = True if regime is None else bool(regime.get("short_vol_allowed", False))
        if not short_vol_ok:
            return []

        underlying = surface["underlying"]
        context: dict[str, Any] = {
            "underlying": underlying,
            "spot": float(surface["spot"]),
            "forward": float(surface["forward"]),
            "ttm_days": float(surface["ttm_days"]),
            "risk_free_rate": self.risk_free_rate,
            "dividend_yield": float(surface.get("implied_q", 0.0125)),
        }
        puts, calls = split_by_right(surface.get("legs", []))

        # 1. Bull put and 2. bear call credit spreads on adjacent strikes
        candidates = [
            vertical_combo(pair, name_symbol=underlying, **context)
            for pair in adjacent_verticals(puts, calls, min_mid_credit=MIN_VERTICAL_CREDIT)
        ]

        # 3. Iron Condor (Combine OTM Bull Put + OTM Bear Call)
        condor = self._iron_condor(puts, calls, context)
        if condor is not None:
            candidates.append(condor)

        candidates.sort(
            key=lambda c: (c.theo_edge_per_share + 0.15 * c.net_credit),
            reverse=True,
        )
        return candidates[:max_candidates]

    @staticmethod
    def _iron_condor(puts: list, calls: list, context: dict[str, Any]) -> CandidateCombo | None:
        spot = context["spot"]
        otm_puts = [p for p in puts if p["strike"] < spot * 0.99]
        otm_calls = [c for c in calls if c["strike"] > spot * 1.01]
        if len(otm_puts) < 2 or len(otm_calls) < 2:
            return None

        sp, lp = otm_puts[-1], otm_puts[-2]
        sc, lc = otm_calls[0], otm_calls[1]
        width = max(sp["strike"] - lp["strike"], lc["strike"] - sc["strike"])
        net_credit = (sp["mid"] - lp["mid"]) + (sc["mid"] - lc["mid"])
        if net_credit <= MIN_CONDOR_CREDIT or width <= 0:
            return None

        theo_edge = leg_vol_edge(sp) - leg_vol_edge(lp) + leg_vol_edge(sc) - leg_vol_edge(lc)
        return CandidateCombo(
            name=(
                f"{context['underlying']} {int(lp['strike'])}/{int(sp['strike'])}P-"
                f"{int(sc['strike'])}/{int(lc['strike'])}C IC"
            ),
            strategy_type="IRON_CONDOR",
            legs=[
                candidate_leg(lp, +1),
                candidate_leg(sp, -1),
                candidate_leg(sc, -1),
                candidate_leg(lc, +1),
            ],
            net_credit=round(net_credit, 4),
            max_width=round(width, 2),
            theo_edge_per_share=round(theo_edge, 4),
            avg_edge_to_half_spread=round(mean_edge_to_half_spread(sp, lp, sc, lc), 3),
            **context,
        )
