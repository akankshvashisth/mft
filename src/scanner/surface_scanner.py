from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CandidateLeg:
    instrument_id: str
    strike: float
    right: str          # "C" or "P"
    side: int           # +1 = BUY, -1 = SELL
    qty: int            # Contract multiplier per combo unit (usually 1)
    bid: float
    ask: float
    mid: float
    raw_iv: float
    svi_iv: float
    residual_pts: float
    edge_to_half_spread: float


@dataclass
class CandidateCombo:
    name: str
    strategy_type: str  # "BULL_PUT_SPREAD", "BEAR_CALL_SPREAD", "IRON_CONDOR"
    underlying: str
    spot: float
    forward: float
    ttm_days: float
    risk_free_rate: float
    dividend_yield: float
    legs: list[CandidateLeg] = field(default_factory=list)
    net_credit: float = 0.0         # Positive = credit received ($/share)
    max_width: float = 0.0          # Strike width ($/share) for defined-risk spreads
    theo_edge_per_share: float = 0.0
    avg_edge_to_half_spread: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class SurfaceScanner:
    """Scans calibrated SVI surface slices for relative-value & VRP option combos."""

    def __init__(
        self,
        min_residual_pts: float = 0.02,
        min_edge_to_half_spread: float = 0.10,
        risk_free_rate: float = 0.0425,
    ) -> None:
        self.min_residual_pts = min_residual_pts
        self.min_edge_to_half_spread = min_edge_to_half_spread
        self.risk_free_rate = risk_free_rate

    def _make_leg(self, raw: dict, side: int, qty: int = 1) -> CandidateLeg:
        return CandidateLeg(
            instrument_id=raw["instrument_id"],
            strike=float(raw["strike"]),
            right=raw["right"],
            side=side,
            qty=qty,
            bid=float(raw["bid"]),
            ask=float(raw["ask"]),
            mid=float(raw["mid"]),
            raw_iv=float(raw["raw_iv"]),
            svi_iv=float(raw["svi_iv"]),
            residual_pts=float(raw["residual_pts"]),
            edge_to_half_spread=float(raw["edge_to_half_spread"]),
        )

    def scan_surface(
        self,
        surface: dict,
        regime: dict | None = None,
        max_candidates: int = 6,
    ) -> list[CandidateCombo]:
        """Generates ranked defined-risk & relative-value combos from a calibrated surface."""
        if not surface.get("arb_free", False):
            return []

        short_vol_ok = True if regime is None else bool(regime.get("short_vol_allowed", False))
        underlying = surface["underlying"]
        spot = float(surface["spot"])
        forward = float(surface["forward"])
        ttm_days = float(surface["ttm_days"])
        div_yield = float(surface.get("implied_q", 0.0125))
        legs = surface.get("legs", [])

        puts = sorted([x for x in legs if x["right"] == "P"], key=lambda x: x["strike"])
        calls = sorted([x for x in legs if x["right"] == "C"], key=lambda x: x["strike"])

        candidates: list[CandidateCombo] = []

        # 1. Bull Put Credit Spreads (Short higher-strike put, Long lower-strike wing put)
        if short_vol_ok and len(puts) >= 2:
            for i in range(1, len(puts)):
                short_p = puts[i]
                long_p = puts[i - 1]
                width = short_p["strike"] - long_p["strike"]
                if width <= 0 or width > 15.0:
                    continue
                net_credit = short_p["mid"] - long_p["mid"]
                if net_credit <= 0.15:
                    continue
                theo_edge = (
                    (short_p["raw_iv"] - short_p["svi_iv"]) * short_p["vega"]
                    - (long_p["raw_iv"] - long_p["svi_iv"]) * long_p["vega"]
                )
                avg_ehs = 0.5 * (short_p["edge_to_half_spread"] + long_p["edge_to_half_spread"])
                candidates.append(
                    CandidateCombo(
                        name=f"{underlying} {int(short_p['strike'])}/{int(long_p['strike'])}P Bull Put",
                        strategy_type="BULL_PUT_SPREAD",
                        underlying=underlying,
                        spot=spot,
                        forward=forward,
                        ttm_days=ttm_days,
                        risk_free_rate=self.risk_free_rate,
                        dividend_yield=div_yield,
                        legs=[self._make_leg(short_p, -1), self._make_leg(long_p, +1)],
                        net_credit=round(net_credit, 4),
                        max_width=round(width, 2),
                        theo_edge_per_share=round(theo_edge, 4),
                        avg_edge_to_half_spread=round(avg_ehs, 3),
                    )
                )

        # 2. Bear Call Credit Spreads (Short lower-strike OTM call, Long higher-strike wing call)
        if short_vol_ok and len(calls) >= 2:
            for i in range(0, len(calls) - 1):
                short_c = calls[i]
                long_c = calls[i + 1]
                width = long_c["strike"] - short_c["strike"]
                if width <= 0 or width > 15.0:
                    continue
                net_credit = short_c["mid"] - long_c["mid"]
                if net_credit <= 0.15:
                    continue
                theo_edge = (
                    (short_c["raw_iv"] - short_c["svi_iv"]) * short_c["vega"]
                    - (long_c["raw_iv"] - long_c["svi_iv"]) * long_c["vega"]
                )
                avg_ehs = 0.5 * (short_c["edge_to_half_spread"] + long_c["edge_to_half_spread"])
                candidates.append(
                    CandidateCombo(
                        name=f"{underlying} {int(short_c['strike'])}/{int(long_c['strike'])}C Bear Call",
                        strategy_type="BEAR_CALL_SPREAD",
                        underlying=underlying,
                        spot=spot,
                        forward=forward,
                        ttm_days=ttm_days,
                        risk_free_rate=self.risk_free_rate,
                        dividend_yield=div_yield,
                        legs=[self._make_leg(short_c, -1), self._make_leg(long_c, +1)],
                        net_credit=round(net_credit, 4),
                        max_width=round(width, 2),
                        theo_edge_per_share=round(theo_edge, 4),
                        avg_edge_to_half_spread=round(avg_ehs, 3),
                    )
                )

        # 3. Iron Condor (Combine OTM Bull Put + OTM Bear Call)
        otm_puts = [p for p in puts if p["strike"] < spot * 0.99]
        otm_calls = [c for c in calls if c["strike"] > spot * 1.01]
        if short_vol_ok and len(otm_puts) >= 2 and len(otm_calls) >= 2:
            sp, lp = otm_puts[-1], otm_puts[-2]
            sc, lc = otm_calls[0], otm_calls[1]
            width = max(sp["strike"] - lp["strike"], lc["strike"] - sc["strike"])
            net_credit = (sp["mid"] - lp["mid"]) + (sc["mid"] - lc["mid"])
            if net_credit > 0.30 and width > 0:
                theo_edge = (
                    (sp["raw_iv"] - sp["svi_iv"]) * sp["vega"]
                    - (lp["raw_iv"] - lp["svi_iv"]) * lp["vega"]
                    + (sc["raw_iv"] - sc["svi_iv"]) * sc["vega"]
                    - (lc["raw_iv"] - lc["svi_iv"]) * lc["vega"]
                )
                avg_ehs = 0.25 * (
                    sp["edge_to_half_spread"]
                    + lp["edge_to_half_spread"]
                    + sc["edge_to_half_spread"]
                    + lc["edge_to_half_spread"]
                )
                candidates.append(
                    CandidateCombo(
                        name=f"{underlying} {int(lp['strike'])}/{int(sp['strike'])}P-{int(sc['strike'])}/{int(lc['strike'])}C IC",
                        strategy_type="IRON_CONDOR",
                        underlying=underlying,
                        spot=spot,
                        forward=forward,
                        ttm_days=ttm_days,
                        risk_free_rate=self.risk_free_rate,
                        dividend_yield=div_yield,
                        legs=[
                            self._make_leg(lp, +1),
                            self._make_leg(sp, -1),
                            self._make_leg(sc, -1),
                            self._make_leg(lc, +1),
                        ],
                        net_credit=round(net_credit, 4),
                        max_width=round(width, 2),
                        theo_edge_per_share=round(theo_edge, 4),
                        avg_edge_to_half_spread=round(avg_ehs, 3),
                    )
                )

        candidates.sort(
            key=lambda c: (c.theo_edge_per_share + 0.15 * c.net_credit),
            reverse=True,
        )
        return candidates[:max_candidates]
