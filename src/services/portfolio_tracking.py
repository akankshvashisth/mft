"""Greeks P&L attribution against a per-day, per-book baseline, and sleeve equity / drawdown tracking."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.common.constants import NS_PER_DAY


def _zero_attribution(since_ts_ns: int) -> dict[str, Any]:
    return {
        "actual_pnl_change_usd": 0.0,
        "delta_pnl_usd": 0.0,
        "gamma_pnl_usd": 0.0,
        "vega_pnl_usd": 0.0,
        "theta_pnl_usd": 0.0,
        "residual_pnl_usd": 0.0,
        "since_ts_ns": since_ts_ns,
    }


class AttributionTracker:
    """Explains P&L since a baseline with the baseline's Greeks.

    The explanation is only valid while the book is unchanged, so the baseline is reset on each New York
    trading day and after every trade (any change of the book key).
    """

    def __init__(self) -> None:
        self.baseline_key: tuple | None = None
        self.spot: float | None = None
        self.atm_iv: float | None = None
        self.pnl_usd: float | None = None
        self.greeks: dict[str, float] | None = None
        self.ts_ns: int = 0

    def update(
        self,
        session_date: str,
        book_key: tuple,
        bench_spot: float,
        bench_atm_iv: float,
        total_pnl_usd: float,
        greeks: Mapping[str, float],
        now_ns: int,
    ) -> dict[str, Any]:
        """Rebases if needed, then returns the attribution of P&L since the baseline."""
        baseline_key = (session_date, book_key)
        if bench_spot > 0 and baseline_key != self.baseline_key:
            self.baseline_key = baseline_key
            self.spot = bench_spot
            self.atm_iv = bench_atm_iv
            self.pnl_usd = total_pnl_usd
            self.ts_ns = now_ns
            self.greeks = {
                "delta_sh": greeks["delta_sh"],
                "gamma_1pct": greeks["gamma_1pct"],
                "vega_1v": greeks["vega_1v"],
                "theta_day": greeks["theta_day"],
            }

        if self.greeks is None or not self.spot or bench_spot <= 0:
            return _zero_attribution(self.ts_ns)

        ds = bench_spot - self.spot
        ds_pct = (ds / self.spot) * 100.0
        div_pts = (bench_atm_iv - (self.atm_iv or bench_atm_iv)) * 100.0
        dt_days = max((now_ns - self.ts_ns) / NS_PER_DAY, 0.0)

        d_pnl = self.greeks["delta_sh"] * ds
        g_pnl = 0.5 * self.greeks["gamma_1pct"] * ds_pct * ds
        v_pnl = self.greeks["vega_1v"] * div_pts
        t_pnl = self.greeks["theta_day"] * dt_days
        act_chg = total_pnl_usd - (self.pnl_usd or 0.0)
        resid = act_chg - (d_pnl + g_pnl + v_pnl + t_pnl)
        return {
            "actual_pnl_change_usd": round(act_chg, 2),
            "delta_pnl_usd": round(d_pnl, 2),
            "gamma_pnl_usd": round(g_pnl, 2),
            "vega_pnl_usd": round(v_pnl, 2),
            "theta_pnl_usd": round(t_pnl, 2),
            "residual_pnl_usd": round(resid, 2),
            "since_ts_ns": self.ts_ns,
        }


class SleeveTracker:
    """Sleeve equity (allocated capital + this book's P&L), its high-water mark and drawdown.

    The in-memory cache only knows this session's realized P&L, so earlier realized P&L and the
    high-water mark are restored from the last published snapshot.
    """

    def __init__(self, allocated_capital_usd: float) -> None:
        self.allocated_capital_usd = allocated_capital_usd
        self.realized_carry_usd = 0.0
        self.high_water_mark = 0.0

    def restore(self, previous: Mapping[str, Any]) -> None:
        self.realized_carry_usd = float(previous.get("realized_total_usd") or 0.0)
        self.high_water_mark = float(previous.get("high_water_mark") or 0.0)

    def snapshot(
        self,
        realized_pnl_usd: float,
        unrealized_pnl_usd: float,
        open_positions: int,
        now_ns: int,
    ) -> dict[str, Any]:
        alloc = self.allocated_capital_usd
        equity = alloc + (realized_pnl_usd + unrealized_pnl_usd)
        if self.high_water_mark <= 0.0:
            self.high_water_mark = max(alloc, equity)
        self.high_water_mark = max(self.high_water_mark, equity)
        drawdown_pct = (equity - self.high_water_mark) / self.high_water_mark * 100.0
        return {
            "allocated_capital_usd": round(alloc, 2),
            "equity": round(equity, 2),
            "high_water_mark": round(self.high_water_mark, 2),
            "drawdown_pct": round(drawdown_pct, 4),
            "realized_total_usd": round(realized_pnl_usd, 2),
            "unrealized_pnl_usd": round(unrealized_pnl_usd, 2),
            "open_positions": open_positions,
            "ts_ns": now_ns,
        }
