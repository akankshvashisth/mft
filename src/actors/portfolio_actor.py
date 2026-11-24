from datetime import timedelta
import json
import math
import time
import numpy as np
import redis
from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.model import (
    InstrumentId,
    OptionContract,
    Position,
)


class PortfolioAnalyticsActorConfig(DataActorConfig):
    def __init__(
        self,
        benchmark_id: InstrumentId,
        allocated_capital_usd: float = 100_000.0,
        risk_free_rate: float = 0.0425,
        interval_secs: int = 15,
        valkey_url: str = "redis://127.0.0.1:6379/0",
    ) -> None:
        super().__init__()
        self.benchmark_id = benchmark_id
        self.allocated_capital_usd = allocated_capital_usd
        self.risk_free_rate = risk_free_rate
        self.interval_secs = interval_secs
        self.valkey_url = valkey_url


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_greeks(
    spot: float,
    strike: float,
    ttm: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
) -> dict[str, float]:
    """Black-Scholes-Merton price + 1st/2nd-order Greeks (Delta, Gamma, Vega/1v, Theta/day, Vanna, Charm)."""
    if ttm <= 1e-6 or sigma <= 1e-6 or spot <= 0.0 or strike <= 0.0:
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        delta = (1.0 if spot > strike else 0.0) if is_call else (-1.0 if spot < strike else 0.0)
        return {
            "price": intrinsic,
            "delta": delta,
            "gamma": 0.0,
            "vega_1v": 0.0,
            "theta_day": 0.0,
            "vanna": 0.0,
            "charm_day": 0.0,
        }

    sqrt_t = math.sqrt(ttm)
    disc_q = math.exp(-q * ttm)
    disc_r = math.exp(-r * ttm)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * ttm) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf_d1 = _norm_pdf(d1)

    if is_call:
        nd1 = _norm_cdf(d1)
        nd2 = _norm_cdf(d2)
        price = spot * disc_q * nd1 - strike * disc_r * nd2
        delta = disc_q * nd1
        theta_ann = (
            -(spot * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
            - r * strike * disc_r * nd2
            + q * spot * disc_q * nd1
        )
        charm_ann = -disc_q * (
            pdf_d1 * (2.0 * (r - q) * ttm - d2 * sigma * sqrt_t) / (2.0 * ttm * sigma * sqrt_t)
            - q * nd1
        )
    else:
        n_neg_d1 = _norm_cdf(-d1)
        n_neg_d2 = _norm_cdf(-d2)
        price = strike * disc_r * n_neg_d2 - spot * disc_q * n_neg_d1
        delta = -disc_q * n_neg_d1
        theta_ann = (
            -(spot * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
            + r * strike * disc_r * n_neg_d2
            - q * spot * disc_q * n_neg_d1
        )
        charm_ann = -disc_q * (
            pdf_d1 * (2.0 * (r - q) * ttm - d2 * sigma * sqrt_t) / (2.0 * ttm * sigma * sqrt_t)
            + q * n_neg_d1
        )

    gamma = (disc_q * pdf_d1) / (spot * sigma * sqrt_t)
    vega_1v = (spot * disc_q * pdf_d1 * sqrt_t) * 0.01  # per +1.00 vol pt (+0.01 sigma)
    vanna = -disc_q * pdf_d1 * (d2 / sigma) * 0.01  # dDelta / d(1 vol pt)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega_1v": vega_1v,
        "theta_day": theta_ann / 365.25,
        "vanna": vanna,
        "charm_day": charm_ann / 365.25,
    }


class PortfolioAnalyticsActor(DataActor):
    """Real-time beta-weighted book Greeks, Spot x Vol stress matrix, and P&L attribution."""

    def __init__(self, config: PortfolioAnalyticsActorConfig) -> None:
        super().__init__(config)
        self.rdb = redis.from_url(config.valkey_url) if config.valkey_url else None
        self.baseline_spot: float | None = None
        self.baseline_atm_iv: float | None = None
        self.baseline_pnl_usd: float | None = None
        self.baseline_greeks: dict[str, float] | None = None
        self.baseline_ts_ns: int = 0

    def on_start(self) -> None:
        self.clock.set_timer(
            "portfolio_analytics_eval",
            timedelta(seconds=self.config.interval_secs),
            callback=self.on_timer,
        )
        self.log.info(
            f"PortfolioAnalyticsActor started (benchmark={self.config.benchmark_id}, "
            f"sleeve=${self.config.allocated_capital_usd:,.0f})"
        )

    def _read_json(self, key: str) -> dict:
        if self.rdb is None:
            return {}
        raw = self.rdb.get(key)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _compute_book_stress_grid(
        self,
        legs_spec: list[dict],
        spot_shocks_pct: tuple[float, ...] = (-10.0, -5.0, -2.5, 0.0, 2.5, 5.0, 10.0),
        vol_shocks_pts: tuple[float, ...] = (-5.0, -2.5, 0.0, 2.5, 5.0, 10.0),
        horizon_days: float = 7.0,
    ) -> list[list[float]]:
        """Computes exact 2D Spot x Vol book P&L matrix ($ USD) across all open legs."""
        dt = horizon_days / 365.25
        grid: list[list[float]] = []
        for s_pct in spot_shocks_pct:
            s_mult = 1.0 + s_pct / 100.0
            row: list[float] = []
            for v_pts in vol_shocks_pts:
                v_shift = v_pts / 100.0
                pnl_total = 0.0
                for leg in legs_spec:
                    qty = leg["signed_qty"]
                    mult = leg["multiplier"]
                    s_new = leg["spot"] * s_mult
                    if leg["is_option"]:
                        t_new = max(leg["ttm"] - dt, 1e-4)
                        sig_new = max(leg["iv"] + v_shift, 0.03)
                        g_new = _bs_greeks(
                            s_new,
                            leg["strike"],
                            t_new,
                            self.config.risk_free_rate,
                            leg["q"],
                            sig_new,
                            leg["is_call"],
                        )
                        pnl_total += qty * mult * (g_new["price"] - leg["theo_price"])
                    else:
                        pnl_total += qty * mult * (s_new - leg["spot"])
                row.append(round(pnl_total, 2))
            grid.append(row)
        return grid

    def on_timer(self, event) -> None:
        now_wall_ns = time.time_ns()
        bench_sym = str(self.config.benchmark_id)
        surf = self._read_json(f"mft:surface:{bench_sym}")
        rv_univ = self._read_json("mft:rv:universe")
        bench_rv = rv_univ.get(bench_sym, self._read_json(f"mft:rv:{bench_sym}"))

        bench_spot = float(surf.get("spot") or bench_rv.get("spot") or 0.0)
        bench_atm_iv = float(surf.get("atm_iv") or bench_rv.get("rv_20d") or 0.20)
        bench_q = float(surf.get("implied_q") or 0.015)

        positions: list[Position] = list(self.cache.positions_open())
        closed_positions: list[Position] = list(self.cache.positions_closed())

        net_delta_shares = 0.0
        beta_delta_shares = 0.0
        net_gamma_1pct = 0.0
        net_vega_1v = 0.0
        net_theta_day = 0.0
        net_vanna = 0.0
        net_charm_day = 0.0
        unrealized_pnl_usd = 0.0
        realized_pnl_usd = sum(float(p.realized_pnl) for p in closed_positions) + sum(
            float(p.realized_pnl) for p in positions
        )

        legs_spec: list[dict] = []
        position_rows: list[dict] = []

        for pos in positions:
            iid = pos.instrument_id
            inst = self.cache.instrument(iid)
            signed_qty = float(pos.signed_qty)
            mult = float(getattr(inst, "multiplier", 100.0 if isinstance(inst, OptionContract) else 1.0))
            avg_px = float(pos.avg_px_open)

            if isinstance(inst, OptionContract):
                und_sym = f"{inst.underlying}.ARCA"
                und_surf = self._read_json(f"mft:surface:{und_sym}") or surf
                s_und = float(und_surf.get("spot") or bench_spot or 1.0)
                q_und = float(und_surf.get("implied_q") or bench_q)
                iv_est = float(und_surf.get("atm_iv") or bench_atm_iv)
                strike = float(inst.strike_price)
                is_call = "CALL" in str(inst.option_kind).upper()
                ttm = max((int(inst.expiration_ns) - now_wall_ns) / (365.25 * 86400.0 * 1e9), 1e-4)

                g = _bs_greeks(
                    s_und, strike, ttm, self.config.risk_free_rate, q_und, iv_est, is_call
                )
                pos_pnl = signed_qty * mult * (g["price"] - avg_px)
                unrealized_pnl_usd += pos_pnl

                d_sh = signed_qty * mult * g["delta"]
                # Beta-weight delta to benchmark SPY via spot ratio (beta=1.0 default)
                b_d_sh = d_sh * (s_und / max(bench_spot, 1e-6)) if bench_spot > 0 else d_sh
                g_1pct = signed_qty * mult * g["gamma"] * (0.01 * s_und)
                v_1v = signed_qty * mult * g["vega_1v"]
                th_d = signed_qty * mult * g["theta_day"]
                van = signed_qty * mult * g["vanna"]
                chm = signed_qty * mult * g["charm_day"]

                net_delta_shares += d_sh
                beta_delta_shares += b_d_sh
                net_gamma_1pct += g_1pct
                net_vega_1v += v_1v
                net_theta_day += th_d
                net_vanna += van
                net_charm_day += chm

                legs_spec.append(
                    {
                        "is_option": True,
                        "signed_qty": signed_qty,
                        "multiplier": mult,
                        "spot": s_und,
                        "strike": strike,
                        "ttm": ttm,
                        "q": q_und,
                        "iv": iv_est,
                        "is_call": is_call,
                        "theo_price": g["price"],
                    }
                )
                position_rows.append(
                    {
                        "instrument": str(iid),
                        "qty": signed_qty,
                        "avg_px": round(avg_px, 3),
                        "mark_px": round(g["price"], 3),
                        "pnl_usd": round(pos_pnl, 2),
                        "delta_sh": round(d_sh, 2),
                        "vega_1v": round(v_1v, 2),
                        "theta_day": round(th_d, 2),
                    }
                )
            else:
                s_und = bench_spot if bench_spot > 0 else avg_px
                pos_pnl = signed_qty * mult * (s_und - avg_px)
                unrealized_pnl_usd += pos_pnl
                d_sh = signed_qty * mult
                net_delta_shares += d_sh
                beta_delta_shares += d_sh
                legs_spec.append(
                    {
                        "is_option": False,
                        "signed_qty": signed_qty,
                        "multiplier": mult,
                        "spot": s_und,
                    }
                )
                position_rows.append(
                    {
                        "instrument": str(iid),
                        "qty": signed_qty,
                        "avg_px": round(avg_px, 3),
                        "mark_px": round(s_und, 3),
                        "pnl_usd": round(pos_pnl, 2),
                        "delta_sh": round(d_sh, 2),
                        "vega_1v": 0.0,
                        "theta_day": 0.0,
                    }
                )

        total_pnl_usd = realized_pnl_usd + unrealized_pnl_usd

        # Initialize or compute Greeks P&L attribution against session baseline
        if self.baseline_greeks is None and bench_spot > 0:
            self.baseline_spot = bench_spot
            self.baseline_atm_iv = bench_atm_iv
            self.baseline_pnl_usd = total_pnl_usd
            self.baseline_ts_ns = now_wall_ns
            self.baseline_greeks = {
                "delta_sh": net_delta_shares,
                "gamma_1pct": net_gamma_1pct,
                "vega_1v": net_vega_1v,
                "theta_day": net_theta_day,
            }

        attribution = {
            "actual_pnl_change_usd": 0.0,
            "delta_pnl_usd": 0.0,
            "gamma_pnl_usd": 0.0,
            "vega_pnl_usd": 0.0,
            "theta_pnl_usd": 0.0,
            "residual_pnl_usd": 0.0,
        }
        if self.baseline_greeks is not None and self.baseline_spot and bench_spot > 0:
            ds = bench_spot - self.baseline_spot
            ds_pct = (ds / self.baseline_spot) * 100.0
            div_pts = (bench_atm_iv - (self.baseline_atm_iv or bench_atm_iv)) * 100.0
            dt_days = max((now_wall_ns - self.baseline_ts_ns) / (86400.0 * 1e9), 0.0)

            d_pnl = self.baseline_greeks["delta_sh"] * ds
            g_pnl = 0.5 * self.baseline_greeks["gamma_1pct"] * ds_pct * ds
            v_pnl = self.baseline_greeks["vega_1v"] * div_pts
            t_pnl = self.baseline_greeks["theta_day"] * dt_days
            act_chg = total_pnl_usd - (self.baseline_pnl_usd or 0.0)
            resid = act_chg - (d_pnl + g_pnl + v_pnl + t_pnl)
            attribution = {
                "actual_pnl_change_usd": round(act_chg, 2),
                "delta_pnl_usd": round(d_pnl, 2),
                "gamma_pnl_usd": round(g_pnl, 2),
                "vega_pnl_usd": round(v_pnl, 2),
                "theta_pnl_usd": round(t_pnl, 2),
                "residual_pnl_usd": round(resid, 2),
            }

        spot_shocks = (-10.0, -5.0, -2.5, 0.0, 2.5, 5.0, 10.0)
        vol_shocks = (-5.0, -2.5, 0.0, 2.5, 5.0, 10.0)
        stress_grid = (
            self._compute_book_stress_grid(legs_spec, spot_shocks, vol_shocks, horizon_days=7.0)
            if legs_spec
            else [[0.0 for _ in vol_shocks] for _ in spot_shocks]
        )
        worst_stress_usd = float(np.min(stress_grid))

        # Approximate P-measure 1-day 99% VaR / ES from book Delta/Gamma/Vega + RV20
        rv_20d = float(bench_rv.get("rv_20d") or bench_atm_iv)
        daily_sigma_usd = bench_spot * (rv_20d / math.sqrt(252.0))
        lin_std_usd = abs(beta_delta_shares) * daily_sigma_usd + abs(net_vega_1v) * (rv_20d * 100.0 * 0.08)
        book_var_99_usd = round(2.326 * lin_std_usd, 2)
        book_es_99_usd = round(2.665 * lin_std_usd, 2)

        payload = {
            "benchmark": bench_sym,
            "benchmark_spot": round(bench_spot, 2),
            "open_positions": len(positions),
            "realized_pnl_usd": round(realized_pnl_usd, 2),
            "unrealized_pnl_usd": round(unrealized_pnl_usd, 2),
            "total_pnl_usd": round(total_pnl_usd, 2),
            "greeks": {
                "net_delta_shares": round(net_delta_shares, 2),
                "beta_delta_shares": round(beta_delta_shares, 2),
                "net_gamma_1pct_shares": round(net_gamma_1pct, 3),
                "net_vega_1v_usd": round(net_vega_1v, 2),
                "net_theta_day_usd": round(net_theta_day, 2),
                "net_vanna_1v": round(net_vanna, 3),
                "net_charm_day": round(net_charm_day, 3),
            },
            "risk": {
                "book_var_99_usd": book_var_99_usd,
                "book_es_99_usd": book_es_99_usd,
                "worst_7d_stress_usd": round(worst_stress_usd, 2),
                "spot_shocks_pct": list(spot_shocks),
                "vol_shocks_pts": list(vol_shocks),
                "stress_grid_usd": stress_grid,
            },
            "attribution": attribution,
            "positions": position_rows,
            "ts_ns": now_wall_ns,
        }

        if self.rdb is not None:
            self.rdb.set("mft:portfolio:risk", json.dumps(payload))

        self.log.info(
            f"PortfolioBook open={len(positions)} PnL=${total_pnl_usd:+.2f} "
            f"BetaDelta={beta_delta_shares:+.1f}sh Vega=${net_vega_1v:+.2f}/v "
            f"Theta=${net_theta_day:+.2f}/d ES99=${book_es_99_usd:.2f}"
        )
