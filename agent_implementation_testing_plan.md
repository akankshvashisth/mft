# Autonomous Agent Execution & Testing Plan — `akankshvashisth/mft`

> **Target Agent:** DeepSeek V4 Pro (or equivalent autonomous coding agent)  
> **Reviewer:** Human operator + Gemini audit pass  
> **Domain Constraints:** US Mid-Cap American Equity Options ($2B–$10B market cap), Medium-Frequency Trading (MFT, holding periods hours to 14–30 days), $100K virtual sleeve scaling to $1M–$2M+, IBKR 100-line live streaming cap, Portfolio Margin $100K cliff.

---

## 0. Non-Negotiable Operating Rules for the Agent

Read these rules before editing any file. Violations will fail code review:

1. **Preserve Hexagonal Architecture:** Keep pure decision/math logic inside `src/quant/`, `src/risk/*_rules.py`, `src/strategies/vrp_core.py`, and `src/services/` free of `nautilus_trader` or `redis` imports. All broker/cache I/O stays behind protocols (`OrderGateway`, `MarketView`, `BookMarket`, `JsonStore`).
2. **Fail-Closed Policy (Zero Silent Fallbacks):** Never replace missing, `None`, `0`, `NaN`, or stale risk/market inputs with optimistic defaults (e.g., no `or 0.25` RV, no `or 10` TOB size, no `max(1, ...)` lot floor, no `$0` ES99 on unmodeled positions). Surface explicit reason codes and block entries.
3. **Never Weaken Existing Tests:** Do not delete, skip, or loosen existing assertions to make `pytest` pass. If a bug fix intentionally changes behavior tested by an old test (e.g., removing the `max(1, ...)` TOB floor), document the exact test update in your handoff report with justification.
4. **Atomic Work Packages:** Complete **one Work Package (WP)** per branch/commit (or clearly separated commits `wp-01`, `wp-02`, ...). Run the full `pytest` suite after every WP before moving on.
5. **Preserve Public Valkey / Dashboard Schemas:** Additive JSON fields are allowed; do not rename or drop existing keys consumed by `src/dashboard/` or `ui/app.py` without updating their readers and tests.
6. **Inspect Before Editing:** Before modifying `src/risk/scenario_pricer.py`, `src/execution/passive_combo.py`, `src/services/portfolio_marks.py`, or `src/quant/risk_measures.py`, read the current file in full—do not assume line numbers.

---

## 1. Work Package Roadmap & Execution Order

Execute in strict numerical order. Stop after **Sprint 1 (WP-1 to WP-3)** and **Sprint 2 (WP-4 to WP-6)** to hand diffs back for review before starting Sprint 3–4.

| Sprint | WP | Priority | Title | Primary Files |
| :--- | :--- | :--- | :--- | :--- |
| **Sprint 1** | **WP-1** | P0 | OHLC Hygiene, Finite-Sample Bipower & Fail-Closed RV | `src/quant/realized_vol.py`, `src/strategies/vrp_engine.py` |
| **Sprint 1** | **WP-2** | P0 | Eliminate Phantom Mid-Cap Liquidity (`0 or 10` & `max(1, ...)` Floor) | `src/strategies/vrp_engine.py`, `src/strategies/vrp_core.py`, `src/risk/gate_rules.py`, `src/risk/gate_types.py` |
| **Sprint 1** | **WP-3** | P0 | Unified 14-Day Nonlinear Book ES99 & Fail-Closed `UNMODELED_SPREAD` | `src/services/portfolio.py`, `src/services/portfolio_marks.py`, `src/quant/risk_measures.py`, `src/risk/gate_rules.py` |
| **Sprint 2** | **WP-4** | P0 | Crash/Restart Reconciliation, Overlapping-Leg Exits & Realized P&L Idempotency | `src/strategies/vrp_engine.py`, `src/strategies/vrp_core.py`, `src/adapters/order_gateway.py`, `src/services/portfolio.py` |
| **Sprint 2** | **WP-5** | P1 | Execution Hardening: Winning Exit Floor, Arrival Mid, Tick Sizing & Adverse Move | `src/strategies/vrp_engine.py`, `src/execution/passive_combo.py` |
| **Sprint 2** | **WP-6** | P1 | `ScenarioPricer` BSM Carry, Round-Trip Friction, Crash Vol & CRN | `src/risk/scenario_pricer.py` |
| **Sprint 3** | **WP-7** | P0/P1 | Walk-Forward Backtester Truth & Live-Gate Parity | `src/backtest/vrp_walkforward.py` |
| **Sprint 4** | **WP-8** | P2 | Delta-Targeted Mid-Cap Verticals & `RiskDaemon` IBKR Margin / $100K PM Cliff | `src/strategies/vrp_core.py`, `src/scanner/verticals.py`, `src/risk/daemon_rules.py`, `src/risk/gate_rules.py` |
| **Sprint 4** | **WP-9** | P2 | Two-Tier Mid-Cap Universe Rotator & Programmatic Event/HTB Protocol | `src/scanner/universe_rotator.py` (new), `src/data/event_calendar.py` |

---

## 2. Detailed Work Package Specifications

---

### WP-1 (P0): OHLC Hygiene, Finite-Sample Bipower & Fail-Closed RV

**Problem:**
1. `yang_zhang_rv()` checks finiteness/positivity across the entire `df` rather than the active `window + 1` slice and performs no OHLC structural or outlier validation (causing `rv_max = 7.5789` on SPY in live Valkey).
2. Any single bad bar collapses `yang_zhang_rv()` to `0.0`, and `vrp_engine.py:403` silently falls back via `float(rv_snap.get("rv_20d") or 0.25)`, inventing a fake +30 vol pt VRP on 55%-vol mid-caps.
3. `bipower_jump_ratio()` omits the $\frac{W}{W-1}$ finite-sample correction, biasing jump share $+5\%$ high at $W=20$.
4. `trailing_rolling_rv()` is $O(N^2)$ Python looping.

**Required Changes:**
1. **`src/quant/realized_vol.py`:**
   * Add a pure helper `sanitize_ohlc(df: pd.DataFrame, max_abs_log_ret: float = 0.45) -> pd.DataFrame`:
     * Drop rows where any of `open, high, low, close` is non-finite or $\le 0$.
     * Drop rows violating OHLC consistency: `high < max(open, close)` or `low > min(open, close)`.
     * Drop unverified outlier spikes where $|\ln(C_t / C_{t-1})| > \text{max\_abs\_log\_ret}$ (configurable; default `0.45` accommodates legitimate mid-cap earnings gaps while catching corrupt 10x data spikes like `rv_max = 7.58`).
   * Update `yang_zhang_rv(df, window, ann=TRADING_DAYS_PER_YEAR)`:
     * Sanitize `df` (or validate only the active `window + 1` rows after sanitization) so an old corrupt bar 200 days ago does not zero out today's 20d RV.
   * Update `bipower_jump_ratio(df, window=20)`:
     * Apply the finite-sample correction:
       $$\text{BV} = \frac{W}{W - 1} \cdot \frac{\pi}{2} \sum_{t=2}^{W} |r_t| \cdot |r_{t-1}|$$
       where $W = \text{len}(r)$.
   * Vectorize `trailing_rolling_rv()` using pandas/numpy rolling windows (verify numerical equivalence to scalar `yang_zhang_rv` within `1e-10`).
2. **`src/strategies/vrp_engine.py`:**
   * Remove `rv_20d = float(rv_snap.get("rv_20d") or 0.25)`.
   * Validate `rv_snap` freshness (`snapshot_is_fresh` against `max_rv_age_secs`, e.g., `SECS_PER_DAY`) and require finite `rv_20d > 0.0`.
   * If `rv_20d` is missing, stale, non-finite, or $\le 0.0$: **do not evaluate entries**; publish `"action": "STAND_DOWN"`, `"entry_block_reason": "RV_MISSING_OR_INVALID"`, `"rv_20d": None`. Still run `manage_working_orders()` and `evaluate_exits()` so existing positions remain managed.

**Required Tests:**
* `tests/quant/test_realized_vol.py`:
  * Corrupt bar at index 0 (`NaN` or `0.0`) does **not** collapse `yang_zhang_rv(df, 20)` on a 60-bar frame.
  * Bar with `high < close` or a 10x bad print is quarantined by `sanitize_ohlc` (no 758% RV spike).
  * `bipower_jump_ratio` on pure i.i.d. Gaussian returns (large-sample average across seeds) has mean near `0.0`, not `+0.05`.
  * Vectorized `trailing_rolling_rv` matches scalar loop within `1e-10` and runs 250 bars in `< 5 ms`.
* `tests/strategies/test_vrp_engine.py`:
  * Missing `mft:rv:<underlying>`, `rv_20d = 0.0`, or stale RV snapshot forces `STAND_DOWN` with `RV_MISSING_OR_INVALID` and routes zero entry orders while still evaluating exits.

---

### WP-2 (P0): Eliminate Phantom Mid-Cap Liquidity (`0 or 10` & `max(1, ...)` Floor)

**Problem:**
* `vrp_engine.py` converts `bid_size=0` to `10` (`int(quote.bid_size or 10)`).
* `vrp_core.py` defaults surface-fallback TOB sizes to `DEFAULT_TOB_SIZE = 10`.
* `gate_rules.py` uses `max(1, int(leg.open_interest * cfg.max_oi_participation))` and `max(1, int(tob_size * cfg.max_tob_participation))`, allowing 1-lot orders even when TOB size is `0` or `1`.

**Required Changes:**
1. **`src/strategies/vrp_engine.py` & `src/strategies/vrp_core.py`:**
   * Preserve `None` vs `0` for `bid_size` and `ask_size` (change `StrategyQuote` to `tuple[float, float, int | None, int | None]`).
   * Replace `int(quote.bid_size or 10)` with `int(quote.bid_size) if quote.bid_size is not None else None`.
   * In `enrich_surface_legs()`, if falling back to the surface snapshot (which has no live TOB depth), set `bid_size = raw.get("bid_size")` (i.e., `None` if absent)—**never** invent `10`.
2. **`src/risk/gate_types.py` & `src/risk/gate_rules.py`:**
   * Allow `bid_size: int | None` and `ask_size: int | None` on `CandidateLegSpec`.
   * Add `min_tob_size: int = 4` and `allow_unknown_tob_size: bool = False` to `GateConfig`.
   * In `check_legs()`:
     * If `tob_size is None`: append `TOB_SIZE_UNKNOWN(<id>)` to `reasons` (unless `allow_unknown_tob_size=True`, in which case add warning).
     * If `tob_size < cfg.min_tob_size`: append `LOW_TOB_SIZE(<id>: size=<tob_size><<min>)` to `reasons`.
     * Compute `tob_cap = int(tob_size * cfg.max_tob_participation)` (NO `max(1, ...)`). If `tob_cap < 1`, append `TOB_PARTICIPATION_ZERO(<id>)` to `reasons` and set `capped_qty = 0`.
     * Same for `oi_cap = int(leg.open_interest * cfg.max_oi_participation)`: if `oi_cap < 1`, append `OI_PARTICIPATION_ZERO(<id>)` to `reasons` and set `capped_qty = 0`.

**Required Tests:**
* `tests/risk/test_gate_rules.py` & `tests/strategies/test_vrp_core.py`:
  * `bid_size=0` on a `SELL` leg is **not** converted to `10` and is rejected by `check_legs()`.
  * `bid_size=None` (surface fallback without TOB) is rejected when `allow_unknown_tob_size=False`.
  * `tob_size=2` with `max_tob_participation=0.25` produces `tob_cap=0` and blocks entry (does not force 1 lot).
  * `open_interest=5` with `max_oi_participation=0.10` produces `oi_cap=0` and blocks entry.

---

### WP-3 (P0): Unified 14-Day Nonlinear Book ES99 & Fail-Closed `UNMODELED_SPREAD`

**Problem:**
1. `PortfolioService` computes `book_es_99_usd` via `parametric_book_var_es(beta_delta_shares, net_vega_1v, bench_spot, rv_20d)`—a **1-day linear normal** estimate that omits `net_gamma_1pct` and ignores idiosyncratic mid-cap risk—while `PreTradeGate` adds it to `ScenarioPricer`'s **14-day nonlinear MC** `incremental_es99_per_lot_usd`. This creates a ~3.7x unit mismatch and permits over-allocation.
2. `es_budget_check()` silently skips the check if `incremental_es99_per_lot_usd <= 0.0`.
3. If `mft:spread_legs:<spread_id>` is missing from Valkey, `PortfolioService` marks the spread `UNMODELED_SPREAD` with `$0` ES99 and `$0` Greeks, blinding the gate.

**Required Changes:**
1. **`src/services/portfolio.py` & `src/services/portfolio_marks.py` & `src/quant/risk_measures.py`:**
   * Add a **14-day nonlinear portfolio Monte Carlo ES99** function `portfolio_horizon_var_es(legs_spec, horizon_days=14.0, r=..., n_scenarios=5_000, seed=42)` in `src/quant/risk_measures.py` (or `src/services/portfolio_marks.py`):
     * Simulate correlated 1-factor (market) + idiosyncratic spot shocks per underlying over `horizon_days` (with tail correlation $\rho_{\text{tail}} = 0.85$ on market down-shocks, Student-$t$ with $\nu = 5$ degrees of freedom) and leverage-correlated IV shocks (matching `ScenarioPricer`'s shock model in WP-6).
     * Reprice every option leg in `legs_spec` nonlinearly via Black-Scholes (and linear legs linearly) at $t + \text{horizon\_days}$, aggregate portfolio P&L per scenario, and compute exact 99% VaR and 99% Expected Shortfall (`book_es_99_usd` $\ge 0$).
     * Keep `parametric_book_var_es` available as a secondary diagnostic field (`book_1d_linear_es_99_usd`) if desired, while `risk["book_es_99_usd"]` is strictly the **14-day nonlinear ES99** on the exact same horizon/units as `ScenarioPricer`.
   * In `PortfolioService.spread_option_legs(spread_id)`:
     * First check `InstrumentInfo` (extend `InstrumentInfo` with `spread_legs: tuple[tuple[str, int], ...] | None = None` populated from `OptionSpread.info["legs"]` in `src/adapters/book_market.py`), and fall back to Valkey `mft:spread_legs:<spread_id>`.
   * Track `unmodeled_positions: list[str]` in `PortfolioService.evaluate()`. Publish `"unmodeled_positions": unmodeled_positions` and `"has_unmodeled_positions": bool(unmodeled_positions)` inside `payload["risk"]`.
2. **`src/risk/gate_rules.py`:**
   * In `portfolio_reasons()`: if `portfolio.get("risk", {}).get("has_unmodeled_positions")` is `True`, return `["PORTFOLIO_HAS_UNMODELED_POSITIONS"]` (fail closed).
   * In `es_budget_check()`: if `incremental_es99_per_lot_usd <= 0.0` (or non-finite), fail closed with `["INVALID_INCREMENTAL_ES99"]` and `capped_qty = 0`.

**Required Tests:**
* `tests/quant/test_risk_measures.py` & `tests/services/test_portfolio_service.py`:
  * **Additivity / Parity Test:** A single 1-lot credit spread in `PortfolioService` produces `book_es_99_usd` on the same order of magnitude (within $\pm 25\%$, same 14d nonlinear units) as `ScenarioPricer.evaluate_combo(..., horizon_days=14.0).es_99_usd`.
  * **Short-Gamma / Beta-Hedged Test:** A delta-hedged short strangle or short put spread hedged with SPY shares (`beta_delta_shares ≈ 0`) still produces substantial positive `book_es_99_usd` (proving Gamma + idiosyncratic risk are captured).
  * **Unmodeled Spread Test:** An `OPTION_SPREAD` with no `info["legs"]` and no Valkey entry sets `has_unmodeled_positions = True` and causes `PreTradeGate` to reject new entries.
  * **Zero/Negative Incremental ES99 Test:** `es_budget_check(..., incremental_es99_per_lot_usd=0.0)` rejects with `INVALID_INCREMENTAL_ES99`.

---

### WP-4 (P0): Crash/Restart Reconciliation, Overlapping-Leg Exits & Realized P&L Idempotency

**Problem:**
1. `VrpEngine.start()` loads `self.book` from Valkey without reconciling against open broker positions or open DAY orders; `NautilusOrderGateway._client_ids` is in-memory only, so orphaned DAY orders after a crash are ignored on fill.
2. `leg_mismatches()` sums all trades in `self.book` before excluding spread-level matched trades, causing overlapping strikes to falsely block exits (`POSITION_NOT_CONFIRMED_IN_CACHE`).
3. `PortfolioService.restore()` + `evaluate()` can double-count same-day realized P&L across intraday restarts.
4. `NautilusOrderGateway._client_ids` grows unboundedly over multi-week runs.

**Required Changes:**
1. **`src/strategies/vrp_core.py`:**
   * Fix `leg_mismatches` / `exit_confirmed`: exclude trades already confirmed at the spread level (`abs(net_position_qty(spread_key) + open_lots) < POSITION_TOLERANCE`) before computing leg-level `expected_leg_positions()`.
2. **`src/strategies/vrp_engine.py` & `src/adapters/order_gateway.py`:**
   * Extend `OrderGateway` protocol with `open_order_keys() -> list[str]` (and `NautilusOrderGateway` implementation using `ports.orders_open()`).
   * In `VrpEngine.start()`:
     * Cancel any pre-existing open orders returned by `gateway.open_order_keys()` on startup (logged as warnings) so no orphaned DAY orders work untracked after a crash.
     * Reconcile `self.book` against `market.net_position_qty`: log a `CRITICAL` warning and set `self._reconciliation_mismatch = True` (blocking new entries via `entry_block_reason`) if any persisted trade in `self.book` has no matching broker position (spread or leg level) when `paper_routing_enabled=True`.
   * Track consecutive `POSITION_NOT_CONFIRMED_IN_CACHE` counts per spread; if $\ge 2$ cycles, set a `book_reconciliation_alert` flag in `VRP_LATEST` and log `ERROR`.
   * Prune finalized `order_key` entries from `NautilusOrderGateway._client_ids` via a `release(order_key)` hook called from `VrpEngine.finalize()`.
3. **`src/services/portfolio.py` & `src/services/portfolio_tracking.py`:**
   * Make `SleeveTracker.restore()` idempotent across intraday restarts: persist `session_realized_usd` alongside `realized_carry_usd` and `session_date` (NY date). On same-day restart, do not add `session_realized_usd` on top of an already-carried `realized_carry_usd`—use `base_carry_usd = max_or_prior_day_carry` so `realized_pnl_usd` does not jump on restart.

**Required Tests:**
* `tests/strategies/test_vrp_core.py`:
  * Trade A (confirmed at spread level) + Trade B (confirmed at leg level) sharing a strike both return `exit_confirmed(...) == True`.
* `tests/strategies/test_vrp_engine.py`:
  * `start()` cancels orphaned open orders from the gateway.
  * `start()` with a persisted trade missing from broker positions blocks new entries with `BOOK_RECONCILIATION_MISMATCH`.
  * `finalize()` releases the order key from `NautilusOrderGateway._client_ids`.
* `tests/services/test_portfolio_service.py`:
  * Calling `evaluate()` $\to$ `restore()` $\to$ `evaluate()` on the same NY trading day with unchanged `closed_realized_usd = 500.0` reports `realized_pnl_usd == 500.0` (not `1000.0`).

---

### WP-5 (P1): Execution Hardening — Winning Exit Floor, Arrival Mid, Tick Sizing & Adverse Move

**Problem:**
1. `VrpEngine.manage_working_orders()` cancels `EXIT` orders when `mid <= 0.0`, stranding winning mid-cap spreads with wide wing quotes.
2. `PassiveComboExecutor` overwrites `clip.net_mid_price` on every step, erasing true arrival-price implementation shortfall.
3. Hardcoded `round(..., 2)` violates non-Penny Pilot mid-cap `$0.05` / `$0.10` option tick increments (causing IBKR Error 110 rejections).
4. Symmetric `abs(spot - ref_spot) > 0.25%` cancels/reprices on *favorable* moves and churns constantly on 45%-vol mid-caps.

**Required Changes:**
1. **`src/strategies/vrp_engine.py`:**
   * In `manage_working_orders()`: only cancel on `mid <= 0.0` when `wo.purpose == ENTRY`. For `wo.purpose == EXIT`, clamp `mid = max(mid, min_exit_tick)` (e.g. `0.05`) and `natural = max(natural, mid)` so winning spreads can still close cleanly.
2. **`src/execution/passive_combo.py`:**
   * Add `arrival_mid_price: float` and `arrival_natural_price: float` fields to `ChildClipOrder` (immutable after creation). Compute logged implementation shortfall against `arrival_mid_price`, while stepping logic uses `current_mid_price` / `current_natural_price`.
   * Add a `tick_size: float = 0.01` parameter (or standard OCC tick rule helper `round_to_tick(price, tick_size, side)` that rounds defensively—up for `CREDIT_OPEN`, down for `DEBIT_CLOSE`—with `$0.05` for non-penny options $< \$3.00$ and `$0.10` for $\ge \$3.00$, or explicit `instrument.price_increment`).
   * Scale `adverse_spot_move_pct` (or allow passing per-underlying `rv_20d` so threshold is $\max(\text{cfg.adverse\_spot\_move\_pct}, 0.15 \cdot \sigma_{\text{daily}})$) and support directional adverse-move checks when combo delta sign is known.
   * Ensure `create_parent_plan()` marks only the active child clip `WORKING` (or document/enforce single-clip registration consistent with `VrpEngine`).

**Required Tests:**
* `tests/strategies/test_vrp_engine.py`:
  * Working `EXIT` order with `mid = -0.02` is **not** cancelled; working `ENTRY` order with `mid = -0.02` **is** cancelled (`CANCELLED_INVALID_MID`).
* `tests/execution/test_passive_combo.py`:
  * After 2 limit steps with changing `new_mid_price`, `clip.arrival_mid_price` remains equal to the original creation mid and `record_fill()` logs shortfall vs. `arrival_mid_price`.
  * `round_to_tick` with `tick_size=0.05` rounds `1.23` $\to$ `1.25` for `CREDIT_OPEN` and `1.20` for `DEBIT_CLOSE` (never worse than intended limit).

---

### WP-6 (P1): `ScenarioPricer` BSM Carry, Round-Trip Friction, Crash Vol & CRN

**Problem:**
1. Passing `r = r - q` without adjusting $S \to S e^{-qT}$ omits the $e^{-qT}$ discount factor in Black-Scholes.
2. Charging only 1-way slippage/commission when `horizon_days < combo.ttm_days` overstates Net EV on wide mid-cap spreads.
3. Mild Gaussian vol-of-vol (`0.35 * rv * sqrt(dt) * z_vol`) + static sticky-strike IV severely understates crash `es_99_usd`.
4. Advancing `self.rng` per candidate adds MC ranking noise between spreads.

**Required Changes (`src/risk/scenario_pricer.py`):**
1. **Exact BSM Carry:** Price options using the exact Merton dividend-yield formulation ($S_{\text{adj}} = S e^{-q T_{\text{rem}}}$ with discount rate $r$, or equivalent forward-based Black-76 $e^{-r T_{\text{rem}}} \cdot \text{Black}(F, K, \sigma, T_{\text{rem}})$). Verify against `src/quant/black_scholes.py` to `< 1e-6`.
2. **Round-Trip Friction:** When `horizon_days < combo.ttm_days`, deduct **entry + exit** commissions (`2 * commission_per_leg * n_legs`) and **entry + exit** spread crossing (`2 * slippage_fraction * half_spread`, or configurable `exit_slippage_fraction` defaulting to `slippage_fraction`). When held to expiry (`horizon_days >= combo.ttm_days`), deduct entry friction only.
3. **Fat-Tailed Spot + Realistic Crash Vol Shock:**
   * Replace pure Gaussian spot returns with standardized Student-$t$ ($\nu = 5$, scaled to unit variance $\sqrt{(\nu-2)/\nu}$) so 3$\sigma$–5$\sigma$ mid-cap gaps appear in the 99% tail.
   * Model horizon IV shock with both leverage effect and sticky-delta skew slide:
     $$\Delta \sigma_i = -\beta_{\text{lev}} \cdot r_S + \eta_{\text{vol}} \sqrt{\Delta t}\, Z_{\text{vol}} + \text{skew\_slope} \cdot \Delta k_i$$
     (e.g., $\beta_{\text{lev}} \approx 0.6\text{–}0.8$ so a $-15\%$ mid-cap drop lifts IV by $+9\text{–}15$ vol pts, floored at `0.05`).
4. **Common Random Numbers (CRN):** Draw scenario shocks from a deterministic per-call local `np.random.default_rng(self.seed)` (or pre-generated antithetic/Sobol shock matrix) so two candidates evaluated on the same underlying/horizon face the exact same market paths.

**Required Tests (`tests/risk/test_scenario_pricer.py`):**
* **BSM Carry Golden Check:** For $S=100, K=100, T=0.5, r=0.05, q=0.04, \sigma=0.30$, internal pricer matches `src/quant/black_scholes.py` to `< 1e-5`.
* **Round-Trip Friction Check:** Evaluating the same spread at `horizon_days=14` (`< ttm_days=30`) deducts strictly more friction than at expiry (`horizon_days=30`).
* **Crash Tail Monotonicity:** A 1-lot 5-wide OTM put spread has `es_99_usd` strictly larger under Student-$t$ + leverage vol shock than under pure Gaussian constant-vol, and bounded by `max_width * 100`.
* **CRN Determinism:** Evaluating Candidate A then B produces the exact same `net_ev_usd` and `es_99_usd` as evaluating B then A.

---

### WP-7 (P0/P1): Walk-Forward Backtester Truth & Live-Gate Parity

**Problem:**
`src/backtest/vrp_walkforward.py` throws away the actual day-$t+14$ option chain quotes and invents `exit_debit` via `0.35 * remaining_credit` with zero exit slippage, ignores 50% profit-target early exits, mixes trading days with calendar days, and bypasses `PreTradeGate`.

**Required Changes (`src/backtest/vrp_walkforward.py`):**
1. **Mark Exits Off Actual Option Chain Quotes:**
   * Track each open trade day-by-day from entry date until either:
     * **50% Profit Target:** `close_natural_debit <= entry_fill * (1 - profit_target_fraction)` (using actual day-$t$ quotes for the exact `short_strike` and `long_strike`), OR
     * **14-DTE / Horizon Time Stop:** calendar `(expiry_date - current_date).days <= min_exit_dte` (or calendar days held $\ge \text{HOLDING\_DAYS}$).
   * Compute exit debit from actual day-$t$ chain quotes (`short_ask - long_bid` for conservative natural exit, or `close_mid + exit_slippage_frac * (close_nat - close_mid)`). Only if a strike quote is missing on the exit date, fall back to BSM repricing using that day's `spot`, `atm_iv`, and remaining calendar `T` plus a conservative exit spread penalty, and increment a `fallback_exit_marks` counter in the summary dict.
2. **Realistic Trade Friction & Metrics:**
   * Report `sharpe_approx`, `profit_factor`, `avg_holding_calendar_days`, `exits_by_reason` (`PROFIT_TARGET`, `TIME_STOP`), `fallback_exit_marks`, and `total_friction_usd`.
3. **Gate Parity:**
   * Apply the same liquidity/spread checks (`max_leg_spread_pct`, `min_mid_credit`, `min_natural_credit`, `min_open_interest` when OI column is present) and delta/width selection (from WP-8 or configurable width) as live `VrpEngine`.

**Required Tests (`tests/backtest/test_vrp_walkforward.py`):**
* **Vol-Spike Loss Test:** Construct a synthetic 20-day chain where spot drops 5% (staying above short strike) while IV spikes +25 vol pts; verify the backtester records a **loss** (reflecting real exit quote expansion), NOT a fake +65% profit.
* **50% Profit Target Early Exit Test:** When spread quotes decay to $\le 50\%$ of entry credit on day 5, the trade exits on day 5 with reason `PROFIT_TARGET`.
* **Calendar Day Test:** Holding period respects calendar `dte <= min_exit_dte` across weekends rather than 14 trading days (~20 calendar days).

---

### WP-8 (P2): Delta-Targeted Mid-Cap Verticals & `RiskDaemon` IBKR Margin / $100K PM Cliff

**Problem:**
1. `vrp_core.py` calls `adjacent_verticals()`, creating destructive `$1`-wide spreads on `$35` mid-caps where friction eats 15%–25% of width.
2. `RiskDaemon` has zero visibility into IBKR `NetLiquidation`, `MaintMarginReq`, `ExcessLiquidity`, or the `$100K` Portfolio Margin revocation cliff, and hardcodes `SPY.ARCA` as a global halt key.
3. `daemon_block_reason()` does not reject far-future `evaluated_at_ns`, and `futures_check()` spams `FUTURES_CROSS_CHECK_UNAVAILABLE` on non-index mid-caps.

**Required Changes:**
1. **`src/scanner/verticals.py` & `src/strategies/vrp_core.py`:**
   * Add `delta_targeted_verticals(...)` (with configurable `short_delta_range=(0.15, 0.30)`, `min_width_usd=2.50`, `min_width_pct_spot=0.04`, `max_width_pct_spot=0.12`) alongside `adjacent_verticals` so mid-cap verticals have sensible risk/friction ratios.
2. **`src/risk/daemon_rules.py`:**
   * Add optional `account: Payload | None = None` to `RiskInputs` (reading e.g. `mft:account:latest` with `nlv_usd`, `maint_margin_usd`, `excess_liquidity_usd`, `cushion`, `fx_rate_used`, `ts_ns`).
   * Add config fields to `RiskDaemonConfig`:
     * `min_pm_nlv_usd: float = 105_000.0` (5% buffer above the $100K IBKR Portfolio Margin cliff)
     * `min_margin_cushion: float = 0.25` (halt if IBKR cushion $< 25\%$)
     * `require_account_snapshot: bool = False` (default `False` for backward-compatible tests; `True` in live paper/prod)
   * In `evaluate_risk()`: halt with `pm_nlv_buffer_breach` or `margin_cushion_breach` when triggered.
   * Add `global_surface_keys` vs per-underlying staleness note so multi-name mid-caps don't halt the entire sleeve when one mid-cap stops ticking.
3. **`src/risk/gate_rules.py`:**
   * In `daemon_block_reason()`: reject far-future `evaluated_at_ns` (`age_secs < -5.0` $\to$ `RISK_DAEMON_INVALID_TIMESTAMP`).
   * In `futures_check()`: accept `is_index_etf: bool = True` (or check underlying against configured index ETFs); skip futures cross-check cleanly without warning spam on single-name mid-caps.

**Required Tests:**
* `tests/scanner/test_verticals.py`: `$1`-spaced chain on a `$40` mid-cap skips `$1`-wide adjacent pairs and selects `>= $2.50`-wide pairs in the target delta band.
* `tests/risk/test_daemon_rules.py`: Account snapshot with `nlv_usd=102_000` triggers `pm_nlv_buffer_breach`; `cushion=0.18` triggers `margin_cushion_breach`.
* `tests/risk/test_gate_rules.py`: Far-future `evaluated_at_ns` returns `RISK_DAEMON_INVALID_TIMESTAMP`; mid-cap underlying does not emit `FUTURES_CROSS_CHECK_UNAVAILABLE`.

---

### WP-9 (P2): Two-Tier Mid-Cap Universe Rotator & Programmatic Event/HTB Protocol

**Problem:**
IBKR caps live streaming at 100 lines, whereas mid-cap MFT requires scanning 100–300 names and tracking earnings/ex-div/HTB programmatically rather than via a 942-byte static `events.yaml`.

**Required Changes:**
1. **`src/scanner/universe_rotator.py` (new pure module):**
   * Define `UniverseCandidate` (symbol, atm_iv, rv_20d, vrp_ratio, rr25_pct, median_rel_spread, min_oi, sector, has_open_position, earnings_within_window).
   * Implement `select_active_streaming_universe(candidates, *, max_lines=90, lines_per_symbol=8, max_sector_share=0.30, fixed_lines=4) -> RotationPlan`:
     * Always reserves streaming lines for **all open positions first** (never drop quotes for an open spread!).
     * Filters out candidates with earnings inside window, spread above cap, or OI below floor.
     * Ranks remaining candidates by composite VRP + liquidity score while enforcing `max_sector_share` cap.
     * Guarantees `total_lines <= max_lines` (strictly under IBKR's 100-line cap).
2. **`src/data/event_calendar.py`:**
   * Define a clean `CorporateEventProvider` protocol (`earnings_dates(symbol)`, `ex_dividends(symbol)`, `htb_borrow_rate(symbol)`) with the existing YAML loader as `YamlEventProvider` and a JSON/Parquet cache-backed provider `CachedEventProvider` ready for ThetaData/IBKR ingestion.

**Required Tests:**
* `tests/scanner/test_universe_rotator.py`:
  * 200 candidates + 3 open positions $\to$ all 3 open positions are retained, total lines strictly $\le 90$, sector cap enforced, earnings-blackout names excluded.
* `tests/data/test_event_calendar.py`:
  * `CachedEventProvider` merges earnings, ex-div, and HTB borrow rate cleanly and blocks short calls when `borrow_rate > max_htb_rate`.

---

## 3. Master Adversarial Test Checklist (Agent Must Verify)

Before declaring any Sprint complete, the agent must verify all of the following pass with zero warnings/skips:

```bash
pytest -q
```

And confirm zero `nautilus_trader` or `redis` imports leaked into pure modules:

```bash
python -c "
import pathlib, sys
bad = []
for p in [
    *pathlib.Path('src/quant').glob('*.py'),
    pathlib.Path('src/strategies/vrp_core.py'),
    pathlib.Path('src/risk/gate_rules.py'),
    pathlib.Path('src/risk/daemon_rules.py'),
]:
    txt = p.read_text()
    if 'nautilus_trader' in txt or 'import redis' in txt:
        bad.append(str(p))
assert not bad, f'Architecture violation in pure modules: {bad}'
print('Hexagonal boundary check: PASS')
"
```

---

## 4. Handoff Protocol (Returning Updates for Gemini Review)

After completing each Sprint (recommended checkpoints: **after WP-3**, **after WP-6**, **after WP-7**, **after WP-9**), have the agent generate a single review bundle file (`review_bundle_sprintX.txt`) using this command and upload it back to our chat:

```bash
python - << 'EOF'
import subprocess, pathlib

out = []
out.append("=== GIT STATUS ===")
out.append(subprocess.check_output(["git", "status", "-s"], text=True))
out.append("=== GIT DIFF --STAT ===")
out.append(subprocess.check_output(["git", "diff", "--stat"], text=True))
out.append("=== PYTEST OUTPUT ===")
res = subprocess.run(["pytest", "-v"], capture_output=True, text=True)
out.append(res.stdout + res.stderr)
out.append(f"PYTEST EXIT CODE: {res.returncode}")
out.append("=== FULL GIT DIFF (src/ & tests/) ===")
out.append(subprocess.check_output(["git", "diff", "src/", "tests/"], text=True))

pathlib.Path("review_bundle.txt").write_text("\n".join(out))
print("Wrote review_bundle.txt", pathlib.Path("review_bundle.txt").stat().st_size, "bytes")
EOF
```

In addition, instruct the agent to include a brief 5-line summary per WP stating:
1. Exact files modified/added.
2. Any existing test assertions that were updated and why.
3. Any assumption or edge case that needs reviewer sign-off.