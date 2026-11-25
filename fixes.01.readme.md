# Fixes 01 — Logic fixes

**Date:** 2026-11-24 · **Base:** commit `14e9d28` (`ic`) · **Scope:** the 11 main logic issues found in the code review of the initial commit.

Items from the review's "smaller notes" were left alone unless one of these fixes depended on them (see [Not addressed](#not-addressed)).

> **Status:** working-tree changes, not committed. Per the project constraints at the time, nothing was compiled, installed or run. The code was checked by static review only (see [Verification](#verification)). NautilusTrader, the IB adapter, `fast_vollib` and Valkey are assumed to behave as the existing code already expects.

## At a glance

| # | Issue | Risk before the fix | Fix (short) | Main files |
|---|---|---|---|---|
| 1 | Regime gate fed VIXY/VIXM ETF prices | Gate meaningless: permanently open or closed | Real `^VIX.CBOE` / `^VIX3M.CBOE` levels via 1-minute bars | `run_paper.py`, `regime_actor.py`, `risk_daemon.py` |
| 2 | Spread orders pointed the wrong way | Opposite position, or never fills | Ratios +1 sold leg / −1 wing; SELL to open, BUY to close | `vrp_income.py` |
| 3 | Risk halt flag failed open | A dead daemon's last "0" keeps approving trades | TTL'd keys, heartbeat age check, stale inputs block | `risk_daemon.py`, `pre_trade_gate.py` |
| 4 | Per-lot risk divided twice | Tail risk under-counted; riskier trades preferred | Per-lot figures used as-is; size-aware ranking | `vrp_income.py` |
| 5 | Orders never managed after submission | Stacked GTC orders, $0.00 repricing, no exits | Order lifecycle, one clip at a time, exits, persisted book | `vrp_income.py`, `passive_combo.py` |
| 6 | Rate limit consumed by scoring | Best candidate starved of order budget | Throttle counts real submissions only | `pre_trade_gate.py`, `vrp_income.py` |
| 7 | Realized vol computed on the wrong bars | RV too low / VRP inflated; event-loop stalls | Daily-only history, cached percentile, daily session roll, date-aligned correlation | `realized_vol_actor.py` |
| 8 | SMA/RSI mixed daily and 1-minute bars | Trend and RSI meaningless after start-up | Daily bars only, de-duplicated | `surface_metrics_actor.py` |
| 9 | Portfolio marks approximate or wrong | Wrong P&L, Greeks and ES budget | Smile IV per strike, live mids, own-price marks, spread decomposition | `portfolio_actor.py` |
| 10 | Futures cross-check circular | Divergence ≈ 0 by construction | Same-minute SPX/SPY ratio; "unavailable" instead of anchored | `futures_actor.py`, `pre_trade_gate.py`, `ui/app.py` |
| 11 | No-op or noisy checks | SVI fit, OI and drawdown never enforced; audit-log flood | RMSE gate, real OI, drawdown snapshot, stable reason codes | several |

---

## Detailed fixes

### 1. Regime gate used ETF prices instead of the VIX term structure

**Problem.** [scripts/run_paper.py](scripts/run_paper.py) wired `VIXY.ARCA` / `VIXM.ARCA` into `MacroRegimeActor`, which compares `vix / vix3m` with `1.00`. ETF prices reflect their launch level and accumulated roll decay, not the shape of the VIX curve. The ratio therefore says nothing about contango or backwardation, and the gate would sit permanently on one side. The risk daemon had the same flaw as a fallback (`vixy_price / vixm_price`).

**Fix.**
- `run_paper.py` now loads the cash indices `^VIX.CBOE` and `^VIX3M.CBOE` (IB simplified symbology), plus `^SPX.CBOE` for fix 10.
- `MacroRegimeActorConfig` gained `use_quotes` and `use_bars` (both default `True`):
  - Cash indices publish levels rather than two-sided quotes, so the actor also subscribes to 1-minute LAST bars and uses the bar close as the level.
  - When both sources are active, the newest observation (by `ts_event`) wins.
  - `run_paper.py` sets `use_quotes=False, use_bars=True`.
- The daemon's VIXY/VIXM ratio fallback was removed.
- The daemon also halts with `stale_regime_feed` when the regime payload reports `is_stale: true`. The actor republishes every 15 s, so a fresh payload can still carry stale index levels.
- The UI regime page text now describes the VIX / VIX3M indices.

**Requires:** CBOE index market-data permissions on the IB account. Without them the regime signal never arrives and the system stays halted (fail-closed).

### 2. Spread orders pointed the wrong way

**Problem.** `_route_atomic_spread_order` built the `OptionSpread` with ratio −1 on the leg being sold and +1 on the wing, then submitted `OrderSide.SELL` at a positive credit. Selling a combo reverses every leg. A bull put 700/695 would have *bought* the 700 put and *sold* the 695 put (a bear put spread), at a price whose sign contradicts the combo's value. This was hidden only because routing was disabled.

**Fix.** One sign convention, used everywhere:

| Spread definition | Spread value | Open | Close |
|---|---|---|---|
| +1 × leg sold at entry, −1 × protective wing | Σ ratio × price > 0 (a credit) | **SELL** at +credit | **BUY** at +debit |

Example, bull put 700/695: legs `(700P, +1)` and `(695P, −1)`. Selling the spread sells the 700P and buys the 695P; buying it back reverses both legs.

Natural prices follow the side:
- Open: bid(700P) − ask(695P).
- Close: ask(700P) − bid(695P).

All prices stay positive, which matches `PassiveComboExecutor`'s conventions.

### 3. Risk halt flag failed open

**Problem.**
- The daemon wrote `mft:risk:halted` without expiry, and the gate only checked that it equalled `"0"`. If the daemon died, its last `"0"` stayed forever and trades kept being approved.
- Several gate inputs were accepted at any age.
- A missing portfolio snapshot read as zero tail risk.

**Fix.**
- **Daemon:**
  - `mft:risk:halted` and `mft:risk:status` are written with a TTL of `max(state_ttl_sec, 3 × poll_interval_sec)` (default 10 s), including on the error path. A dead or hung daemon now looks "missing", which the gate treats as halted.
  - Error-path writes are wrapped so a Valkey outage doesn't kill the daemon loop.
  - New environment variable: `RISK_STATE_TTL_SEC`.
- **Gate — daemon check.** The new `PreTradeGate.risk_daemon_block_reason()` requires all three of:
  - `mft:risk:halted == "0"`.
  - A status whose `evaluated_at_ns` is at most `max_risk_status_age_secs` (30 s) old.
  - `status.halted is False`.

  With no Valkey client the result is `RISK_DAEMON_UNREACHABLE`.
- **Gate — input freshness:**
  - Regime, surface, surface-metrics and portfolio snapshots older than `max_input_age_secs` (120 s, by their `ts_ns`) block with `*_MISSING_OR_STALE` reasons.
  - A missing or stale `mft:portfolio:risk` now blocks (`PORTFOLIO_RISK_MISSING_OR_STALE`) instead of counting as $0 of ES.
- **Strategy:**
  - Working **entry** orders are cancelled as soon as the daemon check fails. The check is re-run at most once per second on underlying ticks.
  - Exit orders keep working because they reduce risk.

### 4. Per-lot risk was divided twice; ranking favoured riskier trades

**Problem.** `ScenarioPricer` already returns EV and ES99 per 1 lot, but the strategy divided both by `recommended_contracts` again. The gate therefore under-counted tail risk. Even with that fixed, ranking by per-lot EV still prefers wider, riskier spreads.

**Fix.**
- The gate receives the pricer's figures unchanged:
  - `per_lot_es99 = max(-es_99_usd, 1.0)`, the same floor the pricer uses for sizing.
  - `per_lot_ev = net_ev_usd`.
- Gate legs carry the requested quantity via `dataclasses.replace(...)`. Previously the code mutated a frozen dataclass with `object.__setattr__`.
- Candidates are ranked by `(gate_approved, expected_ev_usd, ev_to_es99)`, where `expected_ev_usd = per_lot_ev × approved_qty`. Sizing is limited by the tail budget, so this maximises expected profit per unit of ES budget. Example:
  - A: $10 EV and $100 ES per lot → 10 lots → $100 expected.
  - B: $15 EV and $500 ES per lot → 2 lots → $30 expected.
  - A now wins; per-lot ranking picked B.
- New candidate row fields: `dte`, `ev_to_es99`, `recommended_qty`, `expected_ev_usd`.

### 5. Orders were never managed after submission

**Problem.**
- No fill, cancel or reject handlers; no cancel or modify; no exits (`profit_target_fraction` and `min_exit_dte` were unused).
- The executor was "repriced" with mid = natural = 0.0, which would have produced $0.00 limits had its actions been applied.
- Every approval submitted all clips at once as GTC orders, and resting orders didn't count against the ES budget.
- Shadow mode registered a new clip plan every 15 s, which then produced spurious adverse-move warnings.

**Fix — entries.**
- At most **one working order** strategy-wide. Each approval submits a **single clip** of at most `max_child_clip_lots` (8 lots); the next clip needs a fresh gate approval.
- After any fill, new entries wait until `mft:portfolio:risk` has a `ts_ns` newer than the fill (`WAITING_PORTFOLIO_REFRESH`). That way the book ES includes the new position before more budget is used.
- Orders use `TimeInForce.DAY`, and `on_stop` cancels anything still working.
- Entries are refused when DTE ≤ `min_exit_dte` (`DTE_INSIDE_EXIT_WINDOW`), since the time stop would close them immediately.
- Shadow mode builds a *preview* plan (`register=False`), so the executor no longer accumulates clips that were never sent.

**Fix — order lifecycle.**
- The order context is registered *before* `submit_order`, because denials can arrive synchronously.
- `on_order_filled`:
  - Updates the trade book: entries set a quantity-weighted entry credit, exits reduce open lots.
  - Records the fill time and persists the book.
  - When the order completes, logs slippage through `PassiveComboExecutor.record_fill` (partial fills use the filled quantity).
- `on_order_canceled`, `on_order_expired`, `on_order_rejected` and `on_order_denied` release the working slot and mark the clip terminal.
- `on_order_modify_rejected` cancels the order, because the executor has already stepped its limit.
- A sweep finalises orders that reached a terminal state without a matching event.

**Fix — repricing.**
- On every underlying quote and every timer, each working order is repriced with the live spread mid/natural computed from its legs' quotes.
- `PassiveComboExecutor.on_market_tick` is filtered by `parent_order_id`, so prices only apply to the right spread.
- `MODIFY_LIMIT` → `modify_order`. `CANCEL_REPLACE` (adverse spot move) and `CANCEL_FINAL` (slippage cap) → `cancel_order`.
- Orders pending an update or cancel are skipped, and an order whose spread mid is no longer positive is cancelled.

**Fix — exits.** On each timer, every open trade in the book is checked against two triggers:
- **Time stop:** DTE ≤ `min_exit_dte` (default 14).
- **Profit target:** live close mid-debit ≤ entry credit × (1 − `profit_target_fraction`) (default 50%).

Exits are not gated by `PreTradeGate`, because they reduce risk. They are only routed when the cache confirms the short exposure, either as a spread-level position of −open lots or as leg positions matching the book. Otherwise the signal shows `POSITION_NOT_CONFIRMED_IN_CACHE`, because buying back a spread you don't hold would open the opposite position.

Exits also work one clip at a time, and expired spreads are dropped from the book.

**Fix — persistence.**
- The trade book (legs, open lots, entry credit, expiry) is saved to `mft:strategy:vrp:book:<underlying>` after each fill and restored on start.
- Each spread's legs are registered under `mft:spread_legs:<spread id>`, so the portfolio actor can decompose spread positions (fix 9).

**UI.** The strategy page shows working orders, the spread book and exit signals.

**Executor API additions (backward compatible):**
- `create_parent_plan(..., register=True)`
- `on_market_tick(..., parent_order_id=None)`
- `mark_clip_status()`
- `discard_clip()`
- `record_fill(..., filled_lots=None)`

`scripts/run_backtest.py` keeps working unchanged.

### 6. Rate limiter consumed by scoring

**Problem.** `evaluate_combo` recorded a timestamp for every approved evaluation, so scoring the first six candidates used up the six-per-minute order budget.

**Fix.** `evaluate_combo` only *checks* the throttle. The strategy calls the new `PreTradeGate.record_order_submission()` when it actually routes an entry order.

### 7. Realized vol computed on the wrong bars

**Problem.**
- The catalog warm start loaded every bar type for an instrument, including the 1-minute SPY bars from `sync_historical_data.py`, so "daily" RV could be computed on minute bars.
- The RV percentile recomputed one Yang–Zhang estimate per history row, for 8 symbols, on every 15 s timer, blocking the event loop.
- The live intraday bar only reset on DAY bars, which were never subscribed, so it accumulated across days.
- Correlations paired returns by row position.

**Fix.**
- The warm start requests `"<id>-1-DAY-LAST-EXTERNAL"` bars only (the ThetaData parquet fallback is unchanged).
- The percentile uses the trailing `trading_days` (252) rolling 20-day RVs of **completed** history. It is cached and rebuilt only when the history changes, about once a day.
- The live session:
  - Uses regular-hours prints only (09:30–16:00 America/New_York, weekdays).
  - Rolls into the daily history when the next session's first print arrives.
  - Is appended for RV only if it is newer than the last history bar.
- The history cap became `max(300, trading_days + 25)`.
- Cross-asset correlation aligns returns by New York trading date (inner join) before taking the last 20 days.

### 8. SMA/RSI mixed timeframes

**Problem.** The indicators were warmed with daily bars and then fed 1-minute bars.

**Fix.** `SurfaceRegimeMetricsActor` subscribes to `1-DAY-LAST` bars, and only 1-day bars of the underlying update SMA20, SMA50 and RSI14. Bars at or before the last one already applied are ignored, for example those overlapping the catalog warm start.

### 9. Portfolio marks were approximate or wrong

**Problem.**
- Every option was marked at the flat ATM IV.
- Any non-option position (MES, other ETFs) was marked at the SPY price.
- The underlying's venue was hard-coded as `ARCA`, and other underlyings fell back to SPY data.
- The P&L attribution baseline never reset.

**Fix.**
- **Options:**
  - IV comes from the underlying's SVI slice at the option's own strike.
  - Same expiry: `k = ln(K / F_surface)`.
  - Other expiries: the option's own forward, with `k` rescaled by `sqrt(T_surface / T_option)` onto the calibrated slice. This is an approximation.
  - IV fallbacks: surface ATM IV → that underlying's RV20 → benchmark ATM IV.
  - Marks use the live mid when the quote is valid and at most `max_mark_quote_age_secs` (300 s) old, otherwise the model price.
- **Underlying lookup by symbol.** Spot comes from any `mft:surface:*` key, then the RV universe, then a live quote for a cached non-option instrument with that symbol. Options whose underlying can't be priced are flagged `UNPRICED_UNDERLYING` (entry price, no Greeks) instead of being priced off SPY.
- **Non-options:**
  - Marked at their own live mid → actor spot → entry price.
  - Beta delta is notional-weighted in SPY shares; 1 MES ≈ 5 × price / SPY spot ≈ 50 shares.
- **Spread positions:**
  - Decomposed into option legs using `mft:spread_legs:<spread id>`: P&L at the spread level, Greeks and stress per leg.
  - An unregistered `OptionSpread` is marked at its own mid or entry price with no Greeks (`UNMODELED_SPREAD`), rather than at the SPY price.
- Each position row now carries `mark_source`: `LIVE_MID`, `MODEL`, `MIXED`, `ACTOR_SPOT`, `ENTRY_PRICE`, `UNPRICED_UNDERLYING` or `UNMODELED_SPREAD`.
- **Attribution** rebases when the New York trading date changes or any open position's quantity changes, because a Greeks explain is only valid while the book is unchanged. `attribution.since_ts_ns` shows the baseline time.

### 10. Futures cross-check was circular

**Problem.** `^SPX` was never subscribed, so the MES→SPY scale was frozen at the first reading as `MES / parity_forward`. The "divergence" was therefore ≈ 0 at start-up by construction. The 10× fallback is off by roughly 20–30 bps, which is the size of the 20 bps gate.

**Fix.**
- **Ratio:**
  - `^SPX.CBOE` and `SPY.ARCA` 1-minute bars are subscribed, and the SPX/SPY ratio is taken from closes stamped at the same minute, so both prices refer to the same moment.
  - The ratio stays valid for `max_ratio_age_secs` (900 s) after the last aligned pair.
  - The anchored ratio and the `DEFAULT_10X` fallback were removed, along with the `default_index_to_etf_ratio` config parameter.
- **Staleness:**
  - SPY or MES quotes older than `max_quote_age_secs` (90 s) → nothing is published, so the gate sees a stale snapshot.
  - A surface older than `max_surface_age_secs` (120 s), or an unknown MES expiry → the cross-check is unavailable.
- **Comparison:** the futures-implied forward is projected from the surface's own spot, so the check compares carry rather than spot drift since the fit.
- **Payload:**
  - Unavailable values are published as `null`.
  - New fields: `cross_check_valid` and `ratio_age_secs`.
  - `ratio_source` is `LIVE_SPX_SPY_1MIN_BARS` or `UNAVAILABLE`.
- **Gate:** `FUTURES_PARITY_DIVERGENCE` applies only when the snapshot is fresh and `cross_check_valid` is true. Otherwise the gate adds the warning `FUTURES_CROSS_CHECK_UNAVAILABLE`.
- **UI:** the futures page renders `null` as "N/A" (it previously would have crashed) and shows a banner while the cross-check is unavailable.

### 11. Other checks were no-ops or noisy

- **SVI fit quality.** The gate blocks with `SVI_FIT_POOR` when `rmse_vol_pts` is missing or above `max_svi_rmse_vol_pts` (0.50 vol points).
- **Open interest:**
  - `FastVolActor` now publishes each leg's `open_interest` (`null` when unknown) and the slice `expiry_ns`.
  - The strategy takes OI from the surface, then from instrument `info["open_interest"]`, otherwise treats it as unknown. It never uses the hard-coded 2500.
  - `CandidateLegSpec.open_interest` defaults to `None`.
  - The gate blocks unknown OI with `OPEN_INTEREST_UNKNOWN` unless `GateConfig.allow_unknown_open_interest=True`, in which case it becomes a warning and the OI caps are skipped.
- **Drawdown input.** `PortfolioAnalyticsActor` now publishes `mft:positions:latest` (fields listed below):
  - Equity = allocated capital + realized P&L (carried across restarts) + unrealized P&L.
  - The high-water mark persists across restarts.
  - `drawdown_pct = (equity − HWM) / HWM × 100`.
  - The daemon halts when the snapshot is missing (`require_positions_snapshot`), older than `max_positions_age_sec` (120 s), or at or below `max_drawdown_pct` (−5%).
- **Transition flood:**
  - The daemon records stable `reason_codes` (no ages or values) alongside the human-readable `reasons`.
  - A transition is logged only when the halted state or the set of codes changes, so a stale feed no longer writes a transition every 2 s.
  - After a daemon error, the recovery evaluation is logged as a transition.

---

## Interface and data-contract changes

### Valkey keys

| Key | Producer → consumers | Change |
|---|---|---|
| `mft:risk:halted`, `mft:risk:status` | Daemon → gate, strategy, UI | Now expire (TTL, default 10 s); status has `reason_codes` |
| `mft:positions:latest` | **New producer:** PortfolioAnalyticsActor → daemon | `{allocated_capital_usd, equity, high_water_mark, drawdown_pct, realized_total_usd, unrealized_pnl_usd, open_positions, ts_ns}` |
| `mft:portfolio:risk` | PortfolioAnalyticsActor → gate, strategy, UI | New `sleeve` section, `positions[].mark_source` and `attribution.since_ts_ns`; `realized_pnl_usd` includes the carried realized P&L |
| `mft:surface:<id>` | FastVolActor → actors, gate, strategy, UI | New `expiry_ns` and `legs[].open_interest` |
| `mft:futures:latest` | FuturesContextActor → gate, UI | Nullable fields; new `cross_check_valid` and `ratio_age_secs` |
| `mft:strategy:vrp:latest` | Strategy → UI | New `entry_block_reason`, `working_orders`, `open_book` and `exit_signals`; candidate rows gain `dte`, `ev_to_es99`, `recommended_qty` and `expected_ev_usd` |
| `mft:strategy:vrp:book:<underlying>` | **New:** strategy (saved and restored by itself) | Persisted spread book |
| `mft:spread_legs:<spread id>` | **New:** strategy → PortfolioAnalyticsActor | Leg ids and ratios of each traded spread |

### New configuration settings

| Where | Setting | Default | Purpose |
|---|---|---|---|
| `GateConfig` | `allow_unknown_open_interest` | `False` | Block (`False`) or warn (`True`) when OI is unknown |
| `GateConfig` | `max_svi_rmse_vol_pts` | `0.50` | Maximum SVI fit RMSE (vol points) |
| `GateConfig` | `max_input_age_secs` | `120.0` | Maximum age of actor snapshots |
| `GateConfig` | `max_risk_status_age_secs` | `30.0` | Maximum age of the daemon's last evaluation |
| `RiskDaemonConfig` | `state_ttl_sec` (env `RISK_STATE_TTL_SEC`) | `10.0` | TTL of the halted/status keys (at least 3 × poll interval) |
| `RiskDaemonConfig` | `max_positions_age_sec` | `120.0` | Maximum age of the sleeve snapshot |
| `RiskDaemonConfig` | `require_positions_snapshot` | `True` | Halt when the sleeve snapshot is missing |
| `MacroRegimeActorConfig` | `use_quotes`, `use_bars` | `True`, `True` | Level sources (`run_paper.py` uses bars only) |
| `FuturesContextActorConfig` | `max_quote_age_secs`, `max_surface_age_secs`, `max_ratio_age_secs` | `90`, `120`, `900` | Staleness limits (`default_index_to_etf_ratio` removed) |
| `PortfolioAnalyticsActorConfig` | `max_mark_quote_age_secs` | `300.0` | Maximum quote age for live marks |

### New reason and status codes

- **Gate reasons:**
  - `RISK_DAEMON_UNREACHABLE`, `RISK_DAEMON_STALE(...)`
  - `MACRO_REGIME_MISSING_OR_STALE`
  - `SVI_SURFACE_MISSING_OR_STALE`, `SVI_SURFACE_NOT_ARB_FREE` (these two replace `SVI_SURFACE_NOT_ARB_FREE_OR_MISSING`), `SVI_FIT_POOR(...)`
  - `SURFACE_METRICS_MISSING_OR_STALE`
  - `OPEN_INTEREST_UNKNOWN(...)`
  - `PORTFOLIO_RISK_MISSING_OR_STALE`
  - Warning: `FUTURES_CROSS_CHECK_UNAVAILABLE`.
  - Added by the strategy: `DTE_INSIDE_EXIT_WINDOW(...)`.
- **Daemon reason codes:** `stale_regime_feed`, `missing_positions_snapshot`, `stale_positions_snapshot` and `daemon_error` are new; the others are unchanged.
- **Strategy actions:** `ENTRY_DEFERRED(ORDER_WORKING | NO_UNDERLYING_QUOTE | WAITING_PORTFOLIO_REFRESH)` and `ORDER_SUBMIT_FAILED`, alongside the existing `STAND_DOWN`, `SHADOW_SIGNAL_READY` and `LIVE_PAPER_ORDER_ROUTED`.
- **Exit statuses:** `SHADOW_SIGNAL`, `EXIT_ORDER_ROUTED`, `ORDER_WORKING_ON_SPREAD`, `NO_LIVE_QUOTES`, `NOTHING_TO_CLOSE`, `POSITION_NOT_CONFIRMED_IN_CACHE` and `ORDER_SUBMIT_FAILED`.

## Operational impact — read before running

1. **Every candidate is blocked with `OPEN_INTEREST_UNKNOWN` until an OI source is wired.** IB instrument data carries no open interest. Either supply it (surface legs or instrument `info`) or set `GateConfig(allow_unknown_open_interest=True)` to warn instead.
2. **CBOE index data is required** for `^VIX`, `^VIX3M` and `^SPX`. Without it the regime gate stays closed (the daemon halts) and the futures cross-check stays unavailable.
3. **The daemon halts if PortfolioAnalyticsActor isn't running**, because `mft:positions:latest` is missing. Set `require_positions_snapshot=False` only if you deliberately run the daemon without the trading node.
4. **Resetting state:**
   - Delete `mft:positions:latest` to reset sleeve equity, the high-water mark and realized carry.
   - Delete `mft:strategy:vrp:book:SPY.ARCA` to clear the strategy's spread book.
5. **Slow daemon polling:** if you raise `RISK_POLL_INTERVAL_SEC` toward 30 s or more, also raise `GateConfig.max_risk_status_age_secs`, or the gate will report `RISK_DAEMON_STALE`.

## Known limitations of these fixes

- **Double-counted spreads:** if the venue reports a spread position **and** its leg positions, the portfolio counts both. The error is conservative: Greeks, ES and P&L are overstated, not understated.
- **Exit confirmation:** it compares leg positions with the book. It can report `POSITION_NOT_CONFIRMED_IN_CACHE`, leaving the exit to manual handling, when:
  - Other strategies or manual trades hold positions in the same options.
  - Two spreads share a strike.
- **Realized-P&L carry:** it assumes the NautilusTrader cache is in-memory, as configured in `run_paper.py`. Enabling a cache database would replay old closed positions and double count them.
- **Off-expiry smile:** IVs for expiries other than the calibrated slice use the √T moneyness rescale, an approximation.
- **Mid-session start:** if the actor starts mid-session, that day's live bar opens at the first print seen and is rolled into the RV history that way.

## Not addressed

- **`src/data/` is still missing.** The `data/` line in `.gitignore` also excludes `src/data/`, so `event_calendar`, `thetadata_chain_loader` and `catalog_loader` were never committed. Until the package is committed and the rule changed to `/data/`, none of these can import:
  - The gate and the strategy.
  - `run_paper.py`, `run_backtest.py` and `sync_historical_data.py`.
- **Smaller review notes left as they are:**
  - Only one put-call parity pair from the loaded strikes.
  - `config/sleeve.yaml` is never loaded, and its values drift from the code defaults.
  - Two different 25-delta skew definitions.
  - IV inversion passes `r − q` as `r`.
  - No common random numbers across Monte Carlo evaluations.
  - The pricer recommends about 1,000 lots when ES99 is near zero or positive; liquidity caps still bound the final size.
  - `SurfaceScanner` and `_bs_delta` are unused.
  - `scripts/verify_vol_pipeline.py` imports names that no longer exist.

## Verification

Done (static only; nothing was compiled, installed or run):
- Read-through of every diff; `git diff --check` is clean.
- Scan for f-strings that reuse their own quote type inside a replacement field (invalid on Python 3.11): none.
- No remaining references to removed names: `default_index_to_etf_ratio`, `_prev_reasons`, `DEFAULT_10X`, `vixy_price`, or the hard-coded `open_interest=2500`.

Suggested next checks:
1. **Shadow-mode run.** Start Valkey, the risk daemon, `run_paper.py` (shadow mode) and the UI, then confirm:
   - Regime levels arrive via bars.
   - The futures page shows `cross_check_valid: true` once SPX bars flow.
   - `mft:risk:transitions` only grows on real state changes.
   - Stopping the daemon switches the UI banner to "heartbeat missing" within about 10 s.
2. **Candidate rows.** `es99_per_lot` should equal the pricer's per-lot ES, and the reasons should list the expected blocks (for example `OPEN_INTEREST_UNKNOWN`).
3. **One-lot paper order.** Confirm that:
   - The IB combo is a credit bull put or bear call.
   - The book entry appears after the fill.
   - A halt cancels a working entry.
   - Exits fire on the time stop.
4. **Unit tests to add:**
   - Executor: `parent_order_id` filter, `register=False`.
   - Gate: stale inputs, unknown OI, throttle.
   - Daemon: TTL, transitions on code changes.
   - Strategy: spread price signs.

## Files changed

| File | Change |
|---|---|
| [scripts/run_paper.py](scripts/run_paper.py) | Index instruments, regime via bars, SPX for the futures actor |
| [src/actors/regime_actor.py](src/actors/regime_actor.py) | Bars and/or quotes as level sources |
| [src/actors/vol_actor.py](src/actors/vol_actor.py) | `open_interest` per leg, `expiry_ns` in the surface payload |
| [src/actors/futures_actor.py](src/actors/futures_actor.py) | Rewritten: aligned SPX/SPY ratio, nullable cross-check, staleness limits |
| [src/actors/realized_vol_actor.py](src/actors/realized_vol_actor.py) | Rewritten: daily history, cached percentile, regular-hours session roll, date-aligned correlation |
| [src/actors/surface_metrics_actor.py](src/actors/surface_metrics_actor.py) | Daily-bar indicators with de-duplication |
| [src/actors/portfolio_actor.py](src/actors/portfolio_actor.py) | Rewritten: smile/live marks, own-price linear marks, spread decomposition, attribution rebase, sleeve snapshot |
| [src/execution/passive_combo.py](src/execution/passive_combo.py) | Preview plans, per-parent repricing, clip status helpers, partial-fill logging |
| [src/risk/pre_trade_gate.py](src/risk/pre_trade_gate.py) | Daemon heartbeat, stale inputs, SVI RMSE, OI handling, submission-based throttle, futures validity, portfolio fail-closed |
| [src/risk/risk_daemon.py](src/risk/risk_daemon.py) | TTL'd state, reason codes, positions snapshot, regime feed staleness, no VIXY/VIXM fallback |
| [src/strategies/vrp_income.py](src/strategies/vrp_income.py) | Rewritten: spread ratios, per-lot risk, ranking, order lifecycle, exits, persisted book |
| [ui/app.py](ui/app.py) | None-safe futures page, regime page text, strategy working orders / book / exit tables |
