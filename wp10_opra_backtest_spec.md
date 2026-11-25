# WP-10: Real OPRA Historical Chain Pipeline, Multi-Expiry Backtester & VRP Experiment Suite

> **Target Agent:** DeepSeek V4 Pro  
> **Baseline:** `main` (`v0.2-mft-hardened`, 712 tests passing)  
> **Starter Pack Target:** 6 months of `DECK.OPT` + `XBI.OPT` (`~$15–$17` of `$124.94` Databento credit), or 12 months of `DECK.OPT` (`~$9`).  
> **Critical Rule:** `pytest` must **never** make live network calls or spend Databento credit. All parsing, snapping, IV inversion, multi-expiry backtesting, and ablation tests must run 100% offline on synthetic/mocked DataFrames.

---

## 0. Non-Negotiable Operating Rules

1. **Preserve Hexagonal Architecture:** Pure chain-building and backtest logic (`src/data/opra_chain_builder.py`, `src/backtest/vrp_walkforward.py`) must remain free of `nautilus_trader`, `redis`, and live `databento` HTTP calls. Only `scripts/fetch_opra_databento.py` imports `databento` to fetch/cache `.dbn.zst` files.
2. **Fail-Closed & Cost-Safe:** `scripts/fetch_opra_databento.py` must always check `client.metadata.get_cost(...)` for any uncached `.dbn.zst` files first, print an itemized USD cost table, and **abort without spending credit** unless `--confirm` is passed **and** `total_cost <= --max-cost-usd` (default `20.0`). Cached `.dbn.zst` files must be reused at `$0.00`.
3. **100% Backward Compatibility:** All 712 existing tests must continue to pass without weakening assertions. Single-expiry DataFrames without an `expiry` or `bid_size`/`ask_size` column must still work identically in `run_midcap_vrp_backtest()`.
4. **No Lookahead Bias:** `rv_20d` on date $t$ must be computed strictly from bars $\le t$ using `sanitize_ohlc()` + `yang_zhang_rv()` from `src/quant/realized_vol.py`.

---

## 1. Architecture & File Deliverables

| File | Type | Responsibility |
| :--- | :---: | :--- |
| `src/data/opra_chain_builder.py` | **New (Pure)** | Normalizes Databento `definition` + `cbbo-1m` DataFrames, extracts 15:45 ET daily NBBO snapshots, joins underlying OHLC + sanitized `rv_20d`, inverts BSM `iv`, computes `atm_iv`, and outputs the canonical backtest Parquet schema. |
| `scripts/fetch_opra_databento.py` | **New (CLI)** | Cost-checked, disk-cached Databento downloader (`OPRA.PILLAR` `definition` + `cbbo-1m`, plus `EQUS.MINI` `ohlcv-1d` if underlying bars aren't already cached locally) that calls `opra_chain_builder.py` to write `data/backtest/<SYMBOL>_opra_daily.parquet`. |
| `src/backtest/vrp_walkforward.py` | **Modified** | Adds multi-expiry (`expiry` column) entry selection & exact-expiry exit matching, optional `use_delta_targeted=True` mode using `WP-8` `delta_targeted_verticals()`, TOB size parity (`bid_size`/`ask_size`), configurable `min_vrp`, and cleans up the BSM fallback `exit_mid` reset nit. |
| `scripts/run_vrp_experiments.py` | **New (CLI)** | 5-experiment VRP ablation harness (Friction Frontier, TP vs. Time-Stop, Adjacent vs. Delta-Targeted, VRP Threshold Sweep, Cross-Symbol Comparison) with pure `run_ablation_suite()` helper + Markdown table & JSON export. |
| `tests/data/test_opra_chain_builder.py` | **New** | Offline unit & adversarial tests for OPRA definition filtering, 15:45 ET snapshot selection, no-lookahead `rv_20d`, IV inversion, and canonical schema output. |
| `tests/backtest/test_vrp_walkforward.py` | **Modified** | Adds multi-expiry isolation test, `use_delta_targeted=True` test, TOB size parity test, and BSM fallback friction accounting test. |
| `tests/backtest/test_vrp_experiments.py` | **New** | Fast offline test for `run_ablation_suite()` verifying all 5 experiment tables return valid schemas and non-empty metrics. |

---

## 2. Canonical Parquet Schema (`data/backtest/<SYMBOL>_opra_daily.parquet`)

Every row represents one standard (`multiplier == 100`) option contract's **15:45 ET NBBO snapshot** on one NY trading date:

| Column | Type | Notes |
| :--- | :--- | :--- |
| `date` | `str` (`YYYY-MM-DD`) | NY trading date (ISO). |
| `expiry` | `str` (`YYYY-MM-DD`) | Option expiration date (ISO). |
| `dte` | `int` | Calendar days `(expiry - date).days`. |
| `spot` | `float` | Underlying close on `date`. |
| `rv_20d` | `float` | Sanitized 20-day Yang-Zhang RV using bars $\le$ `date` (no lookahead). |
| `atm_iv` | `float` | Median IV of nearest-to-spot OTM/ATM options in the `[21, 45]` DTE slice (fallback to nearest available expiry). |
| `right` | `str` (`"P"` or `"C"`) | Normalized option right. |
| `strike` | `float` | Strike price in USD. |
| `bid` | `float` | 15:45 ET NBBO bid (`> 0`). |
| `ask` | `float` | 15:45 ET NBBO ask (`>= bid`). |
| `mid` | `float` | `0.5 * (bid + ask)`. |
| `bid_size` | `int` | 15:45 ET NBBO bid size (contracts). |
| `ask_size` | `int` | 15:45 ET NBBO ask size (contracts). |
| `iv` | `float` | BSM implied volatility inverted via `src/quant/black_scholes.py` (`NaN` rows dropped). |

---

## 3. Detailed Component Specifications

### Component A: `src/data/opra_chain_builder.py` (Pure Transformation Module)

Inspect `src/quant/black_scholes.py` and `src/quant/realized_vol.py` in full first to reuse their exact function signatures.

Implement these pure functions:

1. **`parse_opra_definitions(def_df: pd.DataFrame) -> pd.DataFrame`:**
   * Extracts `instrument_id`, `raw_symbol` (or `symbol`), `strike` (from `strike_price`), `expiry` (ISO date `YYYY-MM-DD` from `expiration`), `right` (`"C"` or `"P"` from `instrument_class`), and `multiplier` (from `contract_multiplier` or `unit_of_measure_qty`, defaulting to `100` if absent).
   * **Hygiene filters:** Keep only `right in ("C", "P")`, `strike > 0`, valid `expiry`, and standard contracts (`multiplier == 100`, excluding adjusted/mini OCC root symbols if detectable via non-standard multiplier). Deduplicate by `instrument_id` keeping the latest definition.

2. **`extract_daily_nbbo_snapshot(cbbo_df: pd.DataFrame, *, snap_hour_et: int = 15, snap_minute_et: int = 45, window_minutes: int = 30) -> pd.DataFrame`:**
   * Converts event/receive timestamp (`ts_recv` or `ts_event` or DatetimeIndex) to `America/New_York`.
   * Filters to valid two-sided quotes: `bid_px_00 > 0.0`, `ask_px_00 >= bid_px_00`, finite prices, within `[15:15, 15:45]` ET (configurable via `snap_hour_et`, `snap_minute_et`, `window_minutes`) on each NY trading date.
   * For each `(date, instrument_id)`, takes the **last** valid observation at or before `15:45 ET`.
   * Outputs columns: `date`, `instrument_id`, `bid`, `ask`, `mid`, `bid_size`, `ask_size`.

3. **`compute_daily_rv_table(bars_df: pd.DataFrame, window: int = 20) -> pd.DataFrame`:**
   * Takes daily OHLC bars (`date` or `ts_ns`, `open`, `high`, `low`, `close`), sorts chronologically, sanitizes with `sanitize_ohlc()`, and computes `rv_20d` as-of each date $t$ using strictly bars $\le t$ (via `yang_zhang_rv` / `trailing_rolling_rv`).
   * Returns `date` (ISO `YYYY-MM-DD`), `spot` (`close`), `rv_20d` (dropping initial warm-up dates where `rv_20d <= 0` or non-finite).

4. **`build_opra_backtest_chain(def_df: pd.DataFrame, cbbo_df: pd.DataFrame, bars_df: pd.DataFrame, *, risk_free_rate: float = DEFAULT_RISK_FREE_RATE, dividend_yield: float = 0.0, min_dte: int = 1, max_dte: int = 90) -> pd.DataFrame`:**
   * Joins `extract_daily_nbbo_snapshot(cbbo_df)` with `parse_opra_definitions(def_df)` on `instrument_id` and `compute_daily_rv_table(bars_df)` on `date`.
   * Computes `dte = (expiry - date).days`, keeps `min_dte <= dte <= max_dte`.
   * Computes BSM `iv` per row using `src/quant/black_scholes.py` (`implied_vols` or equivalent vectorized helper) with `T = dte / 365.25`, `r = risk_free_rate`, `q = dividend_yield`. Drops rows where `iv` is `None`/`NaN` or outside `[0.02, 4.00]`.
   * Computes `atm_iv` per `date`: median `iv` of the 4 strikes closest to `spot` (by `|strike - spot|`) in the expiration closest to 30 DTE (preferring `[21, 45]` DTE, else nearest available expiry).
   * Returns a clean DataFrame sorted by `["date", "expiry", "right", "strike"]` matching Section 2's canonical schema.

---

### Component B: `scripts/fetch_opra_databento.py` (Cost-Safe CLI Downloader)

1. **CLI Arguments (`argparse`):**
   * `--symbols`: one or more root symbols (e.g. `--symbols DECK XBI`).
   * `--start`: ISO start date (e.g. `2026-05-20`).
   * `--end`: ISO end date (e.g. `2026-11-20`).
   * `--max-cost-usd`: float safety cap (default `20.0`).
   * `--confirm`: boolean flag. **Without `--confirm`, the script only prints the itemized cost estimate and exits 0 without spending a cent.**
   * `--raw-dir`: default `data/databento/opra`.
   * `--out-dir`: default `data/backtest`.
   * `--snap-time`: default `15:45` (ET).

2. **Caching & Cost Safety Flow:**
   * For each symbol `SYM`:
     * Raw cache paths:
       * `data/databento/opra/OPRA.PILLAR_{SYM}_definition_{start}_{end}.dbn.zst`
       * `data/databento/opra/OPRA.PILLAR_{SYM}_cbbo-1m_{start}_{end}.dbn.zst`
     * Underlying daily bars:
       * First check if local ThetaData Parquet bars cover `[start - 45d, end]` in `data/thetadata/bars/{SYM}_*.parquet`.
       * If not covered, plan a Databento `EQUS.MINI` `ohlcv-1d` fetch for `[start - 45d, end]` cached to `data/databento/equities/EQUS.MINI_{SYM}_ohlcv-1d_{bar_start}_{end}.dbn.zst` (note the 45-day warm-up buffer so `rv_20d` is ready on `start`!).
   * Call `client.metadata.get_cost(...)` for **every uncached** file. Print a clean table of `(symbol, schema, cached?, cost_usd)` and `TOTAL_ESTIMATED_COST_USD`.
   * If `not args.confirm`: print `"Dry run complete. Re-run with --confirm to download."` and exit `0`.
   * If `total_cost > args.max_cost_usd`: print an error and exit `1` without downloading.
   * Download missing `.dbn.zst` files via `client.timeseries.get_range(..., path=str(cache_path))`.
   * Load each `.dbn.zst` via `databento.DBNStore.from_file(cache_path).to_df()`, call `build_opra_backtest_chain(...)`, and write `data/backtest/{SYM}_opra_daily.parquet`. Print row count, date span, unique expiries, and mean VRP.

---

### Component C: Multi-Expiry, Delta-Targeted & TOB Parity Upgrade to `src/backtest/vrp_walkforward.py`

Inspect `src/backtest/vrp_walkforward.py` and `src/scanner/verticals.py` in full before editing.

1. **Multi-Expiry Support (`expiry` column):**
   * Update `_quote_by_strike(day_df, strike, right="P", expiry: str | None = None)`:
     * When `expiry is not None` and `"expiry"` is in `day_df.columns`, filter `(day_df["expiry"] == expiry) & (day_df["right"] == right) & (day_df["strike"] == strike)`.
   * Update `compute_exit_debit(day_df, short_strike, long_strike, exit_slippage_frac, expiry: str | None = None)` to pass `expiry` through.
   * On entry, add helper `select_entry_slice(day_df: pd.DataFrame, as_of: date, target_dte: int = TARGET_DTE_DAYS, min_entry_dte: int = 21, max_entry_dte: int = 45) -> tuple[pd.DataFrame, date, str | None]`:
     * If `"expiry"` is **not** in `day_df.columns`: return `(day_df, as_of + timedelta(days=target_dte), None)` (100% backward compatible).
     * If `"expiry"` **is** in `day_df.columns`: compute calendar DTE for each unique expiry in `day_df`, keep expiries with `min_entry_dte <= dte <= max_entry_dte`, choose the expiry minimizing `abs(dte - target_dte)` (tie-break: longer DTE), and return `(slice_df, chosen_date, chosen_iso_str)`. If no expiry falls in `[min_entry_dte, max_entry_dte]`, skip entry for that day.
   * Store `"expiry": chosen_iso_str` and `"expiry_date": chosen_date` on the open trade dict, and pass `expiry=t.get("expiry")` to every exit quote lookup (`PROFIT_TARGET`, `TIME_STOP`, and force-close).
   * Pass the real `ttm_days = float((chosen_date - as_of).days)` into `put_spread_combo(...)` and into BSM fallback `ttm_remaining = max((t["expiry_date"] - as_of).days, 1.0)`.

2. **Optional `use_delta_targeted: bool = False` Mode:**
   * Add `use_delta_targeted: bool = False` and `min_vrp: float = MIN_VRP` parameters to `run_midcap_vrp_backtest(...)`.
   * When `use_delta_targeted=True`, convert the put rows of the selected entry slice into dicts and call `delta_targeted_verticals(puts, [], spot=spot, short_delta_range=(0.15, 0.30), min_width_usd=2.50, min_width_pct_spot=0.04, max_width_pct_spot=0.12, min_mid_credit=MIN_MID_CREDIT, ttm_days=entry_ttm_days)` from `src/scanner/verticals.py`.
   * Filter surviving pairs through `_gate_parity_checks(...)`, evaluate each with `pricer.evaluate_combo(...)`, and select the approved pair with the highest `net_ev_usd` (or skip if none approved).

3. **TOB Size Parity (When `bid_size` / `ask_size` Exist):**
   * Add `min_tob_size: int = 4` and `max_tob_participation: float = 0.25` to `_gate_parity_checks` / `run_midcap_vrp_backtest`.
   * If `"bid_size"` and `"ask_size"` are in `day_df.columns`:
     * Require `short["bid_size"] >= min_tob_size` and `wing["ask_size"] >= min_tob_size` (else reject with `LOW_TOB_SIZE`).
     * Cap `lots` at `min(lots, int(short_bid_sz * max_tob_participation), int(wing_ask_sz * max_tob_participation))`, blocking entry if `capped_lots < 1`.

4. **Clean Up Sprint 4 Review Nit:**
   * Remove the two redundant `exit_mid = 0.0` lines inside the `PROFIT_TARGET` and `TIME_STOP` branches so BSM-fallback `exit_mid` is preserved when computing `exit_slippage` for `total_friction_usd`.

---

### Component D: `scripts/run_vrp_experiments.py` (5-Experiment Ablation Harness)

Create `scripts/run_vrp_experiments.py` with:
1. A clean, importable pure function `run_ablation_suite(df: pd.DataFrame, symbol: str, *, pricer=None, calendar=None) -> dict[str, list[dict[str, Any]]]` running 4 single-symbol experiments:
   * **`friction_frontier`:** `passive_touch_fill_prob` $\in \{0.25, 0.50, 0.75\}$ $\times$ `exit_slippage_frac` $\in \{0.25, 0.50, 0.75, 1.00\}$.
   * **`exit_rules`:** `profit_target_fraction` $\in \{0.25, 0.50, 0.75, 1.00\}$ $\times$ `min_exit_dte` $\in \{7, 14, 21\}$.
   * **`vertical_selection`:** A/B comparison of `use_delta_targeted=False` (adjacent) vs. `use_delta_targeted=True` (`WP-8` BSM delta-targeted).
   * **`vrp_threshold`:** `min_vrp` $\in \{0.00, 0.025, 0.05, 0.08\}$.
2. A CLI `main()` accepting `--parquet <path> [<path2> ...]` and `--out-json logs/vrp_experiments.json` that:
   * Runs `run_ablation_suite()` on each Parquet file.
   * Prints formatted ASCII/Markdown summary tables to stdout (including a **Cross-Symbol Comparison** table when multiple Parquet files are passed).
   * Writes the structured results to `--out-json`.

---

## 4. Required Unit & Adversarial Tests (100% Offline)

1. **`tests/data/test_opra_chain_builder.py` (New):**
   * `test_parse_opra_definitions_filters_non_standard_multipliers`: Drops mini/adjusted options (`multiplier != 100`) and non-C/P rows.
   * `test_extract_daily_nbbo_snapshot_picks_last_valid_before_1545_et`: Given intraday quotes at 15:20, 15:44, 15:45, and 15:59 ET (plus a crossed quote at 15:45:30), picks the exact 15:45 ET valid quote and ignores post-15:45 quotes.
   * `test_compute_daily_rv_table_has_no_lookahead`: Injecting a huge volatility shock on day 40 does **not** change `rv_20d` on day 39.
   * `test_build_opra_backtest_chain_round_trips_bsm_iv`: Synthetic option quotes priced at known `iv=0.35` round-trip through `build_opra_backtest_chain()` with `|iv - 0.35| < 1e-3`, accurate `atm_iv`, and all canonical schema columns present.

2. **`tests/backtest/test_vrp_walkforward.py` (Additions):**
   * `test_multi_expiry_exit_matches_exact_expiry`: Chain with both a 7-DTE expiry (cheap quotes) and a 30-DTE expiry (expensive quotes) at the same strikes — verify entry selects the 30-DTE expiry and exits mark strictly against the 30-DTE quotes (never falsely triggering `PROFIT_TARGET` off the 7-DTE quotes!).
   * `test_delta_targeted_verticals_mode_in_backtest`: `$1`-spaced `$40` mid-cap chain with `use_delta_targeted=True` skips `$1`-wide pairs and trades a `>= $2.50`-wide OTM put spread.
   * `test_tob_size_parity_caps_and_blocks`: Chain with `bid_size=2` blocks entry (`completed_trades == 0`); chain with `bid_size=8, ask_size=8` caps lots at `int(8 * 0.25) = 2`.
   * `test_bsm_fallback_preserves_exit_mid_in_friction`: Missing exit quotes trigger BSM fallback and compute `total_friction_usd` using the BSM spread penalty (not raw `exit_debit`).

3. **`tests/backtest/test_vrp_experiments.py` (New):**
   * `test_run_ablation_suite_returns_all_four_tables`: Runs `run_ablation_suite()` on a 25-day synthetic multi-expiry chain with `FakePricer` and verifies all 4 experiment keys (`friction_frontier`, `exit_rules`, `vertical_selection`, `vrp_threshold`) have the expected row counts and summary metrics.
