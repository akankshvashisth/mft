# Refactor 02 — Modular components and isolated tests

**Date:** 2026-11-24 · **Base:** commit `2293c9e` (fixes.01) · **Scope:** the whole repository (`src/`, `ui/`, `scripts/`).

Goals:
- Pull repeated behaviour into shared, pure functions.
- Turn the NautilusTrader actors and the strategy into thin shells over testable services.
- Add a pytest suite that tests every component, and its sub-functions, in isolation.

> **Status:** working-tree changes, not committed. Per the project constraints, nothing was compiled, installed or run — **not even the new tests**. Everything was checked statically only (see [Verification](#verification)).
>
> NautilusTrader, the IB adapter, `fast_vollib` and Valkey are assumed to behave as the existing code already expects. QuantLib, scipy, scikit-learn and fakeredis are assumed to be installed. Where they already provide a computation or a test double, they replace the hand-written version; see [Library-backed implementations](#library-backed-implementations). The first test run may need a few small tolerance or API tweaks; see [Caveats](#caveats--read-before-the-first-run).

## At a glance

| Area | Before | After |
|---|---|---|
| Actors (6) | 130–536 lines each, mixing logic, Valkey I/O and Nautilus calls | 54–90-line shells over pure services in `src/services/` |
| Strategy | `vrp_income.py`, 801 lines | Three modules:<br>• pure rules in `vrp_core.py`<br>• orchestration behind protocols in `vrp_engine.py`<br>• a 103-line Nautilus shell |
| Gate and daemon | Rules, Valkey reads and clocks in one class each | Pure `gate_rules.py` / `daemon_rules.py`, plus thin `PreTradeGate` / `OutOfProcessRiskDaemon` with injected store, calendar, clocks and sink |
| Shared behaviour | Repeated in up to 11 files:<br>• Redis clients and JSON readers<br>• freshness checks<br>• Black–Scholes and SVI maths<br>• quote checks<br>• time units and constants | One implementation each, in `src/common/` and `src/quant/`; pricing and calendars on QuantLib, statistics on scipy and scikit-learn |
| UI | 1,168 lines, with its own Redis client, BS/SVI maths and JSON helpers | 958 lines over `src/dashboard/` and the shared store |
| Scripts | Business logic inline: backtest loop, contract lists, node config | Thin CLIs over `src/runtime/` and `src/backtest/` |
| Tests | None | 430 test functions in 51 files, plus parametrised cases:<br>• 393 pure tests need numpy, pandas, scipy, scikit-learn, QuantLib and fakeredis; a few also need an optional package and skip without it<br>• 37 need NautilusTrader and skip themselves without it |
| `src/data/` | Missing (excluded by `.gitignore`'s `data/` rule) | `.gitignore` uses `/data/`; minimal `src/data/event_calendar.py` stand-in |

Twelve small fixes were made along the way; see [Fixes made while extracting](#fixes-made-while-extracting).

---

## Architecture

### Layers and dependency rules

```
ui/app.py, scripts/*.py                         entry points (thin)
        │
src/actors/*, src/strategies/vrp_income.py      Nautilus shells: convert events, forward to the core
src/adapters/{nautilus,ports,order_gateway,book_market}.py, src/runtime/ib_node.py
        │   ports and protocols
src/services/*, src/strategies/{vrp_core,vrp_engine}.py, src/risk/*, src/scanner/*,
src/execution/*, src/dashboard/*, src/backtest/*, src/data/*,
src/adapters/{extract,history}.py, src/runtime/instruments.py      pure, testable core
        │
src/common/*, src/quant/*                       shared building blocks
```

**Pure layer.** No top-level import of `nautilus_trader`, `redis`, `fast_vollib`, `torch`, `streamlit` or `yaml`. numpy, pandas, scipy, scikit-learn and QuantLib are ordinary dependencies. Each remaining heavy dependency is imported inside the one function that needs it:

| Dependency | Imported lazily in |
|---|---|
| `redis` | `src.common.store.make_store` |
| `fast_vollib` | `src.services.vol_surface.fast_vollib_iv` / `fast_vollib_arb_free`, `src.risk.scenario_pricer.fast_vollib_price` |
| `yaml` | `src.data.event_calendar` (file loader only) |
| `nautilus_trader` | `src.adapters.history.open_catalog` |
| `src.data.event_calendar` | `src.risk.calendar.default_event_calendar` (only when no calendar is injected) |

**Nautilus layer:**
- `src/adapters/{nautilus,ports,order_gateway,book_market}.py`
- `src/actors/*`
- `src/strategies/vrp_income.py`
- `src/runtime/ib_node.py`
- `src/models/*` (unchanged)

**Direction:** shells → services → common/quant, never the reverse.

**What stays the same:**
- Actor and strategy class names.
- Config classes, their constructor parameters and their defaults.
- Valkey keys and payload fields (except where noted under [Fixes](#fixes-made-while-extracting)).
- Every public name of a modified `src/` module is still importable from its old path, re-exported where it moved. For example:
  - `from src.risk.pre_trade_gate import CandidateLegSpec, GateConfig, GateDecision`
  - `from src.scanner.surface_scanner import CandidateCombo, CandidateLeg`

### Module map

| Package | Modules | Contents |
|---|---|---|
| [src/common/](src/common/) | `constants`, `keys`, `store`, `timeutil`, `freshness`, `quotes`, `market`, `numeric`, `flags`, `jsonl`, `log` | • Units and defaults (`NS_PER_SEC`, `DAYS_PER_YEAR`, `DEFAULT_RISK_FREE_RATE`, …)<br>• Every Valkey key and key builder<br>• `JsonStore` / `JsonBatch` / `NullStore` / `make_store`<br>• NY/UTC dates, year fractions, regular-session test on QuantLib's NYSE calendar<br>• Payload timestamp, age and freshness<br>• Two-sided mid, tradable quote, combo mid/natural<br>• Plain `QuoteObs` / `TradeObs` / `BarObs` / `OptionSpec`, `symbol_of`, `LatestValue`<br>• `is_number`, `finite_or_none`, `round_or_none`, `is_truthy_flag`<br>• `JsonlSink` / `RecordSink`, `LogLike` (services fall back to a stdlib `logging` logger) |
| [src/quant/](src/quant/) | `black_scholes`, `svi`, `density`, `parity`, `quote_hygiene`, `realized_vol`, `risk_measures`, `stress`, `surface_metrics` | • Black–Scholes–Merton price and Greeks, vega, vectorised prices and implied vols (QuantLib `BlackCalculator`, `blackFormula`, `blackFormulaImpliedStdDev`)<br>• SVI total variance, IV, Roger Lee utilisation, QuantLib `SviSmileSection` adapter, slice calibration (scipy `least_squares`, scikit-learn RMSE)<br>• Breeden–Litzenberger density (QuantLib `SmileSection.density`, scipy `trapezoid`)<br>• Put–call-parity forward and implied q<br>• Slice quote selection, clean-quote mask<br>• Yang–Zhang, bipower jump ratio, log returns, pairwise correlation, percentile rank (scipy), trailing RV<br>• VaR/ES tail metrics, per-lot tail loss, lot sizing, parametric book VaR/ES (scipy normal multipliers)<br>• Spot × vol shock grids<br>• 25Δ wings (RR/BF), strategy bias, trend classification |
| [src/services/](src/services/) | `regime`, `futures_basis`, `realized_vol`, `surface_metrics`, `vol_surface`, `portfolio_marks`, `portfolio_tracking`, `portfolio` | Each actor's logic: plain observations in, payloads published through a `JsonStore` |
| [src/strategies/](src/strategies/) | `vrp_core`, `vrp_engine` (new); `vrp_income` (shell) | • `vrp_core`: spread book, exit/entry rules, candidate building, gate legs, ranking<br>• `vrp_engine`: order lifecycle behind `OrderGateway` / `MarketView` protocols |
| [src/risk/](src/risk/) | `calendar`, `gate_types`, `gate_rules`, `daemon_rules` (new); `pre_trade_gate`, `risk_daemon`, `scenario_pricer` (slimmed) | • Calendar protocol with a lazy default<br>• Gate dataclasses<br>• Pure gate and daemon rules<br>• Thin I/O classes |
| [src/scanner/](src/scanner/) | `combos`, `verticals` (new); `surface_scanner` | `CandidateLeg` / `CandidateCombo`; vertical-pair construction shared by the scanner and the strategy |
| [src/execution/](src/execution/) | `passive_combo` | Pure `stepped_limit` / `slippage_record`; executor with injected clock, id factory, sink and wall clock |
| [src/adapters/](src/adapters/) | `extract`, `history` (pure); `nautilus`, `ports`, `order_gateway`, `book_market` (Nautilus) | Nautilus ↔ plain conversions, the ports seam, Nautilus gateway / market view / book market, daily-history loading |
| [src/dashboard/](src/dashboard/) | `formatting`, `state`, `views` | Value formatting, live-state loading with SCAN, the manual-halt flag, every table and curve the UI renders |
| [src/runtime/](src/runtime/) | `instruments` (pure), `ib_node` (Nautilus) | • `instruments`: instrument ids, universes, the option slice, IB contracts<br>• `ib_node`: IB provider, data, exec and node configs |
| [src/backtest/](src/backtest/) | `vrp_walkforward` | The walk-forward backtest that used to live inline in `scripts/run_backtest.py` |
| [src/data/](src/data/) | `event_calendar` (new, minimal) | `EventCalendar` stand-in (see [Caveats](#caveats--read-before-the-first-run)) |

### The shell pattern

Every actor now follows the same shape (regime actor, abridged):

```python
class MacroRegimeActor(DataActor):
    def __init__(self, config: MacroRegimeActorConfig) -> None:
        super().__init__(config)
        self.ports = ActorPorts(self)                        # every engine call goes through here
        self.core = RegimeService(
            plain_settings(config, RegimeSettings),          # config -> frozen dataclass of plain values
            store=make_store(config.valkey_url),             # NullStore when the URL is empty
            log=PortsLog(self),                              # logs follow a ports swap
        )
        self.wall_clock_ns = time.time_ns

    def on_quote(self, tick: QuoteTick) -> None:
        self.core.on_quote(quote_obs(tick))                  # Nautilus object -> plain QuoteObs

    def on_timer(self, event) -> None:
        state = self.core.evaluate(event_ts_ns=int(event.ts_event), wall_ns=self.wall_clock_ns())
        ...                                                  # publish the custom data via self.ports
```

How the pieces fit:
- **Config defaults** come from the settings dataclass, so there is one source of truth. For example, `interval_secs: int = RegimeSettings.interval_secs`.
- **`ports`** wraps the clock, cache, subscriptions, timers, publishing and logging. `StrategyPorts` adds order handling. A test swaps the attribute for a fake.
- **`PortsLog`** forwards logs, and **`PortsProxy`** forwards any other call, to `owner.ports` at call time. Objects built in `__init__`, such as the order gateway or the book market, therefore follow a later ports swap.
- **The strategy's engine attribute is `vrp_engine`**, so it doesn't clash with Nautilus `Strategy` attributes.

### Seams (injection points)

| Component | Seam | Production default | Typical test double |
|---|---|---|---|
| Every service | `store`, `log` | `make_store(valkey_url)`, `PortsLog(actor)` (`NullStore` / the module's `logging` logger when omitted) | `JsonStore` over fakeredis, `create_autospec(logging.Logger)` |
| `VolSurfaceService` | `iv_solver`, `arb_validator`, `svi_fitter` | `fast_vollib_iv`, `fast_vollib_arb_free`, `fit_svi_slice` | Scripted callables; `bsm_implied_vols` (QuantLib) with the real `fit_svi_slice` end to end |
| `PortfolioService` | `market` (`BookMarket`) | `NautilusBookMarket(PortsProxy(actor))` | `FakeBookMarket` |
| `RealizedVolActor` | `history_loader_factory` | `DailyHistoryLoader` | Lambda returning a frame |
| `SurfaceMetricsActor` | `catalog_factory` | `open_catalog` | `mock.Mock` catalog |
| Actor and strategy shells | `ports`, `wall_clock_ns` | `ActorPorts(self)` / `StrategyPorts(self)`, `time.time_ns` | `FakeActorPorts` / `FakeStrategyPorts`, fixed value |
| `VrpEngine` | `gate`, `pricer`, `executor`, `store`, `gateway`, `market`, `log`, `clock_ns`, `monotonic` | `PreTradeGate`, `ScenarioPricer`, `PassiveComboExecutor`, Valkey, `NautilusOrderGateway`, `NautilusMarketView` | `FakeGate`, `FakePricer`, real executor with an autospecced sink, `FakeOrderGateway`, `FakeMarket`, `mock.Mock` clocks |
| `PreTradeGate` | `calendar`, `store`, `clock_ns`, `monotonic` | `default_event_calendar()`, `make_store(valkey_url)`, `time.time_ns`, `time.monotonic` | Autospecced `EventCalendar`, `JsonStore` over fakeredis, `mock.Mock` clocks |
| `OutOfProcessRiskDaemon` | `store`, `sink`, `clock_ns`, `sleep`, plus `run_forever(max_iterations, install_signal_handlers)` | Valkey, `JsonlSink(jsonl_path)`, `time.time_ns`, `time.sleep` | `JsonStore` over fakeredis (a disconnected server for outages), autospecced `JsonlSink`, `mock.Mock` clock and sleep |
| `ScenarioPricer` | `price_fn`, `seed` | `fast_vollib_price` | `bsm_prices` (QuantLib) |
| `PassiveComboExecutor` | `clock`, `id_factory`, `fill_sink`, `wall_clock_ns` | `time.monotonic`, uuid, `JsonlSink`, `time.time_ns` | `mock.Mock` clock, fixed ids, autospecced `JsonlSink` |
| `run_midcap_vrp_backtest` | `calendar`, `pricer`, `executor`, `symbol` | File-backed calendar, `ScenarioPricer`, `PassiveComboExecutor` | Autospecced `EventCalendar`, `FakePricer`, executor with an autospecced sink |
| `print_summary` | `target` (URL or store), `echo` | `print` | `mock.Mock` |
| Dashboard state | `store` argument on every loader | `get_store()` in `ui/app.py` (cached `make_store`) | `JsonStore` over fakeredis |

## Duplication removed

File counts are from `git grep` on the base commit.

| Behaviour | Repeated in (base) | Now |
|---|---|---|
| Redis client creation | 11 files: 6 actors, gate, daemon, strategy, UI, verify script | `make_store` in [src/common/store.py](src/common/store.py), the only `redis` import |
| "Read JSON or None" helpers (`_read_json`, `_get_json`, `_safe_json_load`, `load_json*`) | 7 files | `JsonStore.get_json*` / `safe_json_loads` |
| Key strings (`mft:surface:<id>`, `mft:rv:<id>`, …) | Producers and consumers separately | [src/common/keys.py](src/common/keys.py) |
| Payload age, freshness and future-skew rules | Gate, daemon, futures actor, UI | [src/common/freshness.py](src/common/freshness.py) |
| `365.25`, `86400` / `1e9` unit conversions, `0.0425` | 10, 9 and 7 files | [src/common/constants.py](src/common/constants.py), [src/common/timeutil.py](src/common/timeutil.py) |
| `ZoneInfo("America/New_York")` | 3 files | `timeutil.NY_TZ`, `ny_date` |
| Bid/ask validity, mids, combo mid/natural | 7 files: 5 actors, gate, strategy | [src/common/quotes.py](src/common/quotes.py) (`QuoteObs.mid`) |
| Normal CDF/PDF, BS Greeks/price (hand-written) | Portfolio actor, UI | [src/quant/black_scholes.py](src/quant/black_scholes.py), on QuantLib |
| SVI evaluation and calibration | 4 files (vol, metrics and portfolio actors, UI) | [src/quant/svi.py](src/quant/svi.py) |
| 25Δ wings (two different definitions) | Vol actor, metrics actor | [src/quant/surface_metrics.py](src/quant/surface_metrics.py) |
| Clean-quote mask | Vol actor, `bench_fast_vollib.py`, verify script | [src/quant/quote_hygiene.py](src/quant/quote_hygiene.py) |
| VaR/ES tail metrics, stress grids | Pricer, portfolio actor, bench script | [src/quant/risk_measures.py](src/quant/risk_measures.py), [src/quant/stress.py](src/quant/stress.py) |
| Vertical-spread construction | Scanner, strategy | [src/scanner/verticals.py](src/scanner/verticals.py) |
| Instrument symbol parsing (`split(".")[0]`) | Portfolio actor, UI | `symbol_of` (`rsplit`, keeps dotted symbols) |
| Nautilus → plain conversions | Actors and the strategy | [src/adapters/nautilus.py](src/adapters/nautilus.py) |
| Contract lists and node config | `run_paper.py`, `probe_instrument.py` | [src/runtime/](src/runtime/) |

## Library-backed implementations

Where QuantLib, scipy, scikit-learn, numpy or the standard library already provide a computation or a test double, the code calls the library instead of a hand-written version. The APIs were checked against the installed sources of QuantLib 1.43, scipy 1.17, scikit-learn 1.9, numpy 2.4 and pandas 3.0.

### In `src/`

| Was hand-written | Now | Behaviour |
|---|---|---|
| Normal CDF/PDF, and the Black–Scholes price, delta, gamma, vega, theta and vanna formulas in `bs_greeks` | QuantLib `BlackCalculator` on the carry forward `S·e^((r−q)T)`, standard deviation `σ√T` and discount `e^(−rT)` | Same values and units. QuantLib has no charm, so charm is derived from QuantLib's delta and gamma |
| Vectorised Black-76 (`black_forward_price`) and numpy Black–Scholes–Merton (`bs_price_numpy`) | `bsm_prices` over QuantLib `blackFormula`; new `bsm_implied_vols` over `blackFormulaImpliedStdDev` | Both take the fast_vollib arguments, so either can be injected as `price_fn` / `iv_solver`. Production defaults are still fast_vollib |
| Vega from a hand-built `d1` (`black_vega_spot`) | `black_vega`: `BlackCalculator.vega` | Same value: `F·e^(−rT) = S·e^(−qT)` holds by construction of the parity forward |
| Intrinsic value | QuantLib `PlainVanillaPayoff` | Same |
| Breeden–Litzenberger density: two `np.gradient` passes over hand-priced calls, then numpy's trapezoid | QuantLib `SviSmileSection.density`, then `scipy.integrate.trapezoid` | Same density up to finite-difference error. It's undiscounted, so the rate argument is gone. Parameters QuantLib rejects raise, and the dashboard shows no density for them |
| Percentile rank `100·mean(x <= v)` | `scipy.stats.percentileofscore(kind="weak")` | Same, except NaNs in the history are now ignored instead of counted |
| VaR/ES multipliers typed as 2.326 and 2.665 | `scipy.stats.norm`: `ppf(c)` and `pdf(z)/(1−c)`, behind a `confidence` argument | Book VaR/ES move by about 0.01% (2.32635 and 2.66521) |
| 25-delta constant 0.6745 | `norm.ppf(0.75)` | Same to four decimals |
| SVI fit RMSE and maximum residual | scikit-learn `root_mean_squared_error` and `max_error` | Same |
| Correlated spot/vol draws `ρ·z1 + √(1−ρ²)·z2` | `Generator.multivariate_normal(..., method="cholesky")` | Statistically identical, but individual scenarios differ, so pricer outputs and backtest numbers shift by Monte Carlo noise |
| Weekday-only regular-session test | NYSE business days from QuantLib's `UnitedStates(NYSE)` calendar (`is_nyse_trading_day`) | NYSE holidays no longer count as sessions |
| Running peak for the backtest's maximum drawdown | `np.maximum.accumulate` over the equity curve | Same |
| `sum(...) / len(...)` in the scanner | `statistics.fmean` | Same |
| `NullLog` | `logging.getLogger(__name__)` when no log is injected | Services without an injected log emit to Python logging instead of discarding messages |

Kept hand-written, because no library offers the same thing:
- **The raw-SVI formula and the spread-weighted calibration objective.** QuantLib's `SviInterpolatedSmileSection` is date-based (whole days) and supports only vega weights. The optimiser also has to evaluate parameters that QuantLib's `SviSmileSection` rejects.
- **Other estimators and rules:**
  - Yang–Zhang and bipower variation.
  - Quote hygiene and the parity forward.
  - Tail metrics: VaR as a P&L percentile, ES as the mean at or below it.
  - The stress-grid layout and OCC ids.

### In `tests/`

| Was hand-written | Now |
|---|---|
| `FakeValkey`, `FakePipeline`, `FailingValkey` | fakeredis via `valkey_client()`. TTLs are read with `PTTL`; outages use a disconnected `FakeServer` |
| `FakeClock`, `MemorySink`, `FailingSink`, `RecordingLog`, `FakeCalendar`, the call-recording classes, and the `scipy.optimize` module injected into `sys.modules` | `unittest.mock`:<br>• `Mock(return_value=…)` clocks<br>• `create_autospec(JsonlSink)`, `create_autospec(logging.Logger)` and `create_autospec(EventCalendar)`, so calls are checked against the real signatures<br>• plain `Mock` recorders<br>• `least_squares` patched on `src.quant.svi` |
| `FakeEventResult` | The real `EventCheckResult` |
| Hand-written finite differences and hand-derived reference numbers | QuantLib `AnalyticEuropeanEngine`, `blackFormula`, `SviSmileSection`, `InverseCumulativeNormal` and `NormalDistribution`; `scipy.differentiate.derivative`; `scipy.stats.lognorm` and `norm` |
| Synthetic option prices built from the parity formula | QuantLib-priced quotes, in the parity tests and in an end-to-end vol-surface test that inverts them with QuantLib and calibrates with scipy |
| Epoch-nanosecond helper built on `datetime` arithmetic | `pd.Timestamp(...).value` |

## Fixes made while extracting

Small defects found while moving code. Fixes 1–10 and 12 each have targeted tests; fix 11 is a clean-up.

### 1. Dividend handling in IV inversion and scenario repricing

**Problem.** The vol actor and the scenario pricer passed `r − q` as the rate to a no-dividend Black–Scholes. That keeps `d1`/`d2` right but multiplies every price by `e^(qT)`:
- Model prices came out too high.
- Inverted IVs came out slightly too low, of the order of 0.01 vol point at this slice's maturities.
- The error grows with maturity.

**Fix.** Exact Black–Scholes–Merton: spot discounted to `S·e^(−qT)`, with the true rate `r`. Applies to both the inversion (`VolSurfaceService`) and the repricing (`fast_vollib_price`).

### 2. One 25-delta skew definition

**Problem.** The surface payload's `skew_25d` was the smile difference at a fixed ±0.05 log-moneyness, labelled "25d". The metrics actor's risk reversal used true 25Δ strikes from the SVI smile. The same concept showed different numbers on different pages.

**Fix.** The surface now publishes `smile_wings(atm_iv, T, params).rr_25d`, the same function the metrics service uses (`delta25_log_moneyness` on the SVI smile). The sign convention is unchanged (put minus call). A test checks the two agree.

### 3. Common random numbers in `ScenarioPricer`

**Problem.** One RNG advanced across evaluations. Two candidates scored back-to-back saw different scenarios, and re-scoring the same candidate gave a different result, so the ranking was noisy.

**Fix.** Every `evaluate_combo` draws from a fresh `np.random.default_rng(seed)`. Candidates are compared on identical scenarios, and results are reproducible. `self.rng` was removed. Backtest results shift slightly as a result.

### 4. One option slice drives loading, trading and fitting

**Problem.** The strikes were defined in three separate places:
- `run_paper.py` loaded one strike set (`build_spy_otm_slice`).
- The strategy got a hard-coded 675–745 list that included strikes never loaded.
- The vol actor's strike window was configured on its own.

**Fix.** `SPY_DEC26_SLICE` in [src/runtime/instruments.py](src/runtime/instruments.py) (puts 685–720, calls 720–755) is the single source. It generates:
- The IB contracts.
- The strategy's `option_ids`.
- The vol actor's `strike_min` / `strike_max`.

### 5. Unusable or far-future timestamps fail closed

**Problem.**
- The daemon treated a payload with no usable timestamp as fresh. A stamp more than 900 s in the future gave a negative age, which also passed.
- The futures and dashboard checks each had their own variant of these rules.

**Fix.** One rule set in `src/common/freshness.py`: a stamp must be after `MIN_EPOCH_NS` and at most 900 s ahead.
- **Daemon:** halts with `invalid_timestamp:<key>`.
- **Gate:** treats the snapshot as stale.
- **UI:** shows no age.

### 6. Real short-call extrinsic reaches the gate and the calendar

**Problem.** The strategy built gate legs without an extrinsic value, so `CandidateLegSpec.extrinsic_value` defaulted to `1.0`. The early-assignment (ex-dividend) check therefore assumed $1 of time value on every short call. For any dividend under $1, a deep in-the-money call whose real time value was below the dividend could pass.

**Fix.**
- `gate_leg` computes the leg's real time value (mid − intrinsic at the current spot).
- The field now defaults to `None`, meaning unknown, which the calendar treats as plausible assignment (fail closed).

### 7. New York trading date for expiries and `as_of`

**Problem.** The strategy's default candidate expiry and the gate's `as_of` used the machine's local `date.today()`. On a UTC host, dates roll at 19:00/20:00 New York time, so DTE and expiry checks were off by one in the evening.

**Fix.**
- Both use `ny_date(...)`.
- The engine passes `as_of=today` (New York date) to the gate.

### 8. No filesystem side effects in constructors

`PassiveComboExecutor` and `OutOfProcessRiskDaemon` no longer create log directories when constructed. `JsonlSink` creates the directory on first write. Constructing either object in a test or a read-only deployment no longer touches the disk.

### 9. The UI lists surfaces with SCAN

`KEYS mft:surface:*` blocks Valkey while it runs. `JsonStore.list_keys` uses `SCAN` (`scan_iter`) instead.

### 10. Backtest preview clips are not registered

The backtest called `create_parent_plan` with registration on. Every simulated entry therefore stayed in `executor.active_clips` for the rest of the run. It now passes `register=False`.

### 11. Clean-ups

- Removed the UI's unused `_bs_delta`.
- Repaired [scripts/verify_vol_pipeline.py](scripts/verify_vol_pipeline.py):
  - It imported names that no longer existed (`VolSurfaceSnapshot`, `ComboLegSpec`, `evaluate_combo_risk`, `filter_clean_quotes`).
  - It now uses the hygiene mask, fast_vollib, `IVSurface`, real `SviSliceParams` and a `ScenarioPricer` iron condor.
  - Its result goes to `mft:verify:vol_pipeline` instead of `surface:XBI.ARCA`.

### 12. An unreadable book ES99 blocks instead of reading as zero

**Problem.** If a fresh `mft:portfolio:risk` payload had a null, non-numeric or NaN `risk.book_es_99_usd`, the old gate either raised (`float(None)`) or, for NaN, blocked by accident with a misleading `SLEEVE_ES99_BUDGET_EXHAUSTED`. Simply reading such values as `0` while extracting the rule would have approved trades against an unknown book.

**Fix.** `portfolio_reasons` now returns the new reason `PORTFOLIO_RISK_INVALID`, using the new `finite_or_none` helper in `src/common/numeric.py`. The portfolio service always publishes a finite number, so this only fires on corrupt or incompatible payloads.

## Other behaviour changes

- **Futures mids:** the newest observation wins (`LatestValue`, by event time). Previously the last message to arrive won.
- **Store fallback:** a missing Valkey URL gives a `NullStore`: reads are empty and writes are dropped. This replaces the scattered `rdb is None` checks.
- **Daemon error path:**
  - Error transitions are capped at `transitions_max_len` like normal ones. Before, they were pushed without a trim, so a persistent error grew the list without bound.
  - A failing JSONL sink can no longer stop the daemon from publishing its halt. The old error handler's file write could raise out of the loop.
- **JSON reading:** invalid UTF-8 counts as "no value" everywhere. The daemon's old helper decoded outside its `try` and raised; the daemon now flags the input as missing instead (still a halt).
- **Persisted book:** `SpreadBook.from_json` keeps only well-formed open trades, so a hand-edited or stale book can't crash the strategy at start-up.
- **Shared store:** `PreTradeGate` shares the strategy's store instead of opening a second connection.
- **Dashboard values:**
  - Formatting is None-safe: `fmt` shows `N/A`.
  - The realized-vol table shows blanks for missing horizons. Before, it showed `0` for a missing key and raised on an explicit `null`.
- **Symbol matching:** uses `symbol_of`, which splits at the last dot, so dotted symbols keep their full name.
- **Event calendar:** the minimal stand-in matches events by symbol root via `symbol_of`; see [Caveats](#caveats--read-before-the-first-run).

## Interface changes

| What | Before | After |
|---|---|---|
| `CandidateLegSpec.extrinsic_value` | Default `1.0` | Default `None` (unknown, fail closed) |
| `PreTradeGate(...)` | `(config, calendar, valkey_url)` | Same, plus keyword-only `store`, `clock_ns`, `monotonic`; the default calendar is loaded lazily |
| `OutOfProcessRiskDaemon(...)` | `(config)` | `(config, *, store, sink, clock_ns, sleep)`; new `read_inputs`, `evaluate_once`, `record_error`; `run_forever(max_iterations=None, install_signal_handlers=True)` |
| `risk_daemon.print_summary` | URL only | URL or `JsonStore`, plus an `echo` callable |
| `ScenarioPricer(...)` | Shared `self.rng` | New `price_fn`; `self.rng` removed (CRN, fix 3) |
| `PassiveComboExecutor(...)` | `(config)` | `(config, *, clock, id_factory, fill_sink, wall_clock_ns)`; pure `stepped_limit` / `slippage_record` functions |
| `VrpIncomeStrategy` | Logic on the strategy | Logic on `strategy.vrp_engine` (`VrpEngine`) |
| `run_midcap_vrp_backtest` | In `scripts/run_backtest.py` | In `src/backtest/vrp_walkforward.py`, with keyword-only `symbol`, `calendar`, `pricer`, `executor`; summary keys unchanged |
| `scripts/run_paper.py` | `build_spy_otm_slice()` | `SPY_DEC26_SLICE` and `ib_option_contracts()` in `src/runtime/instruments.py` |
| `ui/app.py` internals | `get_redis`, `rdb`, `load_json`, `load_json_list`, `_bs_price`, `_bs_delta`, `_norm_cdf`, `_eval_svi_iv` | `get_store()` plus `src.dashboard` / `src.quant` helpers |
| Gate reasons | — | New `PORTFOLIO_RISK_INVALID` (fix 12) |
| Daemon reason codes | — | New `invalid_timestamp:<key>` (fix 5) |
| Valkey keys | `surface:XBI.ARCA` (verify script) | `mft:verify:vol_pipeline` |
| `src.quant.black_scholes` (new in this refactor) | `norm_cdf`, `norm_pdf`, `norm_cdf_array`, `black_forward_price`, `black_vega_spot`, `bs_price_numpy` | Removed; use QuantLib or scipy directly, or `bsm_prices`, `bsm_implied_vols` and `black_vega` |
| `parametric_book_var_es` | `z_var`, `z_es` keywords | `confidence` keyword (default 0.99) |
| `breeden_litzenberger_density`, `dashboard.views.smile_density` | `r` argument | Removed, because the density doesn't depend on the rate. `smile_density` returns `None` for parameters QuantLib rejects |
| `services.vol_surface.leg_row` | `(quote, raw_iv, svi_iv, spot, forward, ttm, q)` | `(quote, raw_iv, svi_iv, forward, ttm, discount)` |
| `src.common.log.NullLog` (new in this refactor) | Default service log | Removed; services fall back to `logging.getLogger(__name__)` |

No other key, payload field or config parameter changed.

## Tests

### Running

```bash
pytest -m "not nautilus"   # the pure suite (dependencies below)
pytest                     # everything; Nautilus-backed files skip themselves without nautilus_trader
pytest tests/risk          # one area
```

`pytest.ini` sets `testpaths = tests`, `pythonpath = .` and registers the `nautilus` marker.

### Dependencies

| Package | Needed for | Without it |
|---|---|---|
| pytest, numpy, pandas, scipy, scikit-learn, QuantLib | Everything (the last three are also runtime dependencies of `src/`) | — |
| fakeredis (with redis-py) | `tests/fakes.py`, which `conftest.py` imports | The session fails at collection |
| fast_vollib | One cross-check of `fast_vollib_price` against `bsm_prices` | That test skips |
| PyYAML | Calendar file-loading tests | Those tests skip |
| pyarrow | ThetaData parquet loading test | That test skips |
| nautilus_trader | 11 files marked `nautilus`: actor and strategy shells, ports, order gateway, conversions | Those files skip |
| nautilus_trader IB adapter | `tests/runtime/test_ib_node.py` | That file skips |

`zoneinfo` needs the IANA time-zone database. On Windows it comes from the `tzdata` package, which pandas installs. This was already a requirement of the base code.

### Layout

```
tests/
  conftest.py          fixtures: fake_valkey, store, healthy_store, now_ns, wall_clock, mono_clock, mem_sink, recording_log
  fakes.py             test doubles (below)
  factories.py         payload builders and shared constants (NOW_NS = 2026-11-24 15:00 UTC, UNDERLYING = SPY.ARCA)
  nautilus_objects.py  real Nautilus value objects for shell tests: iid, quote_tick, bar, timer_event
  common/      test_store, test_freshness, test_keys, test_quotes, test_timeutil, test_misc
  quant/       test_black_scholes, test_svi, test_density, test_parity, test_quote_hygiene,
               test_realized_vol, test_risk_measures, test_stress, test_surface_metrics
  data/        test_event_calendar
  risk/        test_gate_rules, test_pre_trade_gate, test_daemon_rules, test_risk_daemon, test_scenario_pricer
  scanner/     test_verticals, test_surface_scanner
  execution/   test_passive_combo
  services/    test_regime, test_futures_basis, test_realized_vol_service, test_surface_metrics_service,
               test_vol_surface, test_portfolio_marks, test_portfolio_service
  strategies/  test_vrp_core, test_vrp_engine, test_vrp_income_shell (N)
  adapters/    test_extract, test_history, test_nautilus_conversions (N), test_ports (N), test_order_gateway (N)
  actors/      test_regime_shell, test_vol_shell, test_realized_vol_shell, test_surface_metrics_shell,
               test_futures_shell, test_portfolio_shell (all N)
  dashboard/   test_formatting, test_state, test_views
  runtime/     test_instruments, test_ib_node (N)
  backtest/    test_vrp_walkforward
(N) = marked `nautilus`; skipped when nautilus_trader is not installed
```

### Test doubles ([tests/fakes.py](tests/fakes.py), [tests/conftest.py](tests/conftest.py))

Infrastructure doubles come from libraries. The classes further down fake project interfaces whose behaviour no library provides.

| Double | Stands in for | Notes |
|---|---|---|
| `valkey_client()` (fakeredis); `fake_valkey` / `store` fixtures | redis-py client | Real Redis semantics in memory, including TTLs, pipelines and SCAN. With `connected=False`, every command raises `redis.ConnectionError` |
| `recording_sink()`, `written()`; `mem_sink` fixture | `RecordSink` (JSONL) | `create_autospec(JsonlSink)`; `written` lists the recorded records; `error=` makes `write` raise |
| `stub_calendar()` | `EventCalendarLike` | `create_autospec(EventCalendar)`, returning a real `EventCheckResult` (allowed by default). Calls are checked against the real signature |
| `wall_clock`, `mono_clock` fixtures | Wall or monotonic clocks | `mock.Mock(return_value=…)`; tests move time through `return_value` |
| `recording_log` fixture | `LogLike` | `create_autospec(logging.Logger)`; assertions read the `info` / `warning` / `error` call arguments |
| `FakeBookMarket` | `BookMarket` | Instruments, quotes and symbol lookups from dicts |
| `FakeGate`, `FakePricer` | `PreTradeGate`, `ScenarioPricer` | Approve the requested size by default; record calls |
| `FakeOrderGateway`, `FakeMarket` | `OrderGateway`, `MarketView` | Order views, failure injection (`fail_prepare`, `fail_submit`, `modify_ok`) and an `on_submit` hook to simulate venue events |
| `FakeActorPorts`, `FakeStrategyPorts` | `ActorPorts`, `StrategyPorts` | Record subscriptions, timers, publications, logs and orders |

### Conventions

- **Reference values.** Where a library computes the same quantity independently, the library is the reference: QuantLib's analytic engine, Black formula, smile sections and normal distribution, or scipy's distributions and numerical derivatives. Otherwise expected numbers are hand-derived in the test or its comments. Nothing is copied from an implementation run. Inputs avoid rounding boundaries such as `.xx5` cents and the exact minimum-credit thresholds.
- **Lock-in tests.** Key strings, payload fields and reason codes are asserted literally, so accidental renames fail.
- **Mocks.** Optional dependencies are skipped with `pytest.importorskip`. The SVI fallback paths are driven by a `mock.Mock` patched over `least_squares` in `src.quant.svi`.
- **How shell tests work.** They build the real actor or strategy with a fake store patched into the module's `make_store`, then swap `ports` for a fake. They then drive Nautilus objects through the handlers and assert on what was published, subscribed or ordered.

## Caveats — read before the first run

1. **Nothing has been run.** The first run will probably surface a few small issues, which should be fixes to the tests rather than the design:
   - **Float tolerances.**
   - **Library API details** assumed from the existing code:
     - `Price.from_str`, `Quantity.from_int`, `OptionSpread.from_legs`, engine-less actor construction.
     - `str(BarType)` formatting, `DataType(str)`.
     - `LiveNodeConfig` / IB config field names.
     - fast_vollib argument names.
     - pandas' `.dt.as_unit`.
   - **Library behaviour read from source, never executed.** QuantLib 1.43, scipy 1.17 and scikit-learn 1.9 were read in a local environment; the first run confirms:
     - `BlackCalculator.vanna` is per unit of volatility. A test compares it with scipy's numerical derivative of delta.
     - `SviSmileSection` takes `(a, b, sigma, rho, m)`. This order is also confirmed by QuantLib's own error message.
     - fakeredis wasn't installed there. The disconnected-server outage tests and the `PTTL` checks follow its documented API. They assume commands fail immediately rather than retrying: redis-py 6 retries by default, but fakeredis builds its own connection pool.
2. **`src/data/event_calendar.py` is a minimal stand-in.** The real module was never committed because of the old `data/` ignore rule.
   - If you have the original, compare it with the stand-in's interface (`EventCalendar.check_entry(...)`, `EventCheckResult`) before replacing it.
   - With the `/data/` rule, a local `src/data/` is now visible to git, so check `git status` before committing so the two copies don't clash.
3. **Two loaders are still missing:** `src/data/catalog_loader.py` and `src/data/thetadata_chain_loader.py`.
   - `scripts/sync_historical_data.py` and `scripts/run_backtest.py` import them inside `main()`. The modules import cleanly, but those two commands fail until the loaders are restored.
4. **Runtime dependencies:**
   - PyYAML, for the default (file-backed) calendar that the gate and the strategy use.
   - QuantLib, scipy and scikit-learn, for pricing, calendars and statistics in `src/`.
5. **Out of scope, unchanged:**
   - Loading `config/sleeve.yaml`.
   - The pricer's tail-sizing policy (≈ 1,000 lots when ES99 ≈ 0; liquidity caps still bound the size).
   - CI setup.

## Verification

Static only: nothing was compiled, installed or run, including the tests. The checks used read-only text analysis.

| Check | Result |
|---|---|
| Every `from src… import …` / `from tests… import …` (946 names, 153 files) resolves to a definition, re-export or submodule | Clean except the two known-missing loaders (caveat 3) |
| Every argument of every test and fixture function (436) is a fixture, a parametrize name or a pytest built-in | Clean |
| No call, constant or class name that a file uses without defining or importing it; the checker was validated on a planted-error probe | Clean |
| f-strings reusing their own quote inside a replacement field (invalid on 3.11): 279 literals scanned | None |
| Every `ql.*` name used in `src/` and `tests/` exists in the installed QuantLib 1.43 wrapper. The signatures of `BlackCalculator`, `SmileSection.density`, `blackFormulaImpliedStdDev`, `Calendar.isBusinessDay`, `Settings.evaluationDate` and `SavedSettings` were read from it | Yes |
| The scipy, scikit-learn, numpy and pandas APIs used exist with the signatures used: `percentileofscore`, `integrate.trapezoid`, `differentiate.derivative`, `lognorm`, `norm`, `least_squares`, `root_mean_squared_error`, `max_error`, `Generator.multivariate_normal(method=)`, `Timestamp(...).value` | Yes |
| Bracket balance, block headers with a top-level colon, tab indentation | Clean |
| Fakes match their protocols (`OrderGateway`, `MarketView`, `BookMarket`, ports); calendar, sink and logger doubles are autospecs of the real classes; `monkeypatch.setattr` targets exist | Matches |
| Every public name of a modified `src/` module is still importable from its old path | Yes; only script and UI internals were removed (table above) |
| `redis` imported only in `common/store.py`; `365.25` / `86400` / `0.0425` / `ZoneInfo(` only under `src/common/`; bid/ask checks only in `common/quotes.py`; no top-level heavy imports in pure modules | Yes |
| No leftover references to removed helpers: `_read_json`, `_get_json`, `_safe_json_load`, `build_spy_otm_slice`, `_bs_greeks`, `pricer.rng`, `gate.rdb`, the replaced numerics (`norm_cdf`, `bs_price_numpy`, `black_forward_price`, `black_vega_spot`, `NullLog`) and the replaced doubles (`FakeValkey`, `FakeClock`, `RecordingLog`, `MemorySink`, `FakeCalendar`, …) | None |
| `git diff --check`, LF line endings, trailing whitespace, final newlines on new files | Clean |

## Files changed

**Rewritten**

| File | Change |
|---|---|
| [src/actors/](src/actors/) (all 6) | Thin shells: `ports` + `core` service + `wall_clock_ns`; configs unchanged |
| [src/strategies/vrp_income.py](src/strategies/vrp_income.py) | Shell over `VrpEngine` |
| [src/risk/pre_trade_gate.py](src/risk/pre_trade_gate.py) | Composes `gate_rules`; injected store, calendar and clocks |
| [src/risk/risk_daemon.py](src/risk/risk_daemon.py) | Composes `daemon_rules`; injected store, sink, clock and sleep |
| [src/risk/scenario_pricer.py](src/risk/scenario_pricer.py) | `price_fn` seam, exact BSM (fix 1), CRN (fix 3) |
| [src/scanner/surface_scanner.py](src/scanner/surface_scanner.py) | Built on `verticals` |
| [src/execution/passive_combo.py](src/execution/passive_combo.py) | Pure stepping/slippage functions, injected clock/ids/sink |
| [ui/app.py](ui/app.py) | Uses `src.dashboard`, `src.quant` and the shared store |
| [scripts/](scripts/) (all 6) | Thin CLIs over `src.runtime`, `src.backtest`, `src.quant` |
| [.gitignore](.gitignore) | `data/` → `/data/` |

**New**

| Location | What |
|---|---|
| [src/common/](src/common/), [src/quant/](src/quant/), [src/services/](src/services/), [src/adapters/](src/adapters/), [src/dashboard/](src/dashboard/), [src/runtime/](src/runtime/), [src/backtest/](src/backtest/), [src/data/](src/data/) | New packages (see [Module map](#module-map)) |
| [src/risk/](src/risk/) | `calendar.py`, `gate_types.py`, `gate_rules.py`, `daemon_rules.py` |
| [src/scanner/](src/scanner/) | `combos.py`, `verticals.py` |
| [src/strategies/](src/strategies/) | `vrp_core.py`, `vrp_engine.py` |
| [tests/](tests/) | 51 test files plus `conftest.py`, `fakes.py`, `factories.py` and `nautilus_objects.py` |
| [pytest.ini](pytest.ini) | Test configuration |
| `refactor.02.readme.md` | This document |
