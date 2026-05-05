# Prediction Markets as a Crypto Signal

**Research project** testing whether rapid probability shifts in Kalshi prediction markets can predict short-term Bitcoin/Ethereum/Solana price movements.

Maintained by Kian Jagtiani (kjagtian@usc.edu). Last updated: May 2026.

---

## Table of Contents

1. [Hypothesis](#1-hypothesis)
2. [Project Architecture](#2-project-architecture)
3. [Environment Setup](#3-environment-setup)
4. [Data Overview](#4-data-overview)
5. [Notebook Guide](#5-notebook-guide)
6. [Source Modules](#6-source-modules)
7. [Signal Definitions](#7-signal-definitions)
8. [Known Issues and Bugs](#8-known-issues-and-bugs)
9. [Current Results](#9-current-results)
10. [Next Steps — Priority Order](#10-next-steps--priority-order)
11. [Strategy Viability Checklist](#11-strategy-viability-checklist)
12. [Academic Grounding](#12-academic-grounding)

---

## 1. Hypothesis

Prediction markets on Kalshi (KXBTCMAXY, KXBTCMAX100, KXFED, KXCPI) are traded by informed participants — people who have a view on future crypto prices or macro outcomes that affect crypto. When the probability in one of these markets jumps sharply in a single minute, it may reflect new private information arriving into the market before it is fully reflected in spot crypto prices.

If the prediction market probability leads the spot price by even a few minutes, we can trade BTC/ETH/SOL spot (or perpetuals) at T+1 after the signal fires and capture the subsequent price move.

**Why this edge might exist:**
- Prediction markets trade at low notional (contracts are $1 max), so informed traders may move prediction market prices before they have enough conviction to move spot.
- FOMC and CPI prediction markets aggregate economic forecasters' views — a rapid shift in these probabilities may foreshadow macro moves that crypto then follows.
- Kalshi is a relatively illiquid venue; a single large informed trade can move probabilities significantly.

**Why it might not work:**
- Prediction market probabilities may simply react *after* spot moves (reverse causality).
- The signal is too sparse (most markets have <35% minute-level data coverage) to provide enough trades.
- Costs on crypto spot may exceed any alpha from the signal.

---

## 2. Project Architecture

```
pred_markets_as_a_signal/
├── .env                          # API keys (never commit — in .gitignore)
├── config.yaml                   # All strategy parameters (thresholds, filters, backtest)
├── requirements.txt
├── src/
│   ├── fetch.py                  # Kalshi, Binance, Deribit, LunarCrush data fetchers
│   ├── signal.py                 # 4 jump detection definitions (D1–D4)
│   ├── lead_lag.py               # CCF, Granger, placebo tests
│   ├── event_study.py            # CAR computation and subsample analysis
│   ├── backtest.py               # Vectorized backtest engine + walk-forward
│   ├── metrics.py                # Sharpe, Sortino, Calmar, drawdown, full_report()
│   └── plots.py                  # All visualization functions
├── notebooks/
│   ├── 00_market_selection.ipynb # Pull Kalshi catalog, apply liquidity filter
│   ├── 01_data_collection.ipynb  # Fetch all market data, build aligned dataset
│   ├── 02_eda.ipynb              # Data quality, stationarity, raw correlations
│   ├── 03_signal.ipynb           # Compute D1–D4 signals, save parquets
│   ├── 04_lead_lag.ipynb         # CCF + Granger causality + placebo tests
│   ├── 05_event_study.ipynb      # CAR analysis around jump events
│   ├── 06_options.ipynb          # Options-based strategy variants (WIP, malformed)
│   ├── 07_backtest.ipynb         # Full backtesting: train/val/test + walk-forward
│   └── 08_comparison.ipynb       # Definition comparison, winner selection
└── data/
    ├── raw/
    │   ├── kalshi/               # One .parquet per prediction market + catalog files
    │   ├── crypto/               # 1-min OHLCV CSVs from Binance + compiled parquets
    │   └── deribit/              # DVOL (BTC + ETH) daily volatility index
    └── processed/
        ├── prob_mid_aligned.parquet   # All 53 markets on a common 1-min UTC index
        ├── jump_signals_D1.parquet    # D1 signals for all markets (wide format)
        ├── jump_signals_D2.parquet
        ├── jump_signals_D3.parquet
        └── jump_signals_D4.parquet
```

---

## 3. Environment Setup

### Requirements

```bash
pip install -r requirements.txt
```

Key packages: `pandas>=2.0`, `numpy>=1.24`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `requests`, `pyarrow`, `pyyaml`, `tqdm`, `python-dotenv`, `binance-historical-data`

**Python version**: 3.9+ required. The project was developed on Python 3.9.

### API Keys

Create a `.env` file in the project root (never commit this):

```
KALSHI_API_KEY=your_kalshi_bearer_token
LUNARCRUSH_API_KEY=your_lunarcrush_key   # optional — data not yet collected
```

The Kalshi API key is a Bearer token from `kalshi.com → Account → API`. The free tier supports the endpoints used here.

### Config

All strategy parameters live in `config.yaml`. The key sections:

```yaml
signal:
  d1_lookback_window: 30        # bars for rolling z-score
  d1_z_threshold: 2.0           # z-score threshold for D1/D4
  d1_min_abs_change: 0.02       # minimum 2pp absolute move required
  d2_abs_threshold: 0.05        # 5pp absolute threshold for D2
  d3_rel_threshold: 0.20        # 20% relative change for D3
  d4_volume_ratio_min: 1.5      # D4: volume must be >= 1.5x rolling mean
  cooldown_minutes: 60          # 60-bar cooldown between signals (same market)

liquidity_filter:
  min_active_days: 30
  min_daily_volume_usd: 500     # contracts/day (not dollars — ~$500 at avg $0.50/contract)
  max_bid_ask_spread_pct: 1.0   # effectively disabled — pred market spreads are always wide
  min_open_interest_usd: 25000

backtest:
  holding_periods: [10, 30, 60, 240]   # minutes
  position_size_pct: 0.01              # 1% NAV per trade
  stop_loss_vol_mult: 1.5              # exit if loss > 1.5x pre-event realized vol
  commission_rt: 0.00075               # 0.075% round-trip commission
  slippage: 0.00025                    # 0.025% one-way slippage
```

---

## 4. Data Overview

### Prediction Markets (Kalshi)

53 markets passed the liquidity filter. They fall into two groups:

**Tier 1 — Direct BTC price markets (33 markets):**
- `KXBTCMAXY-*` — "Will BTC reach $X this year?" (year-end price bracket markets)
- `KXBTCMAX100-*` — "Will BTC hit $100K by month X?"
- `KXBTCMAX150-*` — "Will BTC hit $150K by month X?"
- `KXBTCRESERVE-27-JAN01` — Bitcoin strategic reserve market

**Tier 3 — Macro markets (20 markets):**
- `KXCPI-*` — Monthly CPI release markets
- `KXFED-*` — FOMC rate decision markets

**Data range:** November 2024 – April 2026 (only available period for most markets)

**Critical limitation:** All 53 markets have >50% missing 1-minute bars. The most liquid markets (KXBTCMAXY-25-DEC31-149999.99) only have 34.6% 1-min coverage. This is structural — Kalshi's prediction markets are not continuously quoted at the 1-minute resolution; they only record candlesticks when trades occur.

| Market Group | Best Coverage | Typical Coverage |
|---|---|---|
| KXBTCMAXY (year-end) | 34.6% | 10–33% |
| KXBTCMAX100 (monthly) | 14.3% | 5–14% |
| KXCPI / KXFED (macro) | 4.9% | 0.4–4.9% |

### Crypto Spot Data (Binance)

1-minute OHLCV bars for BTC/ETH/SOL from January 2022 through March 2026 (~2.2M bars per symbol). Downloaded from `data.binance.vision` (free, no API key needed). Stored as monthly CSVs and compiled parquets.

Data quality: excellent. Zero-volume bars: 72 (BTCUSDT), 72 (ETHUSDT), 267 (SOLUSDT). One gap >5min per symbol (likely exchange maintenance).

### Deribit DVOL

Daily BTC and ETH implied volatility index. **Warning:** only 384 bars fetched (2026-03-23 to 2026-04-08) — the fetcher is only pulling the most recent data. Historical DVOL back to 2021 is available via `GET /public/get_volatility_index_data` with a `start_timestamp` parameter. This needs to be fixed before DVOL can be used as a conditioning variable.

### LunarCrush Sentiment

**Not collected** — API returned 401 (unauthorized). Need a paid LunarCrush API key or an alternative sentiment source (Santiment, CryptoPanic RSS, Twitter/X academic API).

---

## 5. Notebook Guide

Run notebooks in order. Each saves outputs to `data/processed/` that the next notebook reads.

### 00 — Market Selection (`00_market_selection.ipynb`)

**Status: Complete. Outputs saved.**

1. Queries Kalshi API for crypto-relevant series (KXBTCMAXY, KXBTCMAX100, KXFED, KXCPI, etc.)
2. Classifies 316 markets into Tier 1/2/3/4 using keyword matching from `config.yaml`
3. Fetches daily candlestick data for each candidate and applies liquidity filter
4. **Output**: `data/raw/kalshi/final_market_list.parquet` — 53 markets that passed

**Liquidity filter results:** 53/316 markets passed. The filter was deliberately loose (spread check disabled, volume threshold set at 500 contracts/day) because prediction market spreads are structurally wide and not comparable to traditional markets.

### 01 — Data Collection (`01_data_collection.ipynb`)

**Status: Complete. Outputs saved.**

1. Fetches 1-minute candlesticks for all 53 markets via Kalshi API
2. Downloads monthly BTC/ETH/SOL 1-min CSVs from Binance data archive (2022–2026)
3. Fetches Deribit DVOL (partial — see Known Issues)
4. Attempts LunarCrush sentiment (failed — 401)
5. Aligns all 53 Kalshi markets to a common 1-min UTC DatetimeIndex
6. **Output**: `data/raw/kalshi/*.parquet` (53 files), `data/raw/crypto/crypto_1m_*.parquet` (3 files), `data/processed/prob_mid_aligned.parquet`

### 02 — EDA (`02_eda.ipynb`)

**Status: Complete. No file outputs (read-only analysis).**

Key findings:
- **ALL 53 markets have >50% missing 1-min bars** — this is a structural issue with the data, not a collection bug. Kalshi does not produce a candlestick for every minute; only minutes with actual trades.
- **ADF tests**: probability series are non-stationary (as expected for bounded random walks); first differences are stationary. Log returns are stationary.
- **Daily correlations**: KXBTCMAXY and KXBTCMAX100 markets show strong positive correlation with BTC daily returns (r=0.68–0.84 for the most liquid year-end and $100K target markets). CPI/FED markets show near-zero or weakly negative correlations.
- **Top 5 most correlated markets** (keep these as primary universe):
  1. KXBTCMAXY-26DEC31-109999.99 (r=0.843)
  2. KXBTCMAX100-26-SEP (r=0.799)
  3. KXBTCMAXY-26DEC31-99999.99 (r=0.780)
  4. KXBTCMAX100-26-JUNE (r=0.773)
  5. KXBTCMAX100-26-MAY (r=0.763)

### 03 — Signal Construction (`03_signal.ipynb`)

**Status: Complete. Outputs saved.**

Computes all four jump definitions for every market. Example signal counts for the most liquid market (KXBTCMAXY-25-DEC31-149999.99): D1=143, D2=27, D3=26, D4=123.

**Warning — see Known Issues #1**: The saved parquet files each report ~19.8M total events, which is the total number of rows across all markets (the full time series), not the number of non-zero signal events. The actual signal events are much fewer. Do not rely on the "total events" count from the parquet file shape.

### 04 — Lead-Lag Analysis (`04_lead_lag.ipynb`)

**Status: Framework complete. Analysis NOT yet executed (cells have no output).**

Tests each (market, definition) pair:
1. **Cross-correlation function (CCF)** at lags [-60, -30, -10, -5, -1, 0, 1, 5, 10, 30, 60] minutes
2. **Granger causality** (up to 30 lags) with Bonferroni correction across all tests
3. **Lead-lag regression** by horizon for the best pair
4. **Three placebo tests**: time shuffle, random jump placebo, cross-asset specificity

**Decision gate**: If no definition shows significant lead-lag (p<0.05 before Bonferroni), the hypothesis is rejected and you stop.

This notebook MUST be run next. It is the critical gate.

### 05 — Event Study (`05_event_study.ipynb`)

**Status: Framework complete. Analysis NOT yet executed.**

Pools all markets, computes cumulative abnormal return (CAR) from T=-60 to T=+120 minutes around each detected jump. Separates by direction (+1/-1) and plots mean CAR with confidence bands.

**Decision gate**: If peak mean CAR < 0.3% net of costs at the best post-event horizon → strategy is not viable.

### 06 — Options Strategy (`06_options.ipynb`)

**Status: Notebook exists but is malformed (JSON parse error). Contents unknown.**

Intent: explore whether options/perpetuals around the signal offer better risk-adjusted returns than spot. Needs to be re-created or the notebook fixed.

### 07 — Backtesting (`07_backtest.ipynb`)

**Status: Framework complete. Analysis NOT yet executed.**

Strategy S1 (spot momentum): go long/short BTC at T+1 after signal fires, hold H minutes.
- Tests all combinations of D1/D2/D3/D4 × holding periods [10, 30, 60, 240 min]
- Chronological train/val/test split: 60%/20%/20% of events
- Selects best holding period on validation set, evaluates on test set once
- Walk-forward validation with 5 windows (6-month train, 2-month test, 7-day gap)

**Bug in walk-forward cell**: The lambda in `max(HOLDING_PERIODS, key=lambda h: ...)` has a `...` placeholder — the key function body is missing. Fix before running.

### 08 — Definition Comparison (`08_comparison.ipynb`)

**Status: Framework complete. Analysis NOT yet executed.**

Loads `backtest_results.parquet` from notebook 07 and produces:
- Head-to-head comparison table (Sharpe, Sortino, max drawdown, CAGR, win rate)
- OOS/IS Sharpe ratio check (must be >0.5 to detect overfitting)
- Parameter sensitivity analysis (±20% perturbation of the primary threshold)
- Winner selection: highest OOS Sharpe among definitions passing all minimum bars

---

## 6. Source Modules

### `src/fetch.py`

Data fetchers for all sources. Key functions:
- `fetch_kalshi_candlesticks(ticker, start_ts, end_ts, period_interval, is_historical, api_key)` — Handles both live and historical market endpoints
- `load_binance_bulk(symbol, data_dir)` — Loads and concatenates monthly Binance CSVs into one DataFrame
- `fetch_deribit_dvol(currency)` — Needs fixing to accept a `start_timestamp` parameter for historical data
- `fetch_lunarcrush_sentiment(coin, api_key)` — Returns empty (needs valid API key)

### `src/signal.py`

Four jump detection definitions — see Section 7 below.

### `src/lead_lag.py`

- `compute_ccf(x, y, lags)` — Pearson cross-correlation at specified lags
- `run_granger(delta_p, ret, maxlag)` — Granger causality via statsmodels
- `bonferroni_correct(p_values, n_tests)` — Bonferroni correction
- `placebo_time_shuffle(signal, ret, n_simulations, lags)` — Shuffle timestamps to build null distribution
- `placebo_random_jumps(signal, ret, n_simulations, lags)` — Random jump timing placebo
- `placebo_cross_asset(signal, ret_target, ret_other, lags)` — Cross-asset specificity check
- `lead_lag_regression(signal, ret, horizons)` — OLS regression of forward return on signal at each horizon

### `src/event_study.py`

- `full_event_study(signal, log_ret, pre, post, normalize, vol_window)` — Aggregated CAR by direction
- `subsample_analysis(signal, log_ret, btc_ret)` — CAR by time sub-period (to check regime stability)

### `src/backtest.py`

- `run_backtest(signals, crypto_ret, holding_period, commission_rt, slippage, stop_loss_vol_mult)` — Vectorized S1 backtest. Uses `signal.shift(1)` to prevent look-ahead bias.
- `train_val_test_split(signals, train_frac, val_frac)` — Chronological split by event timestamps
- `walk_forward(signals, crypto_ret, holding_period, n_windows, train_months, test_months, gap_days)` — Walk-forward validation

### `src/metrics.py`

- `full_report(net_ret, gross_returns, benchmark_returns, label)` — Returns a Series with all required metrics: CAGR, annualized vol, Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, cost drag
- `passes_minimum_bars(report)` — Checks Sharpe>1.0, MaxDD<20%, trades>=30

### `src/plots.py`

Standard visualization functions: `plot_probability_series`, `plot_ccf`, `plot_car`, `plot_equity_curve`, `plot_drawdown`, `plot_rolling_sharpe`, `plot_monthly_returns_heatmap`, `plot_gross_vs_net`, `plot_definition_comparison`, `plot_signal_count_vs_sharpe`.

---

## 7. Signal Definitions

All definitions operate on `prob_mid` (the midpoint of bid and ask probability, 0–1 scale). A signal of `+1` means the prediction market probability jumped UP (bullish crypto signal); `-1` means it jumped DOWN (bearish).

A 60-bar cooldown is applied after each signal fires to enforce approximate independence between events.

| Definition | Logic | Best For |
|---|---|---|
| **D1** | delta_p exceeds 2σ of a 30-bar rolling distribution AND \|delta_p\| > 2pp | Adaptive; handles varying market activity levels |
| **D2** | \|delta_p\| >= 5pp in a single bar | Simple, interpretable; needs liquid markets |
| **D3** | \|delta_p / prev_p\| >= 20% relative change | Markets at extreme probabilities (near 0 or 1) |
| **D4** | D1 conditions + volume >= 1.5x rolling mean volume | High-conviction filter; fewest signals, highest precision |

Signal counts per market range: D1: 5–143, D2: 1–50, D3: 1–71, D4: 0–123.

---

## 8. Known Issues and Bugs

### Critical (block running downstream notebooks)

**Bug 1 — Walk-forward lambda placeholder (notebook 07, cell 5)**

The line `max(HOLDING_PERIODS, key=lambda h: ...)` has `...` as a placeholder. This will crash. Fix:
```python
max(HOLDING_PERIODS, key=lambda h: val_reports.loc[val_reports['holding_period'] == h, 'sharpe'].values[0] if len(val_reports.loc[val_reports['holding_period'] == h]) > 0 else -np.inf)
```

**Bug 2 — Signal parquet event counts are inflated (~19.8M per definition)**

In notebook 03, cell 7, the "total events" count is the number of rows in the full DataFrame (all timestamps × all markets), not the count of non-zero signals. The actual number of signal events per market is 1–143 for D1. When you load these parquets downstream, use `(df != 0).sum()` to get actual event counts. This is not a data corruption bug — the parquets are structurally fine (wide format: rows=timestamps, columns=markets, values=signal). But the print statement is misleading.

**Bug 3 — Notebook 06 (`06_options.ipynb`) is malformed**

The notebook has a JSON parse error and cannot be read by the Jupyter engine. It needs to be re-opened in a text editor, the malformed cell identified, and fixed or deleted.

### Data Issues

**Issue 4 — Deribit DVOL historical data missing**

`fetch_deribit_dvol()` only returns the most recent ~384 bars. The function needs a `start_timestamp` parameter added. Fix in `src/fetch.py`:
```python
def fetch_deribit_dvol(currency: str, start_ts: int = None) -> pd.DataFrame:
    # Add: params['start_timestamp'] = start_ts * 1000 (Deribit uses milliseconds)
```
Historical DVOL back to 2021 is available from `https://www.deribit.com/api/v2/public/get_volatility_index_data`.

**Issue 5 — LunarCrush sentiment not collected**

The free LunarCrush v4 API requires authentication. Either get a paid key or replace with an alternative: Santiment free tier, CryptoPanic RSS feed, or Bitcoin Fear & Greed Index (free, daily).

**Issue 6 — Extreme 1-minute data sparsity (structural)**

All 53 Kalshi markets have >50% missing 1-min bars. This is not a collection error — Kalshi only generates a candlestick for a minute when at least one trade occurs. The sparsity is real and structural.

Options to address this:
1. **Forward-fill probabilities**: treat the market as "last traded price" between trades (standard market microstructure approach). Use `prob_df.ffill()` before signal computation.
2. **Coarser time resolution**: aggregate to 5-min or 15-min bars. At 15-min, most markets will have near-complete coverage.
3. **Restrict to active trading hours**: crypto trades 24/7 but Kalshi prediction markets have lower liquidity overnight. Consider filtering to US market hours (13:00–21:00 UTC) where Kalshi liquidity is higher.

**Issue 7 — CPI/FED markets show no BTC correlation**

The 20 macro markets (KXCPI-*, KXFED-*) show near-zero or negative daily correlations with BTC (r range: -0.28 to +0.16). They are unlikely to be useful for the spot momentum strategy in their current form. Consider:
- Using them as regime filters rather than entry signals (e.g., "only trade during high FOMC uncertainty")
- Dropping them from the universe entirely and focusing on the 33 Tier 1 BTC price markets

**Issue 8 — Market universe correlation (pooling assumption)**

When pooling signals across markets, the code assumes signals from different markets are independent. This is false: KXBTCMAXY-25-DEC31-149999.99 (BTC hits $150K in 2025) and KXBTCMAXY-25-DEC31-129999.99 (BTC hits $130K in 2025) will both move when BTC price moves. They are structurally correlated. Pooling gives the illusion of more events than there are independent informational signals.

**Recommendation**: Run lead-lag analysis on the top 5 most liquid, most BTC-correlated markets individually first. Do not pool before demonstrating individual signal quality.

---

## 9. Current Results

### What is complete

- Market selection and liquidity filtering — 53 qualifying markets identified
- Full data collection — all raw data saved to parquet
- EDA and data quality validation
- Signal computation — D1/D2/D3/D4 computed for all markets and saved

### What is NOT yet run

**Notebooks 04, 05, 07, 08 have no outputs.** The entire analytical pipeline from lead-lag analysis through backtesting has not been executed. The strategy has not been validated.

### Preliminary signal from EDA (daily resolution only)

Strong daily co-movement between Kalshi BTC price markets and BTC spot returns:

| Market | Daily Pearson r with BTC | p-value |
|---|---|---|
| KXBTCMAXY-26DEC31-109999.99 | 0.843 | <0.001 |
| KXBTCMAX100-26-SEP | 0.799 | <0.001 |
| KXBTCMAXY-26DEC31-99999.99 | 0.780 | <0.001 |
| KXBTCMAX100-26-JUNE | 0.773 | <0.001 |
| KXBTCMAX100-26-MAY | 0.763 | <0.001 |

**Important caveat**: This daily correlation is contemporaneous — it shows that prediction markets and spot prices move together on a given day. It does NOT show that prediction markets lead spot prices. That directional evidence requires the Granger analysis in notebook 04.

---

## 10. Next Steps — Priority Order

### Step 1 (Immediate): Fix bugs and run the pipeline

1. Fix the walk-forward lambda bug in notebook 07, cell 5.
2. Fix notebook 06 (options notebook) — open in a text editor, find the malformed cell.
3. Run notebooks 04 → 05 → 07 → 08 in sequence. Read the decision gates carefully:
   - If notebook 04 Granger test shows no significance → **stop the entire project**, hypothesis rejected.
   - If notebook 05 CAR < 0.3% net → **stop**, no tradeable edge.
   - If notebooks 07/08 produce Sharpe < 1.0 on test set → go back to signal design.

### Step 2: Address data sparsity before running the pipeline

Before running notebook 04, decide on and implement one of these approaches:
- **Option A (recommended)**: Forward-fill `prob_df` before computing signals. This is the correct microstructure treatment. Add `prob_df = prob_df.ffill()` at the start of notebook 03 before `compute_all_signals`.
- **Option B**: Aggregate to 15-min bars. This reduces the total universe to ~2,800 bars per market but eliminates the sparsity problem.
- **Option C**: Restrict to the top 5 most liquid markets only (drop the macro markets and low-coverage KXBTCMAX markets).

### Step 3: Improve Deribit DVOL data

Fix `fetch_deribit_dvol()` in `src/fetch.py` to accept a `start_timestamp` parameter and re-run data collection for 2022–2026 DVOL history. Use DVOL as a conditioning variable: hypothesize that signals are stronger during high-volatility regimes.

### Step 4: Focus the market universe

Based on notebook 02 daily correlations, restrict the primary analysis to:
- **5 BTC price markets** with r > 0.75 against BTC daily returns (listed in Section 9)
- Treat macro markets (CPI/FED) as regime filters, not entry signals

### Step 5: Add alternative sentiment data

Replace LunarCrush (requires paid API) with one of:
- **Bitcoin Fear & Greed Index** (free, daily via `alternative.me/crypto/fear-and-greed-index/history/`)
- **Santiment free tier** (limited hourly social sentiment)
- **CryptoPanic RSS** (free, requires parsing)

### Step 6: Investigate options strategy (notebook 06)

Once notebook 07 establishes whether spot momentum works, explore:
- Trading BTC options (Deribit) or perpetuals (Binance) instead of spot — offers leverage and asymmetric payoffs
- Specifically: buying a call/put spread around a predicted directional move
- The options notebook needs to be rebuilt from scratch given the parse error

### Step 7: Live trading infrastructure (only after strategy is validated)

If backtesting shows net Sharpe > 1.5 on the test set:
1. Kalshi WebSocket feed for real-time probability monitoring (`wss://api.elections.kalshi.com/trade-api/ws/v2`)
2. Binance WebSocket for live 1-min candlestick subscription
3. Order execution: Binance REST API for spot, or Binance USDT-M futures for leverage
4. Position sizing: 1% NAV per trade (from config), hard stop at 1.5x pre-event vol

### Step 8: Parameter robustness

Before declaring the strategy viable, confirm:
- OOS Sharpe / IS Sharpe > 0.5 (notebook 08 already tests this)
- ±20% parameter perturbation still yields Sharpe > 1.0 (notebook 08 parameter sensitivity)
- At least 30 independent trades in the test set (minimum statistical significance)

---

## 11. Strategy Viability Checklist

The strategy is only worth pursuing live if ALL of the following pass:

| Check | Minimum Bar | Status |
|---|---|---|
| Lead-lag significance (Granger) | At least one definition p<0.05 | NOT YET RUN |
| Peak CAR (net of costs) | > 0.3% | NOT YET RUN |
| Net Sharpe (test set) | > 1.0 (target > 1.5) | NOT YET RUN |
| Max Drawdown | < 20% | NOT YET RUN |
| OOS/IS Sharpe ratio | > 0.5 | NOT YET RUN |
| Minimum independent trades in test | ≥ 30 | NOT YET RUN |
| Parameter sensitivity (±20%) | Robust | NOT YET RUN |
| Explainable edge | Can articulate why the inefficiency exists | Partial — see Section 1 |

---

## 12. Academic Grounding

**Prediction market literature:**
- Arrow et al. (2008) — "The promise of prediction markets," *Science* — core reference for why prediction markets aggregate information efficiently
- Wolfers & Zitzewitz (2004) — "Prediction markets," *JEP* — foundational review
- Manski (2006) — "Interpreting the predictions of prediction markets" — interpretation caveats

**Lead-lag and price discovery:**
- Glosten & Milgrom (1985) — Adverse selection in bid-ask spreads — why informed traders may reveal information in lower-liquidity venues first
- Easley et al. (2012) — VPIN — framework for detecting informed trading via order flow

**Backtesting methodology:**
- Bailey & López de Prado (2014) — The Deflated Sharpe Ratio — use when testing multiple definitions to correct for selection bias
- Harvey, Liu & Zhu (2016) — "...and the Cross-Section of Expected Returns" — multiple testing problem in strategy research

**Crypto-specific:**
- There is limited academic literature on prediction market → crypto lead-lag specifically. The closest analogues are studies on derivatives/spot lead-lag (futures leading spot) which show futures/derivatives often lead spot in price discovery.

---

## Contributing / Handover Notes

This project was started by Kian Jagtiani. The entire data pipeline (notebooks 00–03) is complete and validated. The core analytical pipeline (notebooks 04–08) is scaffolded but has not been run yet — **this is where the work needs to happen.**

The most important thing to run first is notebook 04 (lead-lag). The Granger causality result there is the binary go/no-go decision for the entire project. If it fails, the hypothesis is rejected and there is no point running anything else. If it passes (even weakly, at raw p<0.05 for any definition), proceed to notebooks 05 and 07.

The data is all local (no re-fetching needed unless you want to extend the date range). The Kalshi data runs through April 2026 and the Binance crypto data through March 2026.
