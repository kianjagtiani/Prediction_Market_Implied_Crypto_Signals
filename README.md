# Prediction Markets as a Crypto Signal

Do sharp moves in Kalshi prediction-market odds tell you anything about where BTC is
headed over the next few hours? We pulled minute-level candlesticks for 53 Kalshi markets
(BTC price targets, CPI, Fed decisions), flagged sudden probability jumps, and tested
whether those jumps lead BTC/ETH/SOL spot returns.

Spring '26 crypto desk project. Kian Jagtiani (kjagtian@usc.edu).

## Results

The short version: the lead-lag effect is statistically real, but it doesn't survive costs
at a T+1 entry. Full findings are in `reports/final_presentation_strategy3.pdf`; the
gate-by-gate detail is in `reports/focused_research_report.md`.

- Restricting to markets with 50+ signals (60 market/definition pairs), 43 pairs show
  significant Granger causality from prediction-market jumps to BTC returns at p < 0.05,
  25 after Bonferroni. For the top markets p ≈ 0.
- D4 (z-score + volume filter) is the strongest definition, with best lags of 23–30
  minutes. D1 leads at 6–45 minutes; D2 and D3 have little predictive power.
- The event study kills the naive trade: direction-adjusted CAR over a 240-min window with
  a 10 bp cost assumption fails for all four definitions in the top-5 markets. This doesn't
  invalidate the Granger result — the edge lives in the 0–1 minute window before a T+1
  entry can capture it.
- Backtests (enter at T+1, hold H minutes, full costs + vol stop) are negative at 10–60 min
  holds for every definition. At H=240 the best is D2 at Sharpe 1.29 (D3 1.21) with max
  drawdown under 8% — decent-looking, but not enough given the event-study evidence.
- Ignore the Sharpe 8–13 printout in notebook 07's stored test-split output — that came
  from the sparse pooled run and is an artifact; the presentation numbers above are the
  final ones.

## What's here

- `notebooks/00`–`08` — the pipeline, in order: market selection, data collection, EDA,
  signal construction, lead-lag tests, event study, DVOL/options check, backtest, comparison
- `src/` — Kalshi/Binance/Deribit fetchers, the four jump definitions, CCF + Granger +
  placebo tests, event study, vectorized backtest with walk-forward, metrics, plots
- `scripts/run_focused_pipeline.py` — end-to-end rerun on the 5-market primary universe;
  regenerates the report
- `reports/final_presentation_strategy3.pdf` — final desk presentation (our strategy's section)
- `reports/focused_research_report.md` — writeup of the focused 5-market rerun
- `data/processed/` + `data/figures/` — aligned probabilities, signal parquets, all plots
  (tracked in git; `data/raw/` is ~1.2 GB and is not — see Setup)
- `build_slides.py` — builds `data/strategy2_slides.pptx` from the saved figures and results

## Signal definitions

All four operate on the 1-min probability mid, fire +1/−1 for direction, and enforce a
60-minute cooldown per market:

- **D1** — change exceeds 2σ of a 30-bar rolling window, and at least 2pp
- **D2** — at least 5pp absolute change in one bar
- **D3** — at least 20% relative change
- **D4** — D1 plus volume ≥ 1.5× its rolling mean

Backtests trade BTC spot at the bar after the signal, hold 10/30/60/240 min, and charge
0.075% round-trip commission + 0.025% slippage with a vol-scaled stop.

## Setup

```bash
pip install -r requirements.txt
```

Put a Kalshi key in `.env` (`KALSHI_API_KEY=...`, from kalshi.com → Account → API).
Binance 1-min data comes from data.binance.vision and needs no key; Deribit DVOL is public.

Raw data isn't in the repo. Notebooks 00–01 rebuild it (Kalshi candlesticks are the slow
part). With `data/raw/` in place, run the notebooks in order — each writes to
`data/processed/` for the next — or run `python scripts/run_focused_pipeline.py` for the
focused universe end-to-end. Strategy parameters live in `config.yaml`.

## Known limitations

- Kalshi 1-min bars only exist for minutes with trades; even the most liquid market has
  ~35% coverage. Forward-filling vs. not materially changes results (both are reported).
- The CPI/Fed markets show near-zero correlation with BTC and contribute nothing here.
- Bracket markets on the same underlying are structurally correlated, so pooled event
  counts overstate the number of independent signals.

Since the edge sits inside the first minute after a jump, the natural next step is
sub-minute (~10s) Kalshi data, plus extending the pipeline to ETH/SOL and checking whether
the lag shifts across market regimes.

## References

Wolfers & Zitzewitz (2004), *Prediction Markets*, JEP. Arrow et al. (2008), *The Promise
of Prediction Markets*, Science. Bailey & López de Prado (2014), *The Deflated Sharpe
Ratio* — the reason the multiple-testing caveats above exist.
