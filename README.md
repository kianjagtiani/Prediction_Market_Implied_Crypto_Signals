# Prediction Markets as a Crypto Signal

Do sharp moves in Kalshi prediction-market odds tell you anything about where BTC is
headed over the next few hours? We pulled minute-level candlesticks for 53 Kalshi markets
(BTC price targets, CPI, Fed decisions), flagged sudden probability jumps, and tested
whether those jumps lead BTC/ETH/SOL spot returns.

Spring '26 crypto desk project. Kian Jagtiani (kjagtian@usc.edu).

## What we found, notebook by notebook

**00–01 · Market selection and data.** Filtered the Kalshi catalog down to 53 markets
with ≥30 active days, a few $k of daily volume, workable spreads, and 1-min data —
BTC price targets ("$100K by June?"), regulatory events, and CPI/Fed markets — plus
Binance 1-min BTC/ETH/SOL bars and Deribit DVOL.

**02 · EDA.** Kalshi 1-min data is sparse: a bar only prints when a trade happens, and
even the most liquid market covers ~35% of minutes. The BTC-target markets track BTC
daily returns closely (r up to 0.84); the CPI/Fed markets show roughly zero correlation
with BTC and drop out of the story here.

**03 · Signals.** Four jump definitions on the 1-min probability mid, each firing +1/−1
with a 60-min per-market cooldown. Event counts across all 53 markets:

- D1 — rolling z-score > 2σ and ≥ 2pp (2,293 events)
- D2 — ≥ 5pp absolute move in one bar (618)
- D3 — ≥ 20% relative move (874)
- D4 — D1 plus volume ≥ 1.5× its rolling mean (1,323)

**04 · Lead-lag.** The core result. Of the 60 (market, definition) pairs with 50+
signals, 43 Granger-cause BTC returns at p < 0.05 and 25 survive Bonferroni; for the top
markets p ≈ 0. D4 is the cleanest signal, leading BTC by 23–30 minutes (D1: 6–45 min;
D2/D3 have little power). Time-shuffle, random-jump, and cross-asset placebos all come
back null, so this isn't a mechanical artifact — though bracket markets on the same
underlying are correlated, so pooled counts overstate how many independent signals exist.

**05 · Event study.** Where the trade dies. Direction-adjusted cumulative abnormal
return over a 240-min window, after a 10 bp cost assumption, is not significantly
positive for any definition in the top-5 markets. Spot has already moved by the first
bar after the jump: the information edge is real, but it's consumed within a minute.

**06 · DVOL.** Scaffold for checking whether Deribit implied vol reacts around jump
events — parked, since the strategy didn't clear the gate above.

**07–08 · Backtest.** Enter BTC long/short one bar after the signal, hold H minutes,
full costs plus a vol-scaled stop, 60/20/20 chronological split. Every definition loses
money at 10–60 min holds. At H=240:

| Definition | Sharpe | Max DD | Trades |
|---|---|---|---|
| D2 | 1.29 | 5.1% | 44 |
| D3 | 1.21 | 7.9% | 65 |
| D4 | 0.94 | 9.1% | 66 |
| D1 | 0.37 | 16.6% | 30 |

That looks like slow post-event drift rather than the 1-minute edge, and it isn't strong
enough to trade given the event-study result. (Ignore the Sharpe 8–13 printout in
notebook 07's stored output — an artifact of the sparse pooled run; these are the final
numbers. Gate-by-gate detail: `reports/focused_research_report.md`.)

## Bottom line

Kalshi jumps carry genuine information about BTC's next few minutes — the statistics are
unambiguous. But Kalshi's public candlesticks are 1-minute, so by the time a jump is
visible the move has happened, and a T+1 entry captures nothing after costs. To make
this tradeable you'd need sub-minute data: scrape Kalshi's websocket at ~10s resolution
for a month or two and re-run notebooks 04–07 at that resolution. Extending to ETH/SOL
and checking whether the lag shifts across market regimes are the other open threads.

The full findings are in `reports/final_presentation_strategy3.pdf`.

## What's here

- `notebooks/00`–`08` — the pipeline, in the order above; each writes to
  `data/processed/` for the next
- `src/` — Kalshi/Binance/Deribit fetchers, jump definitions, CCF + Granger + placebo
  tests, event study, vectorized backtest with walk-forward, metrics, plots
- `scripts/run_focused_pipeline.py` — end-to-end rerun on the 5-market primary universe
- `reports/` — final presentation (our strategy's section) and the focused-rerun writeup
- `data/processed/` + `data/figures/` — signal parquets and all plots (tracked in git;
  `data/raw/` is ~1.2 GB and is not)
- `build_slides.py` — builds `data/strategy2_slides.pptx` from the saved figures

## Setup

```bash
pip install -r requirements.txt
```

Put a Kalshi key in `.env` (`KALSHI_API_KEY=...`, from kalshi.com → Account → API).
Binance 1-min data comes from data.binance.vision and needs no key; Deribit DVOL is
public. Raw data isn't in the repo — notebooks 00–01 rebuild it (the Kalshi candlesticks
are the slow part). Strategy parameters live in `config.yaml`.

## References

Wolfers & Zitzewitz (2004), *Prediction Markets*, JEP. Arrow et al. (2008), *The Promise
of Prediction Markets*, Science. Bailey & López de Prado (2014), *The Deflated Sharpe
Ratio* — the reason for the multiple-testing caution above.
