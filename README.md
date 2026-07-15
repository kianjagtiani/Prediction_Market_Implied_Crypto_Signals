# Prediction Markets as a Crypto Signal

Spring '26 Cryptocurrency Desk Project \
**Desk Head:** Kian Jagtiani \
**Members on Project:** Kian Jagtiani, James Gui

---

Do sharp moves in Kalshi prediction-market odds tell you anything about where BTC is
headed over the next few hours? We pulled minute-level candlesticks for 53 Kalshi markets
(BTC price targets, CPI, Fed decisions), flagged sudden probability jumps, and tested
whether those jumps lead BTC/ETH/SOL spot returns.

## What we found, notebook by notebook

**00–01 · Market selection and data.** Filtered the Kalshi catalog down to 53 markets
with ≥30 active days, a few $k of daily volume, workable spreads, and 1-min data:
BTC price targets ("$100K by June?"), regulatory events, and CPI/Fed markets. Alongside
these, Binance 1-min BTC/ETH/SOL bars and Deribit DVOL.

**02 · EDA.** Kalshi 1-min data is sparse: a bar only prints when a trade happens, and
even the most liquid market covers ~35% of minutes. The BTC-target markets track BTC
daily returns closely (r up to 0.84); the CPI/Fed markets show roughly zero correlation
with BTC and drop out of the story here.

**03 · Signals.** Four jump definitions on the 1-min probability mid, each firing +1/−1
with a 60-min per-market cooldown. Event counts across all 53 markets:

- D1: rolling z-score > 2σ and ≥ 2pp (2,293 events)
- D2: ≥ 5pp absolute move in one bar (618)
- D3: ≥ 20% relative move (874)
- D4: D1 plus volume ≥ 1.5× its rolling mean (1,323)

**04 · Lead-lag.** The core result. Of the 60 (market, definition) pairs with 50+
signals, 43 Granger-cause BTC returns at p < 0.05 and 25 survive Bonferroni; for the top
markets p ≈ 0. D4 is the cleanest signal, leading BTC by 23–30 minutes (D1: 6–45 min;
D2/D3 have little power). Time-shuffle, random-jump, and cross-asset placebos all come
back null, so this isn't a mechanical artifact. One caveat: bracket markets on the same
underlying are correlated, so pooled counts overstate how many independent signals exist.

**05 · Event study.** Where the trade dies. Direction-adjusted cumulative abnormal
return over a 240-min window, after a 10 bp cost assumption, is not significantly
positive for any definition in the top-5 markets. Spot has already moved by the first
bar after the jump: the information edge is real, but it's consumed within a minute.

**06 · DVOL.** Scaffold for checking whether Deribit implied vol reacts around jump
events. Parked, since the strategy didn't clear the gate above.

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
notebook 07's stored output; it's an artifact of the sparse pooled run, and these are
the final numbers. Gate-by-gate detail: `reports/focused_research_report.md`.)

## Conclusion and next steps

Kalshi jumps carry genuine information about BTC's next few minutes; the statistics are
unambiguous. But the public 1-minute candles are too coarse to trade on. By the time a
jump is visible the move has already happened, and an entry at the next bar captures
nothing after costs.

What it would take to make this a profitable strategy:

- Scrape sub-minute Kalshi data (the websocket updates at ~10s resolution) for a month
  or two and re-run the lead-lag → event study → backtest chain on it. The edge lives
  inside the first minute, so this is where the P&L would be.
- Extend the pipeline to ETH and SOL, whose Kalshi markets sat outside the primary universe.
- Segment by market regime to check whether the lag (and the slow drift the H=240
  backtest picks up) is stable enough to size positions on.
- Paper trade only once the sub-minute re-run clears the event-study and OOS Sharpe gates.

The full findings are in `reports/final_presentation_strategy3.pdf`.

## Repository contents

- `notebooks/00`–`08`: the analysis, in the order above
- `src/`: data fetchers, jump definitions, CCF/Granger/placebo tests, event study,
  vectorized backtest with walk-forward, metrics, plots
- `scripts/run_focused_pipeline.py`: end-to-end rerun on the 5-market primary universe
- `reports/`: final presentation (our strategy's section) and the focused-rerun writeup
- `data/processed/` + `data/figures/`: signal parquets and every figure in the analysis

The ~1.2 GB of raw Kalshi/Binance/Deribit data is not tracked; notebooks 00–01 document
exactly how it was collected, and all parameters live in `config.yaml`.

## References

Wolfers & Zitzewitz (2004), *Prediction Markets*, JEP. Arrow et al. (2008), *The Promise
of Prediction Markets*, Science. Bailey & López de Prado (2014), *The Deflated Sharpe
Ratio*, the reason for the multiple-testing caution above.
