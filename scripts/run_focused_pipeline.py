"""
Run a focused end-to-end research pipeline for the primary BTC Kalshi markets.

This script avoids notebook execution and large intermediate CSV files. It fetches
Kalshi candlesticks for the top BTC-correlated markets identified in the README,
downloads Binance monthly zip data into memory, computes signals, runs the core
tests, and writes a Markdown report.
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.backtest import run_backtest, train_val_test_split
from src.event_study import aggregate_car, build_event_windows, compute_car
from src.fetch import (
    fetch_deribit_dvol,
    fetch_kalshi_candlesticks,
    save_deribit_dvol,
    save_kalshi_market,
)
from src.lead_lag import (
    compute_ccf,
    lead_lag_regression,
    placebo_cross_asset,
    placebo_random_jumps,
    placebo_time_shuffle,
    run_granger,
)
from src.metrics import full_report
from src.signal import compute_all_signals, signal_summary


START_DATE = "2025-01-01"
END_DATE = "2026-04-01"

PRIMARY_MARKETS = [
    "KXBTCMAXY-26DEC31-109999.99",
    "KXBTCMAX100-26-SEP",
    "KXBTCMAXY-26DEC31-99999.99",
    "KXBTCMAX100-26-JUNE",
    "KXBTCMAX100-26-MAY",
]

DEFINITIONS = ["D1", "D2", "D3", "D4"]
HOLDING_PERIODS = [10, 30, 60, 240]
LAGS = [-60, -30, -10, -5, -1, 0, 1, 5, 10, 30, 60]
HORIZONS = [1, 10, 30, 60, 240, 1440]
BINANCE_BASE = "https://data.binance.vision/data/spot/monthly/klines"


def _ts(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz="UTC").timestamp())


def _month_iter(start: str, end: str) -> Iterable[pd.Timestamp]:
    cur = pd.Timestamp(start, tz="UTC").replace(day=1)
    stop = pd.Timestamp(end, tz="UTC").replace(day=1)
    while cur < stop:
        yield cur
        cur = cur + pd.DateOffset(months=1)


def fetch_binance_monthly(symbol: str, start: str, end: str, out_path: Path) -> pd.DataFrame:
    if out_path.exists():
        return pd.read_parquet(out_path)

    cols = [
        "timestamp_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "n_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "_ignore",
    ]
    frames = []
    for month in _month_iter(start, end):
        filename = f"{symbol}-1m-{month.year}-{month.month:02d}.zip"
        url = f"{BINANCE_BASE}/{symbol}/1m/{filename}"
        print(f"  Binance {symbol}: {filename}")
        resp = requests.get(url, timeout=90)
        if resp.status_code == 404:
            print(f"    missing {filename}, skipping")
            continue
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [name for name in zf.namelist() if name.endswith(".csv")]
            if not csv_names:
                continue
            with zf.open(csv_names[0]) as fh:
                df = pd.read_csv(fh, header=None)

        df.columns = cols[: len(df.columns)]
        ts = pd.to_numeric(df["timestamp_ms"], errors="coerce")
        df = df[ts.notna()].copy()
        ts = ts[ts.notna()]
        if ts.empty:
            continue
        if ts.median() > 1e14:
            df["timestamp_ms"] = (ts / 1000).astype(np.int64)
        else:
            df["timestamp_ms"] = ts.astype(np.int64)
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No Binance data downloaded for {symbol}")

    df = pd.concat(frames, ignore_index=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_ms"].astype(np.int64), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["symbol"] = symbol
    df = df[["symbol", "timestamp_utc", "open", "high", "low", "close", "volume", "log_ret"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def fetch_primary_kalshi(api_key: str, raw_dir: Path) -> dict[str, pd.DataFrame]:
    start_ts = _ts(START_DATE)
    end_ts = _ts(END_DATE)
    out: dict[str, pd.DataFrame] = {}
    for ticker in PRIMARY_MARKETS:
        path = raw_dir / "kalshi" / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
        else:
            print(f"  Kalshi {ticker}")
            df = fetch_kalshi_candlesticks(
                ticker=ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=1,
                api_key=api_key,
            )
            if df.empty:
                print(f"    no data returned for {ticker}")
                continue
            save_kalshi_market(ticker, df)
            time.sleep(0.2)
        out[ticker] = df
        if not df.empty:
            print(f"    {len(df):,} bars, {df.timestamp_utc.min()} to {df.timestamp_utc.max()}")
    return out


def build_probability_panel(kalshi: dict[str, pd.DataFrame], proc_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(START_DATE, tz="UTC")
    end = pd.Timestamp(END_DATE, tz="UTC") - pd.Timedelta(minutes=1)
    full_index = pd.date_range(start=start, end=end, freq="1min")

    probs = {}
    volumes = {}
    coverage_rows = []
    for ticker, df in kalshi.items():
        sdf = df.set_index("timestamp_utc").sort_index()
        sdf = sdf[~sdf.index.duplicated(keep="last")]
        probs[ticker] = pd.to_numeric(sdf["prob_mid"], errors="coerce").reindex(full_index).ffill()
        volumes[ticker] = pd.to_numeric(sdf.get("volume", pd.Series(dtype=float)), errors="coerce").reindex(full_index).fillna(0)
        observed = sdf["prob_mid"].notna().sum()
        coverage_rows.append({
            "ticker": ticker,
            "observed_bars": int(observed),
            "coverage_pct": observed / len(full_index) * 100,
            "first_bar": str(sdf.index.min()) if len(sdf) else "",
            "last_bar": str(sdf.index.max()) if len(sdf) else "",
        })

    prob_df = pd.DataFrame(probs, index=full_index)
    volume_df = pd.DataFrame(volumes, index=full_index)
    proc_dir.mkdir(parents=True, exist_ok=True)
    prob_df.to_parquet(proc_dir / "prob_mid_primary_ffill.parquet")
    volume_df.to_parquet(proc_dir / "volume_primary.parquet")
    pd.DataFrame(coverage_rows).to_csv(proc_dir / "primary_market_coverage.csv", index=False)
    return prob_df, volume_df


def compute_signals(prob_df: pd.DataFrame, volume_df: pd.DataFrame, cfg: dict, proc_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    all_signals: dict[str, dict[str, pd.Series]] = {d: {} for d in DEFINITIONS}
    summary_rows = []

    for ticker in prob_df.columns:
        prob_mid = prob_df[ticker].dropna()
        volume = volume_df[ticker].reindex(prob_mid.index).fillna(0)
        signals = compute_all_signals(prob_mid, volume=volume, cfg=cfg)
        for defn, sig_df in signals.items():
            all_signals[defn][ticker] = sig_df["signal"]
        summary_rows.append(signal_summary(signals, market_id=ticker))

    signal_frames: dict[str, pd.DataFrame] = {}
    for defn, market_signals in all_signals.items():
        frame = pd.DataFrame(market_signals).fillna(0)
        frame.index.name = "timestamp_utc"
        frame.to_parquet(proc_dir / f"jump_signals_primary_{defn}.parquet")
        signal_frames[defn] = frame

    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(proc_dir / "primary_signal_summary.csv", index=False)
    return signal_frames, summary


def pool_signals(sig_df: pd.DataFrame, index: pd.Index) -> pd.Series:
    combined = sig_df.fillna(0).stack()
    combined = combined[combined != 0]
    if combined.empty:
        return pd.Series(0.0, index=index)
    flat = combined.groupby(level=0).apply(lambda x: x.iloc[x.abs().argmax()])
    return flat.reindex(index).fillna(0)


def run_lead_lag(signal_frames: dict[str, pd.DataFrame], btc_ret: pd.Series, eth_ret: pd.Series, proc_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ccf_rows = []
    granger_rows = []
    n_tests = sum((frame.fillna(0).ne(0).sum() >= 30).sum() for frame in signal_frames.values())

    for defn, sig_df in signal_frames.items():
        for market in sig_df.columns:
            signal = sig_df[market].fillna(0)
            n_events = int(signal.ne(0).sum())
            if n_events >= 10:
                ret = btc_ret.reindex(signal.index).fillna(0)
                ccf = compute_ccf(signal, ret, lags=LAGS)
                for row in ccf.to_dict("records"):
                    ccf_rows.append({"market": market, "definition": defn, "n_events": n_events, **row})
            if n_events < 30:
                continue
            try:
                ret = btc_ret.reindex(signal.index).fillna(0)
                gc = run_granger(signal, ret, maxlag=30)
                min_p = float(gc["p_value_f"].min())
                best_lag = int(gc.loc[gc["p_value_f"].idxmin(), "lag"])
                granger_rows.append({
                    "market": market,
                    "definition": defn,
                    "n_events": n_events,
                    "min_p_raw": min_p,
                    "best_lag": best_lag,
                    "min_p_bonferroni": min(min_p * max(n_tests, 1), 1.0),
                })
            except Exception as exc:
                granger_rows.append({
                    "market": market,
                    "definition": defn,
                    "n_events": n_events,
                    "min_p_raw": np.nan,
                    "best_lag": np.nan,
                    "min_p_bonferroni": np.nan,
                    "error": str(exc)[:120],
                })

    ccf_df = pd.DataFrame(ccf_rows)
    gc_df = pd.DataFrame(granger_rows)
    if not gc_df.empty and "min_p_raw" in gc_df:
        gc_df["significant_raw"] = gc_df["min_p_raw"] < 0.05
        gc_df["significant_bonferroni"] = gc_df["min_p_bonferroni"] < 0.05
        gc_df = gc_df.sort_values("min_p_raw", na_position="last")
    ccf_df.to_csv(proc_dir / "primary_ccf_results.csv", index=False)
    gc_df.to_csv(proc_dir / "primary_granger_results.csv", index=False)

    if not gc_df.empty and gc_df["min_p_raw"].notna().any():
        best = gc_df.dropna(subset=["min_p_raw"]).iloc[0]
        signal = signal_frames[best["definition"]][best["market"]].fillna(0)
        ret = btc_ret.reindex(signal.index).fillna(0)
        reg = lead_lag_regression(signal, ret, horizons=HORIZONS)
        reg.insert(0, "definition", best["definition"])
        reg.insert(0, "market", best["market"])
        reg.to_csv(proc_dir / "primary_best_lead_lag_regression.csv", index=False)

        shuffle = placebo_time_shuffle(signal, ret, n_simulations=200, lags=LAGS)
        random_null = placebo_random_jumps(signal, ret, n_simulations=200, lags=LAGS)
        cross = placebo_cross_asset(signal, ret, eth_ret.reindex(signal.index).fillna(0), lags=LAGS)
        shuffle.to_csv(proc_dir / "primary_placebo_time_shuffle.csv", index=False)
        random_null.to_csv(proc_dir / "primary_placebo_random_jumps.csv", index=False)
        cross.to_csv(proc_dir / "primary_placebo_cross_asset.csv", index=False)
    else:
        reg = pd.DataFrame()

    return gc_df, ccf_df


def run_event_studies(signal_frames: dict[str, pd.DataFrame], btc_ret: pd.Series, proc_dir: Path) -> pd.DataFrame:
    rows = []
    for defn, sig_df in signal_frames.items():
        signal = pool_signals(sig_df, btc_ret.index)
        n_events = int(signal.ne(0).sum())
        if n_events < 10:
            continue
        windows, directions = build_event_windows(signal, btc_ret, pre=60, post=max(HOLDING_PERIODS))
        if windows.empty:
            continue

        # Direction-adjusted CAR is the tradeability gate: positive means the
        # post-event BTC move agreed with the signal direction.
        aligned_windows = windows.mul(directions.to_numpy(), axis=0)
        aligned_car = compute_car(aligned_windows)
        agg = aggregate_car(aligned_car)
        agg.to_csv(proc_dir / f"primary_event_study_aligned_{defn}.csv", index=False)

        post = agg[agg["t"] > 0]
        if post.empty:
            continue
        peak = post.loc[post["mean_car"].idxmax()]
        net_peak = float(peak["mean_car"] - 0.001)
        p_value = float(peak["p_value"]) if pd.notna(peak["p_value"]) else np.nan
        rows.append({
            "definition": defn,
            "n_events": int(peak["n_events"]),
            "peak_t": int(peak["t"]),
            "peak_direction_adjusted_car": float(peak["mean_car"]),
            "peak_p_value": p_value,
            "net_peak_car_after_10bp": net_peak,
            "passes_tradeability_gate": bool(net_peak >= 0.003 and p_value < 0.05 and int(peak["n_events"]) >= 30),
        })

    out = pd.DataFrame(rows)
    out.to_csv(proc_dir / "primary_event_study_summary.csv", index=False)
    return out


def run_backtests(signal_frames: dict[str, pd.DataFrame], btc_ret: pd.Series, cfg: dict, proc_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bt_cfg = cfg["backtest"]
    all_reports = []
    test_reports = []

    for defn, sig_df in signal_frames.items():
        signal = pool_signals(sig_df, btc_ret.index)
        n_events = int(signal.ne(0).sum())
        if n_events < 10:
            continue
        train_sig, val_sig, test_sig = train_val_test_split(signal, train_frac=0.6, val_frac=0.2)

        for hold in HOLDING_PERIODS:
            val_bt = run_backtest(
                val_sig,
                btc_ret,
                holding_period=hold,
                commission_rt=bt_cfg["commission_rt"],
                slippage=bt_cfg["slippage"],
                stop_loss_vol_mult=bt_cfg["stop_loss_vol_mult"],
            )
            report = full_report(
                val_bt["net_ret"],
                gross_returns=val_bt["gross_ret"],
                benchmark_returns=btc_ret.reindex(val_bt.index).fillna(0),
                label=f"{defn}_H{hold}_VAL",
            )
            all_reports.append({
                "definition": defn,
                "holding_period": hold,
                "split": "val",
                "signal_events": n_events,
                **report.drop("label").to_dict(),
            })

        val_df = pd.DataFrame([row for row in all_reports if row["definition"] == defn and row["split"] == "val"])
        best_row = val_df.sort_values("sharpe", ascending=False, na_position="last").iloc[0]
        best_h = int(best_row["holding_period"])
        test_bt = run_backtest(
            test_sig,
            btc_ret,
            holding_period=best_h,
            commission_rt=bt_cfg["commission_rt"],
            slippage=bt_cfg["slippage"],
            stop_loss_vol_mult=bt_cfg["stop_loss_vol_mult"],
        )
        test_bt.to_parquet(proc_dir / f"primary_backtest_{defn}_H{best_h}_test.parquet")
        report = full_report(
            test_bt["net_ret"],
            gross_returns=test_bt["gross_ret"],
            benchmark_returns=btc_ret.reindex(test_bt.index).fillna(0),
            label=f"{defn}_H{best_h}_TEST",
        )
        row = {
            "definition": defn,
            "holding_period": best_h,
            "split": "test",
            "signal_events": n_events,
            **report.drop("label").to_dict(),
        }
        all_reports.append(row)
        test_reports.append(row)

    reports = pd.DataFrame(all_reports)
    tests = pd.DataFrame(test_reports)
    reports.to_csv(proc_dir / "primary_backtest_results.csv", index=False)
    tests.to_csv(proc_dir / "primary_backtest_test_results.csv", index=False)
    return reports, tests


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x * 100:.3f}%"


def markdown_table(df: pd.DataFrame, index: bool = False) -> str:
    if df.empty:
        return "_empty_"
    table = df.copy()
    if index:
        table = table.reset_index()
    table = table.fillna("n/a").astype(str)
    headers = list(table.columns)
    rows = table.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[i])) for row in rows))
        for i, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def write_report(
    report_path: Path,
    coverage: pd.DataFrame,
    signal_summary_df: pd.DataFrame,
    gc_df: pd.DataFrame,
    event_summary: pd.DataFrame,
    backtest_tests: pd.DataFrame,
    proc_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Focused Research Report: Prediction Markets as a BTC Signal")
    lines.append("")
    lines.append(f"Run date: {pd.Timestamp.now(tz='America/Los_Angeles').strftime('%Y-%m-%d %H:%M %Z')}")
    lines.append(f"Sample: {START_DATE} through {END_DATE} UTC, primary BTC Kalshi universe only.")
    lines.append("")
    lines.append("## Executive Conclusion")
    backtest_pass = False
    if not backtest_tests.empty and "sharpe" in backtest_tests:
        backtest_pass = bool((backtest_tests["sharpe"] >= 1.0).any())
    if gc_df.empty or not gc_df.get("significant_raw", pd.Series(dtype=bool)).any():
        lines.append("The focused run does not pass the lead-lag gate. No tested market/definition pair has raw Granger p < 0.05, so the research should stop before event-study/backtest claims.")
    elif event_summary.empty or not event_summary.get("passes_tradeability_gate", pd.Series(dtype=bool)).any():
        n_raw = int(gc_df["significant_raw"].sum())
        n_bonf = int(gc_df["significant_bonferroni"].sum())
        msg = f"The focused run passes the statistical lead-lag screen ({n_raw} raw-significant pairs, {n_bonf} Bonferroni-significant), but it does not clear the direction-adjusted event-study gate: no definition shows statistically significant peak net CAR above 0.3% after a 10 bp cost assumption."
        if backtest_pass:
            best = backtest_tests.sort_values("sharpe", ascending=False).iloc[0]
            msg += f" The best OOS backtest is {best['definition']} H={int(best['holding_period'])} with Sharpe={best['sharpe']:.3f}, but this is not enough to declare a tradeable edge because the event-study evidence points the other way."
        lines.append(msg)
    elif backtest_tests.empty or not (backtest_tests["sharpe"] >= 1.0).any():
        lines.append("The focused run has some event-study evidence, but no tested definition clears the out-of-sample backtest Sharpe >= 1.0 bar.")
    else:
        best = backtest_tests.sort_values("sharpe", ascending=False).iloc[0]
        lines.append(f"The focused run produces a candidate strategy: {best['definition']} with H={int(best['holding_period'])} min, OOS Sharpe={best['sharpe']:.3f}. Treat this as provisional until the full 53-market and robustness runs are repeated.")
    lines.append("")

    lines.append("## Data Coverage")
    lines.append("")
    lines.append(markdown_table(coverage, index=False))
    lines.append("")

    lines.append("## Signal Counts")
    lines.append("")
    pivot = signal_summary_df.pivot(index="market", columns="definition", values="n_total").fillna(0).astype(int)
    lines.append(markdown_table(pivot, index=True))
    lines.append("")

    lines.append("## Lead-Lag Gate")
    lines.append("")
    if gc_df.empty:
        lines.append("No Granger tests were run because all market/definition pairs had fewer than 30 events.")
    else:
        cols = ["market", "definition", "n_events", "min_p_raw", "best_lag", "min_p_bonferroni", "significant_raw", "significant_bonferroni"]
        lines.append(markdown_table(gc_df[cols].head(20), index=False))
    lines.append("")

    reg_path = proc_dir / "primary_best_lead_lag_regression.csv"
    if reg_path.exists():
        reg = pd.read_csv(reg_path)
        lines.append("## Best-Pair Lead-Lag Regression")
        lines.append("")
        lines.append(markdown_table(reg, index=False))
        lines.append("")

    cross_path = proc_dir / "primary_placebo_cross_asset.csv"
    if cross_path.exists():
        cross = pd.read_csv(cross_path)
        lines.append("## Cross-Asset Placebo")
        lines.append("")
        lines.append(markdown_table(cross, index=False))
        lines.append("")

    lines.append("## Event Study Gate")
    lines.append("")
    if event_summary.empty:
        lines.append("No event study results.")
    else:
        display = event_summary.copy()
        display["peak_direction_adjusted_car"] = display["peak_direction_adjusted_car"].map(_fmt_pct)
        display["net_peak_car_after_10bp"] = display["net_peak_car_after_10bp"].map(_fmt_pct)
        lines.append(markdown_table(display, index=False))
    lines.append("")

    lines.append("## Backtest Gate")
    lines.append("")
    if backtest_tests.empty:
        lines.append("No backtest results.")
    else:
        cols = ["definition", "holding_period", "sharpe", "sortino", "max_drawdown_pct", "win_rate_pct", "profit_factor", "n_trades", "signal_events"]
        lines.append(markdown_table(backtest_tests[cols].sort_values("sharpe", ascending=False), index=False))
    lines.append("")

    lines.append("## Remaining Work")
    lines.append("")
    lines.append("- Repeat the run on the full 53-market universe if disk/time permits.")
    lines.append("- Run a no-forward-fill sensitivity check and a 15-minute aggregation check.")
    lines.append("- Treat CPI/FED markets as regime filters only after the primary BTC signal is validated.")
    lines.append("- Use the historical DVOL files for conditioning once the spot signal passes the event-study and OOS backtest gates.")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("KALSHI_API_KEY")
    if not api_key:
        raise RuntimeError("KALSHI_API_KEY is not set in .env")

    with open(ROOT / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    raw_dir = ROOT / "data" / "raw"
    proc_dir = ROOT / "data" / "processed"
    report_dir = ROOT / "reports"

    print("Fetching primary Kalshi data")
    kalshi = fetch_primary_kalshi(api_key, raw_dir)
    if not kalshi:
        raise RuntimeError("No Kalshi data fetched")

    final_markets = pd.DataFrame({"ticker": list(kalshi), "tier": 1, "universe": "primary_btc"})
    (raw_dir / "kalshi").mkdir(parents=True, exist_ok=True)
    final_markets.to_parquet(raw_dir / "kalshi" / "final_market_list_primary.parquet", index=False)

    print("Fetching Binance crypto data")
    btc = fetch_binance_monthly("BTCUSDT", START_DATE, END_DATE, raw_dir / "crypto" / "crypto_1m_BTCUSDT.parquet")
    eth = fetch_binance_monthly("ETHUSDT", START_DATE, END_DATE, raw_dir / "crypto" / "crypto_1m_ETHUSDT.parquet")
    btc_ret = btc.set_index("timestamp_utc").sort_index()["log_ret"]
    eth_ret = eth.set_index("timestamp_utc").sort_index()["log_ret"]

    print("Fetching Deribit DVOL")
    for currency in ["BTC", "ETH"]:
        dvol = fetch_deribit_dvol(currency, start_ts=_ts(START_DATE), end_ts=_ts(END_DATE))
        save_deribit_dvol(currency, dvol)
        print(f"  {currency}: {len(dvol):,} DVOL rows")

    print("Building panels and signals")
    prob_df, volume_df = build_probability_panel(kalshi, proc_dir)
    signal_frames, signal_summary_df = compute_signals(prob_df, volume_df, cfg, proc_dir)
    coverage = pd.read_csv(proc_dir / "primary_market_coverage.csv")

    print("Running lead-lag tests")
    gc_df, _ = run_lead_lag(signal_frames, btc_ret, eth_ret, proc_dir)

    print("Running event studies")
    event_summary = run_event_studies(signal_frames, btc_ret, proc_dir)

    print("Running backtests")
    _, backtest_tests = run_backtests(signal_frames, btc_ret, cfg, proc_dir)

    print("Writing report")
    write_report(
        report_dir / "focused_research_report.md",
        coverage,
        signal_summary_df,
        gc_df,
        event_summary,
        backtest_tests,
        proc_dir,
    )
    print(f"Report written to {report_dir / 'focused_research_report.md'}")


if __name__ == "__main__":
    main()
