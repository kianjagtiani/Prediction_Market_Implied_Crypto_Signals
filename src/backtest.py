"""
backtest.py — Vectorized backtest engine for prediction market jump signals.

Strategy S1: Spot momentum — go long/short on crypto at T+1 after a jump fires,
hold for H bars, exit on stop-loss.

All operations are vectorized (no row loops). Look-ahead bias is prevented via shift(1).
"""

import numpy as np
import pandas as pd
from typing import Optional


def run_backtest(
    signals: pd.Series,
    crypto_ret: pd.Series,
    holding_period: int = 30,
    commission_rt: float = 0.00075,
    slippage: float = 0.00025,
    stop_loss_vol_mult: float = 1.5,
    pre_event_vol_window: int = 30,
) -> pd.DataFrame:
    """
    Vectorized backtest for Strategy S1 (spot momentum).

    signals       : Series of +1/-1/0 from any jump definition, indexed by timestamp_utc
    crypto_ret    : Series of 1-minute log returns for the crypto asset, same index
    holding_period: Number of bars to hold the position
    commission_rt : Round-trip commission rate (fraction)
    slippage      : One-way slippage estimate (fraction)
    stop_loss_vol_mult: Exit early if loss exceeds this multiple of pre-event vol

    Returns DataFrame with columns:
        timestamp_utc, signal, position, gross_ret, cost, net_ret, cumulative_net
    """
    # Align to common index
    idx = crypto_ret.index
    sig = signals.reindex(idx).fillna(0)
    ret = crypto_ret.reindex(idx).fillna(0)

    # shift(1): trade on bar T+1 open after signal fires at T
    # This is the critical look-ahead bias prevention
    positions = sig.shift(1).fillna(0)

    # Forward return over holding period (sum of next H log returns)
    # shift(-holding_period) means: at time T, we know the return over [T, T+H]
    # but we only USE this after shifting positions by 1, so no look-ahead
    fwd_ret = ret.rolling(holding_period).sum().shift(-holding_period + 1)

    # Stop-loss: compute pre-event realized vol (rolling std of past `window` bars)
    pre_vol = ret.rolling(pre_event_vol_window).std()
    stop_threshold = stop_loss_vol_mult * pre_vol

    # Apply stop-loss: zero out the position if expected loss > stop threshold
    # (simplified: cap the forward return at -stop_threshold)
    gross_ret = positions * fwd_ret
    gross_ret = gross_ret.clip(lower=-stop_threshold)

    # Cost: pay commission + slippage whenever position changes (entry or exit)
    position_change = positions.diff().abs().fillna(0)
    cost = position_change * (commission_rt + slippage)

    net_ret = gross_ret - cost

    # Cumulative net return
    cum_net = (1 + net_ret.fillna(0)).cumprod()

    result = pd.DataFrame({
        "signal": sig,
        "position": positions,
        "gross_ret": gross_ret,
        "cost": cost,
        "net_ret": net_ret,
        "cumulative_net": cum_net,
    }, index=idx)

    return result


def run_all_holding_periods(
    signals: pd.Series,
    crypto_ret: pd.Series,
    holding_periods: list[int],
    **kwargs,
) -> dict[int, pd.DataFrame]:
    """Run the backtest for each holding period in the list."""
    return {H: run_backtest(signals, crypto_ret, holding_period=H, **kwargs)
            for H in holding_periods}


def train_val_test_split(
    signals: pd.Series,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Split events into train / validation / test by chronological order.
    Only non-zero signal timestamps are split (not all bars).

    Returns three Series (train, val, test) — each is a subset of signals
    with 0s filled in for bars outside the respective window.
    """
    event_idx = signals[signals != 0].index
    n = len(event_idx)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_end = event_idx[n_train - 1] if n_train > 0 else signals.index[0]
    val_end = event_idx[n_train + n_val - 1] if (n_train + n_val) <= n else signals.index[-1]

    train_sig = signals.copy()
    train_sig[signals.index > train_end] = 0

    val_sig = signals.copy()
    val_sig[signals.index <= train_end] = 0
    val_sig[signals.index > val_end] = 0

    test_sig = signals.copy()
    test_sig[signals.index <= val_end] = 0

    return train_sig, val_sig, test_sig


def walk_forward(
    signals: pd.Series,
    crypto_ret: pd.Series,
    holding_period: int = 30,
    n_windows: int = 5,
    train_months: int = 6,
    test_months: int = 2,
    gap_days: int = 7,
    **kwargs,
) -> list[pd.DataFrame]:
    """
    Walk-forward validation: 5 non-overlapping windows of train/test.
    Returns a list of backtest DataFrames (one per test window).
    """
    results = []
    start = crypto_ret.index.min()

    train_td = pd.DateOffset(months=train_months)
    test_td = pd.DateOffset(months=test_months)
    gap_td = pd.DateOffset(days=gap_days)

    for w in range(n_windows):
        train_start = start + w * (train_td + test_td + gap_td)
        train_end = train_start + train_td
        test_start = train_end + gap_td
        test_end = test_start + test_td

        test_ret = crypto_ret[test_start:test_end]
        test_sig = signals.reindex(test_ret.index).fillna(0)

        if test_ret.empty or (test_sig != 0).sum() < 5:
            continue

        bt = run_backtest(test_sig, test_ret, holding_period=holding_period, **kwargs)
        bt["window"] = w
        results.append(bt)
        print(f"  Walk-forward window {w+1}: {test_start.date()} → {test_end.date()}, "
              f"signals={int((test_sig != 0).sum())}")

    return results


def build_comparison_table(
    results: dict,
    metric_fn,
) -> pd.DataFrame:
    """
    Build a comparison table from a dict of {label: backtest_df}.
    metric_fn: callable that takes a net_ret Series and returns a pd.Series of metrics.

    Example:
        results = {'D1_H30': bt_d1, 'D2_H30': bt_d2, ...}
        table = build_comparison_table(results, lambda r: full_report(r, label=...))
    """
    rows = []
    for label, bt in results.items():
        m = metric_fn(bt["net_ret"].dropna(), label=label)
        rows.append(m)
    return pd.DataFrame(rows).set_index("label")
