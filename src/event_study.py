"""
event_study.py — Cumulative Abnormal Return (CAR) analysis around jump events.

Aligns all events to a common T=0 grid, normalizes by pre-event volatility,
and aggregates mean CAR with confidence intervals.
"""

import numpy as np
import pandas as pd
from scipy import stats


def build_event_windows(
    signal: pd.Series,
    log_ret: pd.Series,
    pre: int = 60,
    post: int = 120,
) -> pd.DataFrame:
    """
    For each jump event in `signal`, extract a window of log returns
    from T=-pre to T=+post minutes and align to a common integer index.

    Returns a DataFrame with shape (n_events, pre+post+1).
    Each row is one event; columns are [-pre, ..., -1, 0, 1, ..., post].
    """
    event_times = signal[signal != 0].index
    windows = []
    directions = []

    for t in event_times:
        start = t - pd.Timedelta(minutes=pre)
        end = t + pd.Timedelta(minutes=post)

        window = log_ret.loc[start:end]
        if len(window) < pre + post:
            continue  # skip events near the edges of the data

        # Reindex to exact minute offsets from T=0
        expected_times = pd.date_range(start=start, periods=pre + post + 1, freq="1min")
        window = window.reindex(expected_times)
        window.index = range(-pre, post + 1)

        windows.append(window)
        directions.append(int(signal.loc[t]))

    if not windows:
        return pd.DataFrame(), pd.Series(dtype=int)

    return pd.DataFrame(windows), pd.Series(directions, name="direction")


def normalize_by_vol(
    event_windows: pd.DataFrame,
    pre: int = 60,
    vol_window: int = 30,
) -> pd.DataFrame:
    """
    Normalize each event's returns by the pre-event realized volatility.
    vol_window: number of pre-event bars to use for vol estimate.

    Returns the same shape DataFrame with normalized returns.
    Uses bars from T=-(vol_window) to T=-1 for the vol estimate.
    """
    pre_cols = list(range(-pre, 0))
    vol_cols = list(range(-vol_window, 0))

    available_vol_cols = [c for c in vol_cols if c in event_windows.columns]
    pre_vol = event_windows[available_vol_cols].std(axis=1).replace(0, np.nan)

    normalized = event_windows.div(pre_vol, axis=0)
    return normalized


def compute_car(event_windows: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Cumulative Abnormal Return for each event.
    Cumulates returns from T=0 forward (and T=0 backward for pre-event drift).

    Returns a DataFrame of the same shape with cumulative sums.
    """
    # Separate pre and post windows
    pre_cols = [c for c in event_windows.columns if c < 0]
    post_cols = [c for c in event_windows.columns if c >= 0]

    # Pre-event: cumulate backward from T=-1 to T=-pre (to see drift)
    pre = event_windows[sorted(pre_cols)].cumsum(axis=1)

    # Post-event: cumulate forward from T=0
    post = event_windows[sorted(post_cols)].cumsum(axis=1)

    return pd.concat([pre, post], axis=1)[sorted(event_windows.columns)]


def aggregate_car(
    car: pd.DataFrame,
    directions: pd.Series = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Aggregate CAR across all events: mean, CI, win rate per time step.

    directions: if provided, splits results by +1 (positive jumps) and -1 (negative jumps).
    Returns DataFrame with columns: t, mean_car, ci_lower, ci_upper, n_events, win_rate.
    """
    if directions is not None:
        results = []
        for dir_val in [1, -1]:
            mask = directions == dir_val
            subset = car[mask.values]
            label = "positive" if dir_val == 1 else "negative"
            agg = _aggregate_single(subset, alpha)
            agg["direction"] = label
            results.append(agg)
        return pd.concat(results).reset_index(drop=True)

    return _aggregate_single(car, alpha)


def _aggregate_single(car: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    n = len(car)
    if n == 0:
        return pd.DataFrame()

    mean_car = car.mean(axis=0)
    se = car.sem(axis=0)  # standard error of the mean
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    rows = []
    for t in sorted(car.columns):
        m = mean_car[t]
        s = se[t]
        win_rate = (car[t] > 0).mean() if n > 0 else np.nan
        t_stat = m / s if s > 0 else np.nan
        p_val = 2 * stats.t.sf(abs(t_stat), df=n - 1) if not np.isnan(t_stat) else np.nan

        rows.append({
            "t": t,
            "mean_car": m,
            "ci_lower": m - t_crit * s,
            "ci_upper": m + t_crit * s,
            "t_stat": t_stat,
            "p_value": p_val,
            "win_rate": win_rate,
            "n_events": n,
        })

    return pd.DataFrame(rows)


def full_event_study(
    signal: pd.Series,
    log_ret: pd.Series,
    pre: int = 60,
    post: int = 120,
    normalize: bool = True,
    vol_window: int = 30,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end event study.

    Returns:
        agg_car : aggregated CAR DataFrame (mean, CI, t-stat, by direction)
        raw_windows : raw (or normalized) event windows DataFrame

    Prints a warning if n_events < 30 (insufficient for inference).
    """
    windows, directions = build_event_windows(signal, log_ret, pre=pre, post=post)

    if windows.empty:
        print("WARNING: No events found for event study.")
        return pd.DataFrame(), pd.DataFrame()

    n_events = len(windows)
    if n_events < 30:
        print(f"WARNING: Only {n_events} events — results have wide confidence intervals.")

    if normalize:
        windows = normalize_by_vol(windows, pre=pre, vol_window=vol_window)

    car = compute_car(windows)
    agg = aggregate_car(car, directions=directions, alpha=alpha)

    return agg, windows


def subsample_analysis(
    signal: pd.Series,
    log_ret: pd.Series,
    prob_mid: pd.Series = None,
    btc_ret: pd.Series = None,
    pre: int = 60,
    post: int = 120,
) -> dict[str, pd.DataFrame]:
    """
    Run event study on subsamples: by jump magnitude, time of day, and market regime.

    Returns dict of {subsample_label: aggregated_car_df}.
    """
    windows, directions = build_event_windows(signal, log_ret, pre=pre, post=post)
    if windows.empty:
        return {}

    event_times = signal[signal != 0].index
    results = {}

    # 1. By time of day (US hours: 14:00–21:00 UTC vs off-hours)
    us_hours = event_times[event_times.hour.isin(range(14, 21))]
    off_hours = event_times[~event_times.isin(us_hours)]

    for label, subset_times in [("us_hours", us_hours), ("off_hours", off_hours)]:
        mask = [t in subset_times for t in event_times]
        subset_w = windows[mask]
        subset_d = directions[mask]
        if len(subset_w) >= 10:
            car = compute_car(normalize_by_vol(subset_w))
            results[label] = aggregate_car(car, directions=subset_d)

    # 2. By jump magnitude quintile (using T=0 column = first post-event bar)
    if prob_mid is not None:
        event_mags = []
        for t in event_times:
            if t in prob_mid.index:
                mag = abs(prob_mid.loc[t] - prob_mid.shift(1).loc[t])
                event_mags.append(mag)
            else:
                event_mags.append(np.nan)

        mags = pd.Series(event_mags)
        quintiles = pd.qcut(mags, 5, labels=False, duplicates="drop")

        for q in quintiles.dropna().unique():
            mask = (quintiles == q).values
            subset_w = windows[mask]
            subset_d = directions[mask]
            if len(subset_w) >= 10:
                car = compute_car(normalize_by_vol(subset_w))
                results[f"magnitude_q{int(q)+1}"] = aggregate_car(car, directions=subset_d)

    # 3. By crypto market regime (bull if BTC 90-day return > 0, bear otherwise)
    if btc_ret is not None:
        btc_trend = btc_ret.rolling(90 * 1440).sum()
        bull_times = event_times[btc_trend.reindex(event_times).fillna(0) > 0]
        bear_times = event_times[~event_times.isin(bull_times)]

        for label, subset_times in [("bull_regime", bull_times), ("bear_regime", bear_times)]:
            mask = [t in subset_times for t in event_times]
            subset_w = windows[mask]
            subset_d = directions[mask]
            if len(subset_w) >= 10:
                car = compute_car(normalize_by_vol(subset_w))
                results[label] = aggregate_car(car, directions=subset_d)

    return results
