"""
signal.py — Four jump detection definitions for prediction market probability series.

All functions take a prob_mid Series (0–1 float, minute-indexed UTC) and return
a signals DataFrame with columns: prob_mid, delta_p, signal (+1/-1/0), jump_magnitude_pct.

signal = +1 : probability jumped UP   (bullish crypto signal)
signal = -1 : probability jumped DOWN (bearish crypto signal)
signal =  0 : no jump
"""

import numpy as np
import pandas as pd


def apply_cooldown(signal: pd.Series, cooldown: int = 60) -> pd.Series:
    """
    After a non-zero signal fires, suppress all signals for `cooldown` bars.
    Ensures statistical independence between detected events.
    """
    result = signal.copy().astype(float)
    last_fire = -np.inf

    for i, val in enumerate(signal):
        if (i - last_fire) < cooldown:
            result.iloc[i] = 0
        elif val != 0:
            last_fire = i

    return result


def jump_D1(
    prob_mid: pd.Series,
    W: int = 30,
    threshold: float = 2.0,
    min_abs: float = 0.02,
    cooldown: int = 60,
) -> pd.DataFrame:
    """
    D1 — Rolling Z-Score (Adaptive Threshold)

    Detects bars where the probability change is `threshold` standard deviations
    above/below the rolling mean of changes over the past W bars.
    Filters out bars with |delta_p| < min_abs (microstructure noise).

    Best for: markets with varying activity levels — adapts to each market's own regime.
    """
    delta_p = prob_mid.diff()
    mu = delta_p.rolling(W, min_periods=W).mean()
    sigma = delta_p.rolling(W, min_periods=W).std()
    z = (delta_p - mu) / sigma.replace(0, np.nan)

    raw_signal = pd.Series(0.0, index=prob_mid.index)
    raw_signal[z > threshold] = 1
    raw_signal[z < -threshold] = -1
    raw_signal[delta_p.abs() < min_abs] = 0

    signal = apply_cooldown(raw_signal, cooldown)

    return pd.DataFrame({
        "prob_mid": prob_mid,
        "delta_p": delta_p,
        "z_score": z,
        "signal": signal,
        "jump_magnitude_pct": delta_p.abs(),
    })


def jump_D2(
    prob_mid: pd.Series,
    abs_threshold: float = 0.05,
    cooldown: int = 60,
) -> pd.DataFrame:
    """
    D2 — Absolute Probability Threshold

    Fires when probability moves >= abs_threshold percentage points in a single bar.
    Simple and interpretable. A 5pp move in 1 minute is unambiguously a real move.

    Best for: high-liquidity markets where even small moves are informative.
    Note: may yield very few signals on illiquid/slow-moving markets.
    """
    delta_p = prob_mid.diff()

    raw_signal = pd.Series(0.0, index=prob_mid.index)
    raw_signal[delta_p >= abs_threshold] = 1
    raw_signal[delta_p <= -abs_threshold] = -1

    signal = apply_cooldown(raw_signal, cooldown)

    return pd.DataFrame({
        "prob_mid": prob_mid,
        "delta_p": delta_p,
        "signal": signal,
        "jump_magnitude_pct": delta_p.abs(),
    })


def jump_D3(
    prob_mid: pd.Series,
    rel_threshold: float = 0.20,
    min_abs: float = 0.01,
    cooldown: int = 60,
) -> pd.DataFrame:
    """
    D3 — Relative (Percentage) Change

    Fires when probability changes >= rel_threshold relative to current level.
    e.g., 40% -> 48% is a 20% relative change.

    Best for: markets at extreme probabilities (near 0 or 1) where absolute moves
    are small but relative information content is high.
    min_abs prevents noise on very-low-probability markets.
    """
    delta_p = prob_mid.diff()
    prev_prob = prob_mid.shift(1).replace(0, np.nan)
    rel_change = delta_p / prev_prob

    raw_signal = pd.Series(0.0, index=prob_mid.index)
    raw_signal[rel_change >= rel_threshold] = 1
    raw_signal[rel_change <= -rel_threshold] = -1
    raw_signal[delta_p.abs() < min_abs] = 0

    signal = apply_cooldown(raw_signal, cooldown)

    return pd.DataFrame({
        "prob_mid": prob_mid,
        "delta_p": delta_p,
        "rel_change": rel_change,
        "signal": signal,
        "jump_magnitude_pct": delta_p.abs(),
    })


def jump_D4(
    prob_mid: pd.Series,
    volume: pd.Series,
    W: int = 30,
    z_threshold: float = 2.0,
    min_abs: float = 0.02,
    vol_ratio_min: float = 1.5,
    cooldown: int = 60,
) -> pd.DataFrame:
    """
    D4 — Z-Score + Volume Confirmation (High-Conviction Filter)

    Same z-score logic as D1, but requires volume at the jump bar to be
    >= vol_ratio_min times the rolling mean volume. Only fires when both
    probability AND volume confirm the move — fewest signals, highest precision.

    Best for: separating genuine information events from thin-market noise.
    """
    delta_p = prob_mid.diff()
    mu = delta_p.rolling(W, min_periods=W).mean()
    sigma = delta_p.rolling(W, min_periods=W).std()
    z = (delta_p - mu) / sigma.replace(0, np.nan)

    # Align volume to prob_mid's index and coerce to numeric
    volume = pd.to_numeric(volume, errors='coerce').reindex(prob_mid.index)
    vol_mean = volume.rolling(W, min_periods=W).mean().replace(0, np.nan)
    vol_ratio = volume / vol_mean

    raw_signal = pd.Series(0.0, index=prob_mid.index)
    raw_signal[(z > z_threshold) & (vol_ratio >= vol_ratio_min)] = 1
    raw_signal[(z < -z_threshold) & (vol_ratio >= vol_ratio_min)] = -1
    raw_signal[delta_p.abs() < min_abs] = 0

    signal = apply_cooldown(raw_signal, cooldown)

    return pd.DataFrame({
        "prob_mid": prob_mid,
        "delta_p": delta_p,
        "z_score": z,
        "vol_ratio": vol_ratio,
        "signal": signal,
        "jump_magnitude_pct": delta_p.abs(),
    })


def compute_all_signals(
    prob_mid: pd.Series,
    volume: pd.Series = None,
    cfg: dict = None,
) -> dict[str, pd.DataFrame]:
    """
    Convenience wrapper: compute all four definitions at once.

    Returns dict with keys 'D1', 'D2', 'D3', 'D4'.
    If volume is None, D4 is skipped.
    cfg: dict loaded from config.yaml (optional, uses defaults if None).
    """
    if cfg is None:
        cfg = {}
    s = cfg.get("signal", {})
    cooldown = s.get("cooldown_minutes", 60)

    results = {
        "D1": jump_D1(
            prob_mid,
            W=s.get("d1_lookback_window", 30),
            threshold=s.get("d1_z_threshold", 2.0),
            min_abs=s.get("d1_min_abs_change", 0.02),
            cooldown=cooldown,
        ),
        "D2": jump_D2(
            prob_mid,
            abs_threshold=s.get("d2_abs_threshold", 0.05),
            cooldown=cooldown,
        ),
        "D3": jump_D3(
            prob_mid,
            rel_threshold=s.get("d3_rel_threshold", 0.20),
            cooldown=cooldown,
        ),
    }

    if volume is not None:
        results["D4"] = jump_D4(
            prob_mid,
            volume,
            W=s.get("d4_lookback_window", 30),
            z_threshold=s.get("d4_z_threshold", 2.0),
            min_abs=s.get("d1_min_abs_change", 0.02),
            vol_ratio_min=s.get("d4_volume_ratio_min", 1.5),
            cooldown=cooldown,
        )

    return results


def signal_summary(signals_dict: dict[str, pd.DataFrame], market_id: str = "") -> pd.DataFrame:
    """
    Produce a summary table for each definition showing signal counts.
    Flags definitions with < 30 signals (insufficient for inference).
    """
    rows = []
    for defn, df in signals_dict.items():
        s = df["signal"]
        n_pos = (s == 1).sum()
        n_neg = (s == -1).sum()
        n_total = n_pos + n_neg
        avg_mag = df.loc[s != 0, "jump_magnitude_pct"].mean() if n_total > 0 else np.nan

        rows.append({
            "market": market_id,
            "definition": defn,
            "n_positive": int(n_pos),
            "n_negative": int(n_neg),
            "n_total": int(n_total),
            "avg_magnitude_pct": round(avg_mag * 100, 2) if not np.isnan(avg_mag) else np.nan,
            "sufficient_for_inference": n_total >= 30,
        })

    return pd.DataFrame(rows)
