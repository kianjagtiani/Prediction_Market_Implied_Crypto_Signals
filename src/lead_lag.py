"""
lead_lag.py — Lead-lag analysis between prediction market jumps and crypto returns.

Three methods:
1. Cross-correlation function (CCF) with confidence bands
2. Granger causality tests (statsmodels)
3. Lead-lag regression (OLS, per forward horizon)

Plus three mandatory placebo tests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def compute_ccf(
    delta_p: pd.Series,
    log_ret: pd.Series,
    lags: list[int] = None,
) -> pd.DataFrame:
    """
    Cross-correlation function between prediction market probability changes
    and crypto log returns at each lag.

    Positive lag k means: corr(delta_p at t, log_ret at t+k)
    → If peak is at k > 0, prediction market leads crypto.

    Returns DataFrame with columns: lag, correlation, ci_lower, ci_upper
    """
    if lags is None:
        lags = [-60, -30, -10, -5, -1, 0, 1, 5, 10, 30, 60]

    # Align on common index
    aligned = pd.concat([delta_p.rename("dp"), log_ret.rename("ret")], axis=1).dropna()
    n = len(aligned)
    ci = 1.96 / np.sqrt(n)

    rows = []
    for lag in lags:
        # lag > 0: shift log_ret backward (future ret)
        # lag < 0: shift log_ret forward (past ret)
        corr = aligned["dp"].corr(aligned["ret"].shift(-lag))
        rows.append({"lag_min": lag, "correlation": corr, "ci_lower": -ci, "ci_upper": ci})

    return pd.DataFrame(rows)


def run_granger(
    delta_p: pd.Series,
    log_ret: pd.Series,
    maxlag: int = 30,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Granger causality test: does delta_p help predict log_ret beyond log_ret's own lags?

    Tests H0: delta_p does NOT Granger-cause log_ret.
    Reports F-test p-values at each lag order.

    Returns DataFrame with columns: lag, p_value_f, p_value_chi2
    """
    aligned = pd.concat([delta_p.rename("dp"), log_ret.rename("ret")], axis=1).dropna()

    # grangercausalitytests expects [dependent, cause] ordering
    data = aligned[["ret", "dp"]].values

    gc_results = grangercausalitytests(data, maxlag=maxlag, verbose=verbose)

    rows = []
    for lag, result in gc_results.items():
        rows.append({
            "lag": lag,
            "p_value_f": result[0]["ssr_ftest"][1],
            "p_value_chi2": result[0]["ssr_chi2test"][1],
        })

    return pd.DataFrame(rows)


def bonferroni_correct(p_values: list[float], n_tests: int = None) -> list[float]:
    """Apply Bonferroni correction to a list of p-values."""
    if n_tests is None:
        n_tests = len(p_values)
    corrected = [min(p * n_tests, 1.0) for p in p_values]
    return corrected


def lead_lag_regression(
    signal: pd.Series,
    log_ret: pd.Series,
    horizons: list[int] = None,
) -> pd.DataFrame:
    """
    Conditional regression: for each forward horizon h,
        log_ret(t → t+h) = alpha + beta * signal(t) + gamma * log_ret(t-60 → t) + eps

    Reports beta, t-stat, p-value, R2 for each horizon.
    A statistically significant beta decaying toward zero at longer horizons
    is evidence of a real short-horizon lead-lag.

    signal  : +1/-1/0 jump signal Series
    log_ret : 1-minute log returns Series
    horizons: list of forward horizons in minutes
    """
    if horizons is None:
        horizons = [1, 10, 30, 60, 240, 1440]

    idx = log_ret.index
    sig = signal.reindex(idx).fillna(0)

    # Pre-event control: past 60-minute return
    past_ret = log_ret.rolling(60).sum().shift(1)

    rows = []
    for h in horizons:
        # Forward return over h bars
        fwd = log_ret.rolling(h).sum().shift(-h + 1)

        df = pd.DataFrame({
            "fwd_ret": fwd,
            "signal": sig,
            "past_ret": past_ret,
        }).dropna()

        # Only use bars where a signal fired
        df_events = df[df["signal"] != 0]
        if len(df_events) < 10:
            rows.append({"horizon_min": h, "beta": np.nan, "t_stat": np.nan,
                         "p_value": np.nan, "r2": np.nan, "n_obs": len(df_events)})
            continue

        X = sm.add_constant(df_events[["signal", "past_ret"]])
        model = sm.OLS(df_events["fwd_ret"], X).fit()

        rows.append({
            "horizon_min": h,
            "beta": model.params.get("signal", np.nan),
            "t_stat": model.tvalues.get("signal", np.nan),
            "p_value": model.pvalues.get("signal", np.nan),
            "r2": model.rsquared,
            "n_obs": len(df_events),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Placebo Tests
# ─────────────────────────────────────────────

def placebo_time_shuffle(
    signal: pd.Series,
    log_ret: pd.Series,
    n_simulations: int = 200,
    lags: list[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Placebo 1 — Time Shuffle:
    Randomly shuffle the timestamps of jump events and recompute the CCF.
    Real signal CCF should be significantly stronger than the shuffle distribution.

    Returns DataFrame with lag and mean/std of shuffled correlations.
    """
    if lags is None:
        lags = [-60, -30, -10, -5, -1, 0, 1, 5, 10, 30, 60]

    rng = np.random.default_rng(seed)
    event_vals = signal[signal != 0].values
    all_idx = signal.index
    sim_results = []

    for _ in range(n_simulations):
        # Randomly assign event timestamps
        shuffled_idx = rng.choice(all_idx, size=len(event_vals), replace=False)
        shuffled = pd.Series(0.0, index=all_idx)
        shuffled.loc[shuffled_idx] = event_vals[rng.permutation(len(event_vals))]

        ccf = compute_ccf(shuffled, log_ret, lags=lags)
        sim_results.append(ccf.set_index("lag_min")["correlation"])

    sim_df = pd.concat(sim_results, axis=1)
    summary = pd.DataFrame({
        "lag_min": lags,
        "placebo_mean_corr": sim_df.mean(axis=1).values,
        "placebo_std_corr": sim_df.std(axis=1).values,
    })
    return summary


def placebo_random_jumps(
    signal: pd.Series,
    log_ret: pd.Series,
    n_simulations: int = 200,
    lags: list[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Placebo 2 — Random Jump Times:
    Generate synthetic jumps at random times with the same frequency as real jumps.
    This produces the null distribution for the CCF under the assumption of no signal.

    Returns DataFrame with lag and 5th/95th percentile of null correlations.
    """
    if lags is None:
        lags = [-60, -30, -10, -5, -1, 0, 1, 5, 10, 30, 60]

    rng = np.random.default_rng(seed)
    n_events = int((signal != 0).sum())
    all_idx = signal.index
    sim_corrs = {lag: [] for lag in lags}

    for _ in range(n_simulations):
        rand_idx = rng.choice(len(all_idx), size=n_events, replace=False)
        synth = pd.Series(0.0, index=all_idx)
        synth.iloc[rand_idx] = rng.choice([-1.0, 1.0], size=n_events)

        ccf = compute_ccf(synth, log_ret, lags=lags)
        for _, row in ccf.iterrows():
            sim_corrs[row["lag_min"]].append(row["correlation"])

    rows = []
    for lag in lags:
        c = sim_corrs[lag]
        rows.append({
            "lag_min": lag,
            "null_p5": np.percentile(c, 5),
            "null_p95": np.percentile(c, 95),
            "null_mean": np.mean(c),
        })
    return pd.DataFrame(rows)


def placebo_cross_asset(
    signal: pd.Series,
    log_ret_target: pd.Series,
    log_ret_other: pd.Series,
    lags: list[int] = None,
) -> pd.DataFrame:
    """
    Placebo 3 — Cross-Asset Specificity:
    A BTC ETF prediction market jump should predict BTC returns MORE than ETH returns.
    Returns CCF for both assets side by side.

    signal        : jump signal from the prediction market
    log_ret_target: log returns of the 'correct' asset (e.g. BTC for BTC reserve market)
    log_ret_other : log returns of a 'wrong' asset (e.g. ETH for BTC reserve market)
    """
    event_signal = signal.reindex(log_ret_target.index).fillna(0)
    ccf_target = compute_ccf(event_signal, log_ret_target, lags=lags).rename(
        columns={"correlation": "corr_target"}
    )
    ccf_other = compute_ccf(event_signal, log_ret_other, lags=lags).rename(
        columns={"correlation": "corr_other"}
    )
    result = ccf_target.merge(ccf_other[["lag_min", "corr_other"]], on="lag_min")
    result["corr_diff"] = result["corr_target"] - result["corr_other"]
    return result
