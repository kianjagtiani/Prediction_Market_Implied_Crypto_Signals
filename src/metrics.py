"""
metrics.py — Performance metrics for strategy evaluation.

All metrics follow the global quant standards: annualized, risk-free rate adjusted,
includes Deflated Sharpe Ratio for multiple-testing correction.
"""

import numpy as np
import pandas as pd
from scipy import stats


def annualized_return(returns: pd.Series, periods_per_year: int = 525_600) -> float:
    """CAGR from a series of per-bar returns. Default: 1-minute bars (525,600/year)."""
    total = (1 + returns.fillna(0)).prod()
    n = len(returns)
    return float(total ** (periods_per_year / n) - 1)


def annualized_vol(returns: pd.Series, periods_per_year: int = 525_600) -> float:
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 525_600) -> float:
    """Annualized Sharpe ratio."""
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_vol(returns, periods_per_year)
    if ann_vol == 0:
        return np.nan
    return (ann_ret - rf) / ann_vol


def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 525_600) -> float:
    """Annualized Sortino ratio (uses downside deviation)."""
    ann_ret = annualized_return(returns, periods_per_year)
    downside = returns[returns < 0].std() * np.sqrt(periods_per_year)
    if downside == 0:
        return np.nan
    return (ann_ret - rf) / downside


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (as a positive fraction)."""
    cum = (1 + returns.fillna(0)).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    return float(-dd.min())


def max_drawdown_duration(returns: pd.Series) -> int:
    """Length of the longest drawdown period in bars."""
    cum = (1 + returns.fillna(0)).cumprod()
    rolling_max = cum.cummax()
    in_dd = cum < rolling_max
    # count consecutive True runs
    duration = 0
    max_dur = 0
    for v in in_dd:
        if v:
            duration += 1
            max_dur = max(max_dur, duration)
        else:
            duration = 0
    return max_dur


def calmar(returns: pd.Series, periods_per_year: int = 525_600) -> float:
    """Calmar ratio = CAGR / max drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return annualized_return(returns, periods_per_year) / mdd


def win_rate(returns: pd.Series) -> float:
    """Fraction of non-zero return bars with positive return."""
    active = returns[returns != 0]
    if len(active) == 0:
        return np.nan
    return float((active > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Gross profit / gross loss."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return np.inf
    return float(gains / losses)


def information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series,
                      periods_per_year: int = 525_600) -> float:
    """Annualized information ratio vs a benchmark."""
    active = strategy_returns - benchmark_returns
    ann_active = active.mean() * periods_per_year
    tracking_err = active.std() * np.sqrt(periods_per_year)
    if tracking_err == 0:
        return np.nan
    return ann_active / tracking_err


def deflated_sharpe_ratio(
    sharpe_star: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).
    Adjusts for the multiple-testing problem when the best Sharpe is selected
    from n_trials parameter combinations.

    sharpe_star : best observed Sharpe ratio (annualized, SR* in the paper)
    n_trials    : number of (strategy/parameter) combinations tested
    n_obs       : number of independent observations in the sample
    skewness    : skewness of strategy returns
    kurtosis    : excess kurtosis of strategy returns (3 = normal)

    Returns: probability that the true Sharpe > 0 after correcting for selection bias.
    A DSR > 0.95 is required for the strategy to be considered non-spurious.
    """
    # Expected maximum Sharpe under repeated IID testing
    euler_mascheroni = 0.5772156649
    expected_max_sr = (
        (1 - euler_mascheroni) * stats.norm.ppf(1 - 1.0 / n_trials)
        + euler_mascheroni * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )

    # Variance of Sharpe estimator
    sr_std = np.sqrt(
        (1 - skewness * sharpe_star + (kurtosis - 1) / 4 * sharpe_star ** 2) / (n_obs - 1)
    )

    dsr = stats.norm.cdf((sharpe_star - expected_max_sr) / sr_std)
    return float(dsr)


def full_report(
    net_returns: pd.Series,
    gross_returns: pd.Series = None,
    benchmark_returns: pd.Series = None,
    n_trials: int = 1,
    periods_per_year: int = 525_600,
    label: str = "",
) -> pd.Series:
    """
    Compute all required metrics and return as a named Series.
    Prints a formatted summary.
    """
    r = net_returns.dropna()
    ann_ret = annualized_return(r, periods_per_year)
    ann_vol_val = annualized_vol(r, periods_per_year)
    sh = sharpe(r, periods_per_year=periods_per_year)
    so = sortino(r, periods_per_year=periods_per_year)
    mdd = max_drawdown(r)
    cal = calmar(r, periods_per_year)
    wr = win_rate(r)
    pf = profit_factor(r)
    n_trades = int((r != 0).sum())

    metrics = {
        "label": label,
        "cagr": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol_val * 100, 2),
        "sharpe": round(sh, 3) if not np.isnan(sh) else np.nan,
        "sortino": round(so, 3) if not np.isnan(so) else np.nan,
        "max_drawdown_pct": round(mdd * 100, 2),
        "calmar": round(cal, 3) if not np.isnan(cal) else np.nan,
        "win_rate_pct": round(wr * 100, 1) if not np.isnan(wr) else np.nan,
        "profit_factor": round(pf, 2) if not np.isinf(pf) else np.inf,
        "n_trades": n_trades,
    }

    if gross_returns is not None:
        cost_drag = annualized_return(gross_returns.dropna(), periods_per_year) - ann_ret
        metrics["cost_drag_pct"] = round(cost_drag * 100, 2)

    if benchmark_returns is not None:
        ir = information_ratio(r, benchmark_returns.reindex(r.index).fillna(0), periods_per_year)
        metrics["information_ratio"] = round(ir, 3) if not np.isnan(ir) else np.nan

    if n_trials > 1:
        sk = float(r.skew())
        ku = float(r.kurtosis()) + 3  # scipy returns excess kurtosis
        dsr = deflated_sharpe_ratio(sh, n_trials, len(r), sk, ku)
        metrics["deflated_sharpe_ratio"] = round(dsr, 3)

    result = pd.Series(metrics)

    # Print summary
    print(f"\n{'─'*45}")
    print(f"  {label or 'Strategy'} Performance")
    print(f"{'─'*45}")
    for k, v in metrics.items():
        if k == "label":
            continue
        print(f"  {k:<30} {v}")
    print(f"{'─'*45}")

    return result


def passes_minimum_bars(report: pd.Series) -> bool:
    """
    Check if a strategy passes all minimum performance bars from the plan.
    Returns True if all bars are met.
    """
    checks = {
        "sharpe >= 1.0": report.get("sharpe", 0) >= 1.0,
        "max_drawdown < 20%": report.get("max_drawdown_pct", 100) < 20.0,
        "n_trades >= 30": report.get("n_trades", 0) >= 30,
    }
    passed = all(checks.values())
    for check, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {check}")
    return passed
