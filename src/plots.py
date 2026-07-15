"""
plots.py — All visualization functions for the prediction market signal project.

Every function saves to a file if `save_path` is provided, and returns the figure.
Uses matplotlib/seaborn throughout.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# Consistent style across all plots
plt.rcParams.update({
    "figure.figsize": (12, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


# ─────────────────────────────────────────────
# Prediction Market Plots
# ─────────────────────────────────────────────

def plot_probability_series(
    prob_mid: pd.Series,
    volume: pd.Series = None,
    signals: pd.Series = None,
    title: str = "Prediction Market Probability",
    event_dates: list = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot probability time series with optional volume overlay and detected jump markers.
    """
    fig, axes = plt.subplots(2 if volume is not None else 1, 1,
                              figsize=(14, 6 if volume is not None else 4),
                              sharex=True)
    ax1 = axes[0] if volume is not None else axes

    ax1.plot(prob_mid.index, prob_mid.values, lw=1, color="steelblue", label="prob_mid")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(0, 1)
    ax1.set_title(title)

    # Mark detected jumps
    if signals is not None:
        up = signals[signals == 1].index
        dn = signals[signals == -1].index
        ax1.scatter(up, prob_mid.reindex(up), marker="^", color="green", s=60,
                    zorder=5, label="Jump +1")
        ax1.scatter(dn, prob_mid.reindex(dn), marker="v", color="red", s=60,
                    zorder=5, label="Jump -1")

    # Annotate known event dates
    if event_dates:
        for d in event_dates:
            ax1.axvline(pd.Timestamp(d), color="orange", lw=1, linestyle="--", alpha=0.7)

    ax1.legend(loc="upper left", fontsize=9)

    if volume is not None:
        axes[1].bar(volume.index, volume.values, width=0.0007, color="gray", alpha=0.5)
        axes[1].set_ylabel("Volume")

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Lead-Lag Plots
# ─────────────────────────────────────────────

def plot_ccf(
    ccf_df: pd.DataFrame,
    placebo_df: pd.DataFrame = None,
    title: str = "Cross-Correlation: Prediction Market vs Crypto",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot the CCF with 95% confidence bands and optional placebo null distribution.

    ccf_df    : output of lead_lag.compute_ccf()
    placebo_df: output of lead_lag.placebo_random_jumps() or placebo_time_shuffle()
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(ccf_df["lag_min"], ccf_df["correlation"], color="steelblue",
           alpha=0.7, label="Observed CCF")
    ax.axhline(ccf_df["ci_upper"].iloc[0], color="red", lw=1, linestyle="--",
               label="95% CI")
    ax.axhline(ccf_df["ci_lower"].iloc[0], color="red", lw=1, linestyle="--")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8, linestyle=":")

    if placebo_df is not None and {"null_p5", "null_p95"}.issubset(placebo_df.columns):
        ax.fill_between(
            placebo_df["lag_min"],
            placebo_df["null_p5"],
            placebo_df["null_p95"],
            color="orange", alpha=0.25, label="Null 5-95th pct",
        )
    elif placebo_df is not None and {"placebo_mean_corr", "placebo_std_corr"}.issubset(placebo_df.columns):
        lo = placebo_df["placebo_mean_corr"] - 1.96 * placebo_df["placebo_std_corr"]
        hi = placebo_df["placebo_mean_corr"] + 1.96 * placebo_df["placebo_std_corr"]
        ax.fill_between(
            placebo_df["lag_min"],
            lo,
            hi,
            color="orange", alpha=0.25, label="Time-shuffle null ±1.96σ",
        )
        ax.plot(
            placebo_df["lag_min"],
            placebo_df["placebo_mean_corr"],
            color="orange", lw=1, linestyle="--", label="Time-shuffle mean",
        )

    ax.set_xlabel("Lag (minutes; positive = prediction market leads)")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_lead_lag_regression(
    reg_df: pd.DataFrame,
    title: str = "Lead-Lag Regression: Beta by Horizon",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot beta coefficient and 95% CI from lead_lag.lead_lag_regression().
    Shows how the signal strength decays across forward horizons.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    valid = reg_df.dropna(subset=["beta"])
    se = valid["beta"] / valid["t_stat"].replace(0, np.nan)
    ci = 1.96 * se

    ax.errorbar(valid["horizon_min"], valid["beta"], yerr=ci.values,
                fmt="o-", capsize=5, color="steelblue", label="beta ± 95% CI")
    ax.axhline(0, color="black", lw=0.8)

    sig_mask = valid["p_value"] < 0.05
    ax.scatter(valid.loc[sig_mask, "horizon_min"], valid.loc[sig_mask, "beta"],
               color="green", s=100, zorder=5, label="p < 0.05")

    ax.set_xlabel("Forward Horizon (minutes)")
    ax.set_ylabel("Beta (signal coefficient)")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Event Study Plots
# ─────────────────────────────────────────────

def plot_car(
    agg_car: pd.DataFrame,
    title: str = "Cumulative Abnormal Return Around Jump Events",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot mean CAR with 95% CI from T=-60 to T=+120, by signal direction.
    agg_car: output of event_study.aggregate_car()
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"positive": "green", "negative": "red"}

    if "direction" in agg_car.columns:
        for dir_label, color in colors.items():
            sub = agg_car[agg_car["direction"] == dir_label]
            if sub.empty:
                continue
            ax.plot(sub["t"], sub["mean_car"], color=color, lw=2, label=f"{dir_label} jump")
            ax.fill_between(sub["t"], sub["ci_lower"], sub["ci_upper"],
                            color=color, alpha=0.15)
    else:
        ax.plot(agg_car["t"], agg_car["mean_car"], lw=2, color="steelblue",
                label="Mean CAR")
        ax.fill_between(agg_car["t"], agg_car["ci_lower"], agg_car["ci_upper"],
                        color="steelblue", alpha=0.15)

    ax.axvline(0, color="black", lw=1, linestyle="--", label="T=0 (jump)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Minutes relative to jump event")
    ax.set_ylabel("Cumulative Abnormal Return")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Backtest Plots
# ─────────────────────────────────────────────

def plot_equity_curve(
    net_ret: pd.Series,
    benchmark_ret: pd.Series = None,
    title: str = "Equity Curve",
    save_path: str = None,
) -> plt.Figure:
    """
    Plot cumulative log returns (strategy vs benchmark) on log scale for long periods.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    cum_strat = (1 + net_ret.fillna(0)).cumprod()
    ax.plot(cum_strat.index, cum_strat.values, lw=1.5, label="Strategy", color="steelblue")

    if benchmark_ret is not None:
        cum_bench = (1 + benchmark_ret.reindex(net_ret.index).fillna(0)).cumprod()
        ax.plot(cum_bench.index, cum_bench.values, lw=1, label="BTC Buy & Hold",
                color="orange", linestyle="--")

    ax.set_yscale("log")
    ax.set_ylabel("Cumulative Return (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_drawdown(
    net_ret: pd.Series,
    title: str = "Drawdown",
    save_path: str = None,
) -> plt.Figure:
    """Plot rolling drawdown over time with max annotated."""
    cum = (1 + net_ret.fillna(0)).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.4)
    ax.plot(dd.index, dd.values, color="red", lw=0.8)

    min_dd = dd.min()
    min_idx = dd.idxmin()
    ax.annotate(f"Max DD: {min_dd:.1%}", xy=(min_idx, min_dd),
                xytext=(min_idx, min_dd * 0.7),
                arrowprops={"arrowstyle": "->", "color": "black"}, fontsize=9)

    ax.set_ylabel("Drawdown")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_rolling_sharpe(
    net_ret: pd.Series,
    window_bars: int = 6 * 30 * 1440,
    title: str = "Rolling 6-Month Sharpe",
    save_path: str = None,
) -> plt.Figure:
    """Plot rolling Sharpe ratio over time to detect regime changes."""
    ann_factor = np.sqrt(525_600)
    rolling_sh = (
        net_ret.rolling(window_bars).mean() /
        net_ret.rolling(window_bars).std()
    ) * ann_factor

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rolling_sh.index, rolling_sh.values, lw=1, color="steelblue")
    ax.axhline(1.0, color="green", lw=1, linestyle="--", label="Sharpe = 1.0")
    ax.axhline(0, color="gray", lw=0.5)

    ax.set_ylabel("Rolling Sharpe")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_monthly_returns_heatmap(
    net_ret: pd.Series,
    title: str = "Monthly Returns Heatmap",
    save_path: str = None,
) -> plt.Figure:
    """Calendar heatmap of monthly returns."""
    monthly = (1 + net_ret.fillna(0)).resample("ME").prod() - 1
    monthly.index = monthly.index.to_period("M")

    df = monthly.rename("ret").reset_index()
    df["year"] = df["index"].dt.year
    df["month"] = df["index"].dt.month

    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 0.6)))
    sns.heatmap(
        pivot * 100,
        annot=True, fmt=".1f", center=0,
        cmap="RdYlGn", linewidths=0.5,
        cbar_kws={"label": "Monthly Return (%)"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Year")

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_gross_vs_net(
    gross_ret: pd.Series,
    net_ret: pd.Series,
    title: str = "Gross vs Net Return (Cost Drag)",
    save_path: str = None,
) -> plt.Figure:
    """Show cumulative gross and net returns to visualize cost drag."""
    cum_gross = (1 + gross_ret.fillna(0)).cumprod()
    cum_net = (1 + net_ret.fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cum_gross.index, cum_gross.values, lw=1.5, label="Gross", color="steelblue")
    ax.plot(cum_net.index, cum_net.values, lw=1.5, label="Net (after costs)", color="orange")
    ax.fill_between(cum_gross.index, cum_gross.values, cum_net.values,
                    alpha=0.2, color="red", label="Cost drag")

    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ─────────────────────────────────────────────
# Comparison Plots (D1 vs D2 vs D3 vs D4)
# ─────────────────────────────────────────────

def plot_definition_comparison(
    comparison_table: pd.DataFrame,
    metric: str = "sharpe",
    title: str = "Jump Definition Comparison",
    save_path: str = None,
) -> plt.Figure:
    """
    Bar chart comparing a metric (e.g. Sharpe) across all four jump definitions.
    comparison_table: output of backtest.build_comparison_table()
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    vals = comparison_table[metric].sort_values(ascending=False)
    colors = ["green" if v >= 1.0 else "steelblue" if v >= 0 else "red" for v in vals]
    ax.bar(vals.index, vals.values, color=colors, alpha=0.8)
    ax.axhline(1.0, color="green", lw=1, linestyle="--", label="Min bar (Sharpe=1.0)")
    ax.axhline(0, color="black", lw=0.5)

    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_signal_count_vs_sharpe(
    comparison_table: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """Scatter: signal count vs net Sharpe — do more signals = lower quality?"""
    fig, ax = plt.subplots(figsize=(7, 5))

    for label, row in comparison_table.iterrows():
        ax.scatter(row.get("n_trades", 0), row.get("sharpe", 0), s=100, label=label)
        ax.annotate(label, (row.get("n_trades", 0), row.get("sharpe", 0)),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.axhline(1.0, color="green", lw=1, linestyle="--", label="Sharpe = 1.0")
    ax.set_xlabel("Number of Trades (Test Set)")
    ax.set_ylabel("Net Sharpe Ratio")
    ax.set_title("Signal Count vs Quality")
    ax.legend()

    plt.tight_layout()
    _save(fig, save_path)
    return fig
