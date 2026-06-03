"""All figure-generating functions. Never calls plt.show()."""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
FIGURES_DIR = ROOT / "figures"


def _save(fig: plt.Figure, name: str) -> pathlib.Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_prices(prices: pd.DataFrame) -> pathlib.Path:
    """Normalized price chart — visual context only, NOT used in analysis."""
    fig, ax = plt.subplots(figsize=(12, 5))
    normalized = prices / prices.iloc[0] * 100
    for col in normalized.columns:
        ax.plot(normalized.index, normalized[col], label=col)
    ax.set_title("Normalized Prices (base=100) — visual context only, not used in analysis")
    ax.set_ylabel("Indexed price (base 100)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    return _save(fig, "prices_normalized.png")


def plot_returns(returns: pd.DataFrame) -> pathlib.Path:
    """Time series of daily log returns for all assets."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, col in zip(axes, returns.columns):
        ax.plot(returns.index, returns[col], linewidth=0.6, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Log return")
        ax.set_title(col)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.suptitle("Daily Log Returns", fontsize=14, y=1.01)
    fig.tight_layout()
    return _save(fig, "returns_timeseries.png")


def plot_rolling_volatility(returns: pd.DataFrame, window: int = 30) -> pathlib.Path:
    """30-day rolling standard deviation of log returns."""
    fig, ax = plt.subplots(figsize=(12, 5))
    rolling_std = returns.rolling(window).std()
    for col in rolling_std.columns:
        ax.plot(rolling_std.index, rolling_std[col], label=col, linewidth=0.8)
    ax.set_title(f"{window}-Day Rolling Volatility (std of log returns)")
    ax.set_ylabel("Rolling std")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    return _save(fig, "rolling_volatility.png")


def plot_correlation_matrix(returns: pd.DataFrame) -> pathlib.Path:
    """Contemporaneous (k=0) correlation heatmap — labeled as NOT predictive."""
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = returns.corr()
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=ax)
    ticks = range(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title("Contemporaneous Correlation (k=0) — not predictive")
    fig.tight_layout()
    return _save(fig, "correlation_matrix.png")


def plot_lag_correlation(
    results: dict[str, pd.DataFrame],
    bonferroni_alpha: float = 0.05,
) -> pathlib.Path:
    """Lag (x) vs correlation (y) with Bonferroni threshold marked."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    assets = ["BTC-USD", "ETH-USD"]
    for ax, asset in zip(axes, assets):
        df = results[asset].reset_index()
        colors = ["tomato" if sig else "steelblue" for sig in df["significant_raw"]]
        ax.bar(df["lag"].astype(str), df["r"], color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        # Bonferroni-corrected threshold on correlation (approximate via t-distribution).
        # Use minimum n across lags for the most conservative (widest) threshold.
        n_min = int(df["n"].min())
        n_tests = len(df) * 2
        bonf_p = bonferroni_alpha / n_tests
        from scipy import stats as _stats
        t_crit = _stats.t.ppf(1 - bonf_p / 2, df=n_min - 2)
        r_crit = t_crit / np.sqrt(t_crit**2 + n_min - 2)
        ax.axhline(r_crit, color="red", linewidth=1, linestyle="--", label=f"Bonferroni threshold (±{r_crit:.3f})")
        ax.axhline(-r_crit, color="red", linewidth=1, linestyle="--")
        ax.set_xlabel("Lag k (days)")
        ax.set_ylabel("Pearson r")
        ax.set_title(f"{asset} → ^GSPC lead correlation\n(blue=not significant, red=p<0.05 uncorrected)")
        ax.legend(fontsize=8)
    fig.suptitle("Lagged Cross-Correlation: Crypto[t] vs SPX[t+k]", fontsize=13)
    fig.tight_layout()
    return _save(fig, "lag_correlation.png")


def plot_backtest(
    dates: pd.DatetimeIndex,
    y_true: pd.Series,
    y_pred: pd.Series,
    spx_returns: pd.Series,
) -> pathlib.Path:
    """Cumulative returns: model strategy vs buy-and-hold SPX."""
    # Strategy: go long SPX if model predicts up (1), else stay flat (0)
    long_only = (y_pred.values == 1).astype(float) * spx_returns.values
    bh = spx_returns.values

    cum_model = np.exp(np.cumsum(long_only))
    cum_bh = np.exp(np.cumsum(bh))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(dates, cum_bh, label="Buy & Hold SPX", color="steelblue")
    ax.plot(dates, cum_model, label="Model (long when predict up)", color="darkorange")
    ax.set_title("Cumulative Returns — Test Set (2025-01-01 onward)")
    ax.set_ylabel("Cumulative return (1 = no change)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    ax2 = axes[1]
    accuracy = (y_true.values == y_pred.values).mean()
    baseline = (y_true.values == 1).mean()  # naive "always up" baseline
    ax2.bar(["Naive baseline\n(always up)", "Model accuracy"], [baseline, accuracy], color=["gray", "darkorange"])
    ax2.set_ylim(0, 1)
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="50% (coin-flip, ignores class imbalance)")
    ax2.set_ylabel("Directional accuracy")
    ax2.set_title("Directional Accuracy vs Naive Baseline")
    ax2.legend()
    for i, v in enumerate([baseline, accuracy]):
        ax2.text(i, v + 0.01, f"{v:.1%}", ha="center")

    fig.tight_layout()
    return _save(fig, "backtest_results.png")


def plot_regime_bands(
    signal: pd.Series,
    regimes: pd.Series,
    label: str,
) -> pathlib.Path:
    """Time series of regime signal with high/low periods shaded in red/blue."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(signal.index, signal.values, color="steelblue", linewidth=0.8, label=label)
    regime_vals = regimes.reindex(signal.index).fillna(False)
    ax.fill_between(
        signal.index, float(signal.min()), float(signal.max()),
        where=regime_vals.values.astype(bool),
        alpha=0.2, color="tomato", label="High regime",
    )
    threshold = float(signal.median())
    ax.axhline(threshold, color="red", linestyle="--", linewidth=0.8,
               label=f"Median threshold ({threshold:.2f})")
    ax.set_title(f"{label}: High/Low Regime Split (median threshold)")
    ax.set_ylabel(label)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fname = f"regime_bands_{label.lower().replace(' ', '_')}.png"
    return _save(fig, fname)


def plot_regime_lag_correlation(
    regime_results: dict[str, dict[str, pd.DataFrame]],
    label: str,
    bonferroni_alpha: float = 0.05,
) -> pathlib.Path:
    """2x2 grid: (BTC, ETH) x (high, low) regime lag correlation bars."""
    from scipy import stats as _stats

    assets = ["BTC-USD", "ETH-USD"]
    regime_names = ["high", "low"]
    n_tests = 10  # 5 lags x 2 assets, matching Phase 3

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)

    for col_idx, asset in enumerate(assets):
        for row_idx, regime_name in enumerate(regime_names):
            ax = axes[row_idx][col_idx]
            df = regime_results[regime_name][asset].reset_index()
            colors = ["tomato" if sig else "steelblue"
                      for sig in df["significant_raw"]]
            ax.bar(df["lag"].astype(str), df["r"], color=colors, alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.5)
            n_min = max(int(df["n"].min()), 3)
            bonf_p = bonferroni_alpha / n_tests
            t_crit = _stats.t.ppf(1 - bonf_p / 2, df=n_min - 2)
            r_crit = t_crit / np.sqrt(t_crit**2 + n_min - 2)
            ax.axhline(r_crit, color="red", linewidth=1, linestyle="--",
                       label=f"Bonferroni (+-{r_crit:.3f})")
            ax.axhline(-r_crit, color="red", linewidth=1, linestyle="--")
            ax.set_title(f"{asset}  |  {regime_name.upper()} {label}")
            ax.set_xlabel("Lag k")
            ax.set_ylabel("Pearson r")
            ax.legend(fontsize=7)

    fig.suptitle(
        f"Regime-Conditional Lag Correlation ({label} split)\n"
        f"Red=p<0.05 uncorrected  |  Dashed=Bonferroni threshold",
        fontsize=12,
    )
    fig.tight_layout()
    fname = f"regime_lag_correlation_{label.lower().replace(' ', '_')}.png"
    return _save(fig, fname)


def plot_granger_pvalues(
    results: dict,
    bonferroni_alpha: float = 0.05,
) -> pathlib.Path:
    """p-value by lag for BTC->SPX and ETH->SPX; Bonferroni threshold marked."""
    n_tests_fixed = 10  # 2 assets x 5 lags
    bonf_threshold = bonferroni_alpha / n_tests_fixed
    aic_lag = results["aic_lag"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    assets = ["BTC-USD", "ETH-USD"]

    for ax, asset in zip(axes, assets):
        df = results["assets"][asset]["fixed_sweep"].reset_index()
        colors = ["tomato" if p < 0.05 else "steelblue" for p in df["p_raw"]]
        ax.bar(df["lag"].astype(str), df["p_raw"], color=colors, alpha=0.8)
        ax.axhline(0.05, color="orange", linewidth=1.2, linestyle="--",
                   label="p=0.05 (uncorrected)")
        ax.axhline(bonf_threshold, color="red", linewidth=1.2, linestyle="--",
                   label=f"p={bonf_threshold:.3f} (Bonferroni)")
        if aic_lag in df["lag"].values:
            ax.axvline(str(aic_lag), color="green", linewidth=1.5,
                       linestyle=":", label=f"AIC-selected lag={aic_lag}")
        ax.set_xlabel("Lag k (days)")
        ax.set_ylabel("p-value (F-test)")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"{asset} -> ^GSPC Granger p-values\n"
            f"(lower = stronger evidence; red dashed = Bonferroni threshold)"
        )
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Granger Causality: Crypto[t-k] improves forecast of SPX[t]?\n"
        f"Fixed-lag sweep  |  AIC-selected lag = {aic_lag}",
        fontsize=12,
    )
    fig.tight_layout()
    return _save(fig, "granger_pvalues.png")
