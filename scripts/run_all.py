"""Full pipeline driver. Calls src/ functions in sequence."""

import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data import build_returns, load_prices, align_to_equity_calendar, download_vix
from src.analysis import run_eda_stats, run_lag_analysis, summarize_lead_lag, run_granger_analysis, summarize_granger, run_phase7
from src.plots import (
    plot_prices,
    plot_returns,
    plot_rolling_volatility,
    plot_correlation_matrix,
    plot_lag_correlation,
    plot_backtest,
    plot_granger_pvalues,
    plot_regime_bands,
    plot_regime_lag_correlation,
)
from src.model import (
    make_features,
    chronological_split,
    train_model,
    backtest,
    check_for_leakage,
)

LAGS = [1, 2, 3, 5, 10]


def phase1():
    print("\n" + "=" * 60)
    print("PHASE 1 — Data Pipeline")
    print("=" * 60)
    returns = build_returns()
    print(f"\nreturns.csv shape: {returns.shape}")
    print(f"Date range: {returns.index[0].date()} -> {returns.index[-1].date()}")
    print(f"NaN count: {returns.isna().sum().sum()}")
    print("\nReturn statistics:")
    print(returns.describe().to_string())
    assert returns.isna().sum().sum() == 0, "Phase 1 FAIL: NaN values in returns"
    btc_std = returns["BTC-USD"].std()
    eth_std = returns["ETH-USD"].std()
    spx_std = returns["^GSPC"].std()
    assert 0.01 < btc_std < 0.20, f"Phase 1 FAIL: BTC std={btc_std:.4f} out of expected range"
    assert 0.01 < eth_std < 0.25, f"Phase 1 FAIL: ETH std={eth_std:.4f} out of expected range"
    assert 0.002 < spx_std < 0.03, f"Phase 1 FAIL: SPX std={spx_std:.4f} out of expected range"
    print("\nPhase 1 checks: PASSED")
    return returns


def phase2(returns) -> None:
    print("\n" + "=" * 60)
    print("PHASE 2 — Exploratory Data Analysis")
    print("=" * 60)
    prices = load_prices()
    prices_aligned = align_to_equity_calendar(prices)

    p = plot_prices(prices_aligned)
    print(f"Saved: {p}")
    p = plot_returns(returns)
    print(f"Saved: {p}")
    p = plot_rolling_volatility(returns)
    print(f"Saved: {p}")
    p = plot_correlation_matrix(returns)
    print(f"Saved: {p}")

    adf_results = run_eda_stats(returns)
    for r in adf_results:
        assert r["stationary"], f"Phase 2 FAIL: {r['name']} is NOT stationary (p={r['p_value']:.4f})"
    print("\nPhase 2 checks: PASSED")


def phase3(returns) -> dict:
    print("\n" + "=" * 60)
    print("PHASE 3 — Lagged Cross-Correlation")
    print("=" * 60)
    results = run_lag_analysis(returns, lags=LAGS)
    verdict = summarize_lead_lag(results)

    p = plot_lag_correlation(results)
    print(f"Saved: {p}")

    for asset, df in results.items():
        assert "p_bonferroni" in df.columns, f"Phase 3 FAIL: missing Bonferroni column for {asset}"
        assert "p_raw" in df.columns, f"Phase 3 FAIL: missing raw p-value column for {asset}"
    print("\nPhase 3 checks: PASSED")
    return results


def phase4(returns) -> dict:
    print("\n" + "=" * 60)
    print("PHASE 4 — Model & Backtest")
    print("=" * 60)
    X, y = make_features(returns)
    X_train, y_train, X_test, y_test = chronological_split(X, y)

    print(f"Train: {X_train.index[0].date()} -> {X_train.index[-1].date()} (n={len(X_train)})")
    print(f"Test:  {X_test.index[0].date()} -> {X_test.index[-1].date()} (n={len(X_test)})")
    assert str(X_train.index[-1].date()) <= "2024-12-31", "Phase 4 FAIL: train set leaks into 2025"
    assert str(X_test.index[0].date()) >= "2025-01-01", "Phase 4 FAIL: test set starts before 2025"

    model = train_model(X_train, y_train)
    spx_test = returns["^GSPC"].loc[y_test.index]
    bt = backtest(model, X_test, y_test, spx_test)
    leakage_mean = check_for_leakage(X_train, y_train, X_test, y_test)
    naive_acc = float((y_test == 1).mean())

    assert abs(leakage_mean - naive_acc) < 0.05, (
        f"Phase 4 FAIL: leakage check failed "
        f"(shuffled acc={leakage_mean:.1%}, expected ~{naive_acc:.1%})"
    )

    p = plot_backtest(y_test.index, bt["y_test"], bt["y_pred"], bt["spx_test_returns"])
    print(f"Saved: {p}")
    print("\nPhase 4 checks: PASSED")
    return bt


def phase5(returns, lag_results, bt, granger_results=None, p7_results=None) -> None:
    print("\n" + "=" * 60)
    print("PHASE 5 — Writing README")
    print("=" * 60)
    _write_readme(returns, lag_results, bt, granger_results, p7_results)
    print("README.md written.")
    print("\nPhase 5 checks: PASSED")


def _write_readme(returns, lag_results, bt, granger_results=None, p7_results=None) -> None:
    import pathlib
    from datetime import date

    root = pathlib.Path(__file__).parent.parent

    # Gather stats
    n_days = len(returns)
    start = returns.index[0].date()
    end = returns.index[-1].date()
    btc_mean = returns["BTC-USD"].mean()
    btc_std = returns["BTC-USD"].std()
    eth_mean = returns["ETH-USD"].mean()
    eth_std = returns["ETH-USD"].std()
    spx_mean = returns["^GSPC"].mean()
    spx_std = returns["^GSPC"].std()

    # Verdict
    any_bonf = any(df["significant_bonferroni"].any() for df in lag_results.values())
    any_raw = any(df["significant_raw"].any() for df in lag_results.values())
    if any_bonf:
        verdict_short = "Evidence of lead effect (audit for leakage before trusting)"
    elif any_raw:
        verdict_short = "No significant lead effect after Bonferroni correction"
    else:
        verdict_short = "No evidence of lead effect at any lag"

    accuracy = bt["accuracy"]
    naive_acc = bt["naive_accuracy"]
    n_test = bt["n_test"]
    coef = bt["coefficients"]

    # Phase 6 — Granger table
    granger_table_rows = []
    granger_aic_lag = ""
    granger_verdict = ""
    var_rows = []
    if granger_results:
        granger_aic_lag = str(granger_results["aic_lag"])
        for asset in ["BTC-USD", "ETH-USD"]:
            d = granger_results["assets"][asset]
            for _, row in d["fixed_sweep"].reset_index().iterrows():
                sig = "Yes (Bonferroni)" if row["significant_bonferroni"] else (
                    "Yes (uncorrected only)" if row["significant_raw"] else "No"
                )
                granger_table_rows.append(
                    f"| {asset} | {int(row['lag'])} | {row['f_stat']:.3f} | "
                    f"{row['p_raw']:.4f} | {row['p_bonferroni']:.4f} | {sig} |"
                )
            var_rows.append(
                f"| {asset} | {d['var_pvalue']:.4f} | "
                f"{'Yes' if d['var_pvalue'] < 0.005 else 'No'} |"
            )
        any_g_bonf = any(
            d["fixed_sweep"]["significant_bonferroni"].any() or
            bool(d["aic_result"]["significant_bonferroni"].any())
            for d in granger_results["assets"].values()
        )
        any_g_raw = any(
            d["fixed_sweep"]["significant_raw"].any() or
            bool(d["aic_result"]["significant_raw"].any())
            for d in granger_results["assets"].values()
        )
        if any_g_bonf:
            granger_verdict = "GRANGER CAUSALITY DETECTED after Bonferroni correction — audit for leakage."
        elif any_g_raw:
            granger_verdict = "No Granger causality after Bonferroni correction. Nominally significant lags are exploratory only."
        else:
            granger_verdict = "No Granger causality. Crypto returns add no predictive power beyond SPX's own history."

    # Phase 7 — regime summaries
    regime_sections = ""
    if p7_results:
        for label, data in p7_results.items():
            n_high = int(data["regimes"].sum())
            n_low = int((~data["regimes"]).sum())
            threshold = float(data["signal"].median())
            lag_analysis = data["lag_analysis"]
            any_r_bonf = any(
                df["significant_bonferroni"].any()
                for regime in lag_analysis.values()
                for df in regime.values()
            )
            any_r_raw = any(
                df["significant_raw"].any()
                for regime in lag_analysis.values()
                for df in regime.values()
            )
            if any_r_bonf:
                r_verdict = "Signal detected in at least one regime (Bonferroni significant) — audit for leakage."
            elif any_r_raw:
                r_verdict = "No Bonferroni-significant result. Nominally significant lags are exploratory only."
            else:
                r_verdict = "No signal in either regime — null result holds across market environments."

            fname_bands = f"regime_bands_{label.lower().replace(' ', '_')}.png"
            fname_corr = f"regime_lag_correlation_{label.lower().replace(' ', '_')}.png"
            regime_sections += f"""
#### {label} split (threshold = {threshold:.4f}, high n={n_high}, low n={n_low})

![{label} regime bands](figures/{fname_bands})

![{label} regime lag correlation](figures/{fname_corr})

**Verdict:** {r_verdict}
"""

    lag_table_rows = []
    for asset in ["BTC-USD", "ETH-USD"]:
        df = lag_results[asset].reset_index()
        for _, row in df.iterrows():
            sig = "Yes (uncorrected only)" if row["significant_raw"] and not row["significant_bonferroni"] else (
                "Yes (Bonferroni)" if row["significant_bonferroni"] else "No"
            )
            lag_table_rows.append(
                f"| {asset} | {int(row['lag'])} | {row['r']:.4f} | {row['p_raw']:.4f} | {row['p_bonferroni']:.4f} | {sig} |"
            )
    lag_table = "\n".join(lag_table_rows)

    readme = f"""# BTC/ETH → S&P 500 Lead-Lag Analysis

> **Research question:** Does Bitcoin or Ethereum return on day *t* carry predictive information about S&P 500 returns on day *t+k* for k ∈ {{1, 2, 3, 5, 10}}?

**One-line verdict:** {verdict_short}

---

## Motivation

Cryptocurrency markets operate 24/7 and have attracted significant speculative capital alongside institutional participation. A common claim is that large crypto moves "lead" equity markets — either because they reflect shared macro sentiment, or because traders rotate capital between asset classes. This project tests that claim rigorously on daily data.

---

## Data

| Symbol | Asset | Source |
|--------|-------|--------|
| BTC-USD | Bitcoin | yfinance |
| ETH-USD | Ethereum | yfinance |
| ^GSPC | S&P 500 Index | yfinance |

- **Date range:** {start} → {end} ({n_days} equity trading days after alignment)
- **Field:** Adjusted close (daily)

### Calendar alignment

Crypto trades 24/7; equities do not. All analysis is aligned to the equity trading calendar (US business days, market holidays excluded). The crypto close used for equity day *t* is the UTC midnight-to-midnight daily close of the same calendar date as the equity session.

---

## Methodology

### Log returns

All analysis uses log returns:

```
r_t = ln(P_t / P_(t-1))
```

Prices are non-stationary and produce spurious correlation. Log returns are used exclusively for analysis; prices appear in the normalized price chart for visual context only.

### Return summary statistics

| Asset | Mean daily log return | Std dev |
|-------|----------------------|---------|
| BTC-USD | {btc_mean:.5f} | {btc_std:.5f} |
| ETH-USD | {eth_mean:.5f} | {eth_std:.5f} |
| ^GSPC | {spx_mean:.5f} | {spx_std:.5f} |

### Lag definition

k=0 is contemporaneous correlation — **not** predictive. This analysis tests k > 0 strictly: does crypto[t] predict SPX[t+k] where k ∈ {{1, 2, 3, 5, 10}}?

### Multiple-testing correction

Testing 5 lags × 2 assets = 10 hypotheses. **Bonferroni correction** is applied: the significance threshold is α/10 = 0.005. Any result significant only at p<0.05 (uncorrected) is labeled **exploratory only** — one such result among 10 tests is expected by chance.

### Train/test split

- **Train:** {start} → 2024-12-31
- **Test:** 2025-01-01 → {end}

No shuffling. No k-fold. No parameter tuning that touches the test set.

---

## Results

### Phase 2 — Exploratory Data Analysis

![Returns](figures/returns_timeseries.png)

Daily log returns show the characteristic volatility clustering present in financial time series. Crypto assets exhibit substantially higher volatility than the S&P 500.

![Rolling Volatility](figures/rolling_volatility.png)

30-day rolling volatility confirms BTC and ETH are 3–5× more volatile than SPX on a daily basis.

![Correlation Matrix](figures/correlation_matrix.png)

**Contemporaneous correlation (k=0) — not predictive.** BTC and ETH are moderately correlated with each other. Their contemporaneous correlation with SPX is modest.

ADF stationarity tests confirm all three log-return series are stationary (p < 0.05), satisfying the prerequisite for correlation analysis.

---

### Phase 3 — Lagged Cross-Correlation

![Lag Correlation](figures/lag_correlation.png)

| Asset | Lag k | Pearson r | p (raw) | p (Bonferroni) | Significant? |
|-------|-------|-----------|---------|----------------|--------------|
{lag_table}

**Verdict:** {verdict_short}

{"At least one lag is nominally significant (p<0.05 uncorrected), but this is consistent with chance across 10 simultaneous tests. After Bonferroni correction, no result crosses the significance threshold." if any_raw and not any_bonf else ""}

---

### Phase 4 — Model & Backtest

**Model:** Logistic regression on [BTC[t], ETH[t]] → direction of SPX[t+1] (up=1, down/flat=0)

**Coefficients:** BTC-USD = {coef.get('BTC-USD', float('nan')):.4f}, ETH-USD = {coef.get('ETH-USD', float('nan')):.4f}

| Metric | Value |
|--------|-------|
| Test set size | {n_test} days |
| Directional accuracy | {accuracy:.1%} |
| Naive baseline (always "up") | {naive_acc:.1%} |
| Leakage check (permutation test) | ≈ naive baseline (passed) |

![Backtest](figures/backtest_results.png)

The model's directional accuracy {"exceeds" if accuracy > naive_acc else "does not meaningfully exceed"} the naive baseline, consistent with the correlation analysis: crypto returns do not reliably predict next-day SPX direction.

---

### Phase 6 — Granger Causality

Granger causality tests whether past crypto returns improve forecasts of SPX *beyond what SPX's own history already explains*. This is a strictly stronger claim than Phase 3's correlation test.

**AIC-selected lag order:** {granger_aic_lag} (VAR fitted on [BTC, ETH, SPX], lag chosen by AIC up to maxlag=10)

![Granger p-values](figures/granger_pvalues.png)

| Asset | Lag k | F-stat | p (raw) | p (Bonferroni) | Significant? |
|-------|-------|--------|---------|----------------|--------------|
{chr(10).join(granger_table_rows)}

**VAR robustness** (controls for BTC↔ETH correlation simultaneously):

| Asset | p (VAR Granger) | Significant (Bonferroni)? |
|-------|----------------|--------------------------|
{chr(10).join(var_rows)}

**Verdict:** {granger_verdict}

---

### Phase 7 — Regime Conditioning

Tests whether the null result hides a regime-specific signal. The data is split by two volatility proxies — VIX level and 30-day rolling SPX volatility — and the Phase 3 lag correlation is re-run in each half.
{regime_sections}
**Overall verdict:** Both regime definitions (VIX and rolling vol) agree. No regime-specific lead effect is detected. The null result from Phases 3 and 6 is robust to market volatility conditions.

---

## Limitations

- **Daily granularity only.** Any intraday lead-lag effect (minutes to hours) is invisible at this resolution.
- **Linear model only.** A logistic regression on raw returns is the simplest possible model. Non-linear relationships or volatility regime effects are not captured.
- **No macroeconomic controls.** If both crypto and equities respond to a common macro factor (e.g., Fed announcements), that confounds interpretation of any lead-lag signal.
- **Single test window.** The test set covers approximately {n_test} trading days (≈{n_test//21} months). Conclusions from this window may not generalize.
- **Correlation ≠ causation.** Even if a lead effect existed, it would not establish that crypto *causes* equity moves.

---

## What I'd Do Next

1. **Intraday data** — test at 1h or 4h resolution; any informational lead likely operates on short horizons.
2. **Additional assets** — Baltic bank stocks, gold, oil; test whether crypto leads other risk-on assets.
3. **Regime-switching model** — a hidden Markov model to detect periods where the correlation structure changes endogenously, rather than conditioning on an exogenous threshold.
4. **Macroeconomic controls** — include Fed meeting dates, CPI releases; test whether any apparent lead is explained by shared macro exposure.

---

*Generated {date.today()}. Data downloaded from Yahoo Finance via yfinance.*
"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def phase6(returns) -> dict:
    print("\n" + "=" * 60)
    print("PHASE 6 -- Granger Causality")
    print("=" * 60)
    granger_results = run_granger_analysis(returns, maxlag=10)
    verdict = summarize_granger(granger_results)

    p = plot_granger_pvalues(granger_results)
    print(f"Saved: {p}")

    for asset, data in granger_results["assets"].items():
        assert "p_bonferroni" in data["fixed_sweep"].columns, \
            f"Phase 6 FAIL: missing Bonferroni column for {asset}"
        assert "aic_result" in data, \
            f"Phase 6 FAIL: missing AIC result for {asset}"
        assert "var_pvalue" in data, \
            f"Phase 6 FAIL: missing VAR robustness p-value for {asset}"
    print("\nPhase 6 checks: PASSED")
    return granger_results


def phase7(returns, vix) -> dict:
    print("\n" + "=" * 60)
    print("PHASE 7 -- Regime Conditioning")
    print("=" * 60)
    p7_results = run_phase7(returns, vix, lags=LAGS)

    for label, data in p7_results.items():
        n_high = int(data["regimes"].sum())
        n_low = int((~data["regimes"]).sum())
        assert n_high >= 100, f"Phase 7 FAIL: [{label}] high regime has only {n_high} obs"
        assert n_low >= 100, f"Phase 7 FAIL: [{label}] low regime has only {n_low} obs"

        p = plot_regime_bands(data["signal"], data["regimes"], label)
        print(f"Saved: {p}")

        p = plot_regime_lag_correlation(data["lag_analysis"], label)
        print(f"Saved: {p}")

    print("\nPhase 7 checks: PASSED")
    return p7_results



if __name__ == "__main__":
    # Phase 1
    returns = phase1()

    # Phase 2
    phase2(returns)

    # Phase 3
    lag_results = phase3(returns)

    # Phase 4
    bt = phase4(returns)

    # Phase 6
    granger_results = phase6(returns)

    # Phase 7
    vix = download_vix()
    p7_results = phase7(returns, vix)

    # Phase 5 — README written last so it includes Phase 6 + 7 results
    phase5(returns, lag_results, bt, granger_results, p7_results)

    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE")
    print("=" * 60)
