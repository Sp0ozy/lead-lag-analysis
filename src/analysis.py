"""Statistical analysis: ADF tests, lagged correlation, Bonferroni correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series) -> dict:
    """Run ADF stationarity test; return result dict."""
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "name": series.name,
        "statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "n_obs": result[3],
        "stationary": result[1] < 0.05,
    }


def run_eda_stats(returns: pd.DataFrame) -> list[dict]:
    """Run ADF on each return series; print and return results."""
    results = []
    print("\n=== ADF Stationarity Tests ===")
    for col in returns.columns:
        r = adf_test(returns[col])
        conclusion = "STATIONARY" if r["stationary"] else "NON-STATIONARY"
        print(f"  {col:10s}  stat={r['statistic']:7.3f}  p={r['p_value']:.4f}  [{conclusion}]")
        results.append(r)
    return results


def lagged_correlation(
    crypto: pd.Series,
    equity: pd.Series,
    lags: list[int],
) -> pd.DataFrame:
    """
    For each lag k > 0, compute Pearson r between crypto[t] and equity[t+k].

    k > 0 strictly: this measures lead effect, not contemporaneous correlation.
    """
    rows = []
    for k in lags:
        assert k > 0, "Lag must be > 0; k=0 is contemporaneous, not predictive."
        # crypto[t] predicts equity[t+k]: shift equity back by k
        equity_shifted = equity.shift(-k)
        paired = pd.DataFrame({"crypto": crypto, "equity": equity_shifted}).dropna()
        r, p = stats.pearsonr(paired["crypto"], paired["equity"])
        rows.append({"lag": k, "r": r, "p_raw": p, "n": len(paired)})
    return pd.DataFrame(rows).set_index("lag")


def apply_bonferroni(results: pd.DataFrame, n_tests: int) -> pd.DataFrame:
    """Add Bonferroni-corrected p-values and significance flags."""
    df = results.copy()
    df["p_bonferroni"] = (df["p_raw"] * n_tests).clip(upper=1.0)
    df["significant_raw"] = df["p_raw"] < 0.05
    df["significant_bonferroni"] = df["p_bonferroni"] < 0.05
    return df


def run_lag_analysis(
    returns: pd.DataFrame,
    lags: list[int] = [1, 2, 3, 5, 10],
) -> dict[str, pd.DataFrame]:
    """
    Run lagged correlation for BTC→SPX and ETH→SPX.
    Returns dict with keys 'BTC-USD' and 'ETH-USD'.
    """
    n_tests = len(lags) * 2  # 2 assets
    results = {}
    print("\n=== Lag Correlation Results ===")
    for crypto_col in ["BTC-USD", "ETH-USD"]:
        df = lagged_correlation(returns[crypto_col], returns["^GSPC"], lags)
        df = apply_bonferroni(df, n_tests)
        df["asset"] = crypto_col
        results[crypto_col] = df
        print(f"\n  {crypto_col} → ^GSPC  (Bonferroni n_tests={n_tests})")
        print(df[["r", "p_raw", "p_bonferroni", "significant_raw", "significant_bonferroni"]].to_string())
    return results


def summarize_lead_lag(results: dict[str, pd.DataFrame]) -> str:
    """Return plain-English verdict on lead-lag evidence."""
    any_bonf = any(
        df["significant_bonferroni"].any() for df in results.values()
    )
    any_raw = any(
        df["significant_raw"].any() for df in results.values()
    )
    if any_bonf:
        verdict = (
            "EVIDENCE OF LEAD EFFECT: At least one lag is significant after "
            "Bonferroni correction. Audit for leakage before claiming this is real."
        )
    elif any_raw:
        verdict = (
            "NO EVIDENCE OF LEAD EFFECT after Bonferroni correction. "
            "One or more lags appear nominally significant (p<0.05 uncorrected), "
            "but this is consistent with noise across 10 tests (EXPLORATORY ONLY)."
        )
    else:
        verdict = (
            "NO EVIDENCE OF LEAD EFFECT. No lag is significant even before "
            "Bonferroni correction."
        )
    print(f"\n=== Verdict ===\n{verdict}")
    return verdict


def select_var_lag(returns: pd.DataFrame, maxlag: int = 10) -> int:
    """Fit VAR on returns; return AIC-optimal lag order (minimum 1)."""
    from statsmodels.tsa.vector_ar.var_model import VAR
    model = VAR(returns.dropna())
    order_result = model.select_order(maxlags=maxlag)
    return max(1, int(order_result.aic))


def granger_bivariate(
    crypto: pd.Series,
    equity: pd.Series,
    maxlag: int,
) -> pd.DataFrame:
    """
    Test whether crypto Granger-causes equity at each lag 1..maxlag.

    Uses the F-test (ssr_ftest) from statsmodels.grangercausalitytests.
    Input order: equity is Y (to be forecast), crypto is X (potential cause).
    Returns DataFrame indexed by lag with columns f_stat and p_raw.
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    data = pd.DataFrame({"equity": equity, "crypto": crypto}).dropna()
    raw = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    rows = []
    for lag, test_result in raw.items():
        f_stat, p_val, _, _ = test_result[0]["ssr_ftest"]
        rows.append({"lag": lag, "f_stat": f_stat, "p_raw": p_val})
    return pd.DataFrame(rows).set_index("lag")


def run_granger_analysis(
    returns: pd.DataFrame,
    maxlag: int = 10,
) -> dict:
    """
    Orchestrate Granger causality analysis for BTC->SPX and ETH->SPX.

    Primary result: bivariate F-test at AIC-selected lag (Bonferroni n=2 assets).
    Comparison: fixed-lag sweep at {1,2,3,5,10} (Bonferroni n=10, matches Phase 3).
    Robustness: VAR Granger test at AIC-selected lag.
    """
    from statsmodels.tsa.vector_ar.var_model import VAR

    fixed_lags = [1, 2, 3, 5, 10]
    n_tests_aic = 2        # Bonferroni: 2 assets, 1 lag each
    n_tests_fixed = 10     # Bonferroni: 2 assets x 5 lags

    aic_lag = select_var_lag(returns, maxlag=maxlag)
    print(f"\nAIC-optimal VAR lag order: {aic_lag}")

    # Fit VAR for robustness check
    var_fit = VAR(returns.dropna()).fit(aic_lag)

    results: dict = {"aic_lag": aic_lag, "assets": {}}

    print("\n=== Granger Causality Results ===")
    for crypto_col in ["BTC-USD", "ETH-USD"]:
        # AIC-selected result
        aic_df = granger_bivariate(returns[crypto_col], returns["^GSPC"], maxlag=aic_lag)
        aic_row = aic_df.tail(1).copy()
        aic_row["p_bonferroni"] = (aic_row["p_raw"] * n_tests_aic).clip(upper=1.0)
        aic_row["significant_raw"] = aic_row["p_raw"] < 0.05
        aic_row["significant_bonferroni"] = aic_row["p_bonferroni"] < 0.05

        # Fixed-lag sweep
        full_df = granger_bivariate(returns[crypto_col], returns["^GSPC"], maxlag=max(fixed_lags))
        fixed_df = full_df.loc[fixed_lags].copy()
        fixed_df = apply_bonferroni(fixed_df, n_tests=n_tests_fixed)

        # VAR robustness
        var_test = var_fit.test_causality("^GSPC", [crypto_col], kind="f")
        var_pvalue = float(var_test.pvalue)

        results["assets"][crypto_col] = {
            "aic_result": aic_row,
            "fixed_sweep": fixed_df,
            "var_pvalue": var_pvalue,
        }

        print(f"\n  {crypto_col} -> ^GSPC")
        print(f"  AIC lag={aic_lag}: F={float(aic_row['f_stat'].iloc[0]):.3f}  "
              f"p_raw={float(aic_row['p_raw'].iloc[0]):.4f}  "
              f"p_bonferroni={float(aic_row['p_bonferroni'].iloc[0]):.4f}  "
              f"significant={bool(aic_row['significant_bonferroni'].iloc[0])}")
        print(f"  VAR robustness p={var_pvalue:.4f}")
        print(f"  Fixed-lag sweep:")
        print(fixed_df[["f_stat", "p_raw", "p_bonferroni", "significant_bonferroni"]].to_string())

    return results


def summarize_granger(results: dict) -> str:
    """Plain-English verdict on Granger causality evidence."""
    any_bonf = any(
        bool(d["aic_result"]["significant_bonferroni"].any()) or
        bool(d["fixed_sweep"]["significant_bonferroni"].any())
        for d in results["assets"].values()
    )
    any_raw = any(
        bool(d["aic_result"]["significant_raw"].any()) or
        bool(d["fixed_sweep"]["significant_raw"].any())
        for d in results["assets"].values()
    )
    if any_bonf:
        verdict = (
            "GRANGER CAUSALITY DETECTED: at least one test is significant after "
            "Bonferroni correction. Audit for spurious regression before claiming "
            "this is real."
        )
    elif any_raw:
        verdict = (
            "NO GRANGER CAUSALITY after Bonferroni correction. One or more lags "
            "are nominally significant (p<0.05 uncorrected) -- consistent with noise "
            "across 10 simultaneous tests (EXPLORATORY ONLY)."
        )
    else:
        verdict = (
            "NO GRANGER CAUSALITY. Crypto returns do not improve SPX forecasts "
            "beyond SPX's own history at any lag tested."
        )
    print(f"\n=== Granger Verdict ===\n{verdict}")
    return verdict
