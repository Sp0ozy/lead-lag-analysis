# tests/test_analysis.py
import numpy as np
import pandas as pd
import pytest

# ── shared synthetic returns fixture ──────────────────────────────────────────
@pytest.fixture
def synthetic_returns():
    """200 business days of independent random log returns for all three assets."""
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2021-01-01", periods=n)
    btc = pd.Series(np.random.normal(0, 0.03, n), index=dates, name="BTC-USD")
    eth = pd.Series(np.random.normal(0, 0.04, n), index=dates, name="ETH-USD")
    spx = pd.Series(np.random.normal(0, 0.01, n), index=dates, name="^GSPC")
    return pd.concat([btc, eth, spx], axis=1)

@pytest.fixture
def causal_returns():
    """BTC at lag 1 deterministically causes SPX — Granger test should detect this."""
    np.random.seed(0)
    n = 300
    dates = pd.bdate_range("2021-01-01", periods=n)
    btc = pd.Series(np.random.normal(0, 0.03, n), index=dates, name="BTC-USD")
    eth = pd.Series(np.random.normal(0, 0.04, n), index=dates, name="ETH-USD")
    # SPX[t] = 0.6 * BTC[t-1] + small noise — strong Granger signal at lag 1
    spx_vals = 0.6 * np.roll(btc.values, 1) + np.random.normal(0, 0.002, n)
    spx_vals[0] = np.random.normal(0, 0.01)
    spx = pd.Series(spx_vals, index=dates, name="^GSPC")
    return pd.concat([btc, eth, spx], axis=1)

# ── granger_bivariate ─────────────────────────────────────────────────────────
def test_granger_bivariate_returns_correct_shape(synthetic_returns):
    from src.analysis import granger_bivariate
    result = granger_bivariate(
        synthetic_returns["BTC-USD"], synthetic_returns["^GSPC"], maxlag=3
    )
    assert list(result.columns) == ["f_stat", "p_raw"]
    assert result.index.tolist() == [1, 2, 3]

def test_granger_bivariate_pvalues_in_range(synthetic_returns):
    from src.analysis import granger_bivariate
    result = granger_bivariate(
        synthetic_returns["BTC-USD"], synthetic_returns["^GSPC"], maxlag=5
    )
    assert (result["p_raw"] >= 0).all() and (result["p_raw"] <= 1).all()

def test_granger_bivariate_detects_true_cause(causal_returns):
    from src.analysis import granger_bivariate
    result = granger_bivariate(
        causal_returns["BTC-USD"], causal_returns["^GSPC"], maxlag=3
    )
    # With 0.6 * BTC[t-1] causing SPX[t], p at lag 1 must be very small
    assert result.loc[1, "p_raw"] < 0.001

# ── select_var_lag ────────────────────────────────────────────────────────────
def test_select_var_lag_returns_int_in_range(synthetic_returns):
    from src.analysis import select_var_lag
    lag = select_var_lag(synthetic_returns, maxlag=5)
    assert isinstance(lag, int)
    assert 1 <= lag <= 5

# ── run_granger_analysis ──────────────────────────────────────────────────────
def test_run_granger_analysis_structure(synthetic_returns):
    from src.analysis import run_granger_analysis
    results = run_granger_analysis(synthetic_returns, maxlag=5)
    assert "aic_lag" in results
    assert "BTC-USD" in results["assets"]
    assert "ETH-USD" in results["assets"]
    for asset_data in results["assets"].values():
        assert "aic_result" in asset_data
        assert "fixed_sweep" in asset_data
        assert "var_pvalue" in asset_data
        assert "p_bonferroni" in asset_data["fixed_sweep"].columns
        assert "significant_bonferroni" in asset_data["fixed_sweep"].columns

def test_summarize_granger_returns_string(synthetic_returns):
    from src.analysis import run_granger_analysis, summarize_granger
    results = run_granger_analysis(synthetic_returns, maxlag=3)
    verdict = summarize_granger(results)
    assert isinstance(verdict, str)
    assert len(verdict) > 0


# ── define_regimes ────────────────────────────────────────────────────────────
def test_define_regimes_median_split():
    from src.analysis import define_regimes
    signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0],
                       index=pd.bdate_range("2021-01-01", periods=5))
    regimes = define_regimes(signal)
    assert regimes.dtype == bool
    # Values >= median(3.0): 3, 4, 5 -> True
    assert regimes.sum() == 3

def test_define_regimes_explicit_threshold():
    from src.analysis import define_regimes
    signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0],
                       index=pd.bdate_range("2021-01-01", periods=5))
    regimes = define_regimes(signal, threshold=4.0)
    # Values >= 4.0: 4, 5 -> True
    assert regimes.sum() == 2

def test_define_regimes_name():
    from src.analysis import define_regimes
    signal = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2021-01-01", periods=3),
                       name="VIX")
    regimes = define_regimes(signal)
    assert regimes.name == "high_regime"


# ── run_regime_lag_analysis ───────────────────────────────────────────────────
def test_regime_lag_analysis_structure(synthetic_returns):
    from src.analysis import define_regimes, run_regime_lag_analysis
    # Use SPX column directly (200 rows) for regime signal — median split gives 100 per regime
    signal = synthetic_returns["^GSPC"]
    regimes = define_regimes(signal)
    result = run_regime_lag_analysis(synthetic_returns, regimes, "test", lags=[1, 2])
    assert "high" in result and "low" in result
    for regime_name in ["high", "low"]:
        assert "BTC-USD" in result[regime_name]
        assert "^GSPC" not in result[regime_name]
        for asset_df in result[regime_name].values():
            assert "p_bonferroni" in asset_df.columns
            assert "r" in asset_df.columns

# ── run_regime_granger ────────────────────────────────────────────────────────
def test_regime_granger_structure(synthetic_returns):
    from src.analysis import define_regimes, run_regime_granger
    # Use SPX column directly (200 rows) for regime signal — median split gives 100 per regime
    signal = synthetic_returns["^GSPC"]
    regimes = define_regimes(signal)
    result = run_regime_granger(synthetic_returns, regimes, "test", maxlag=3)
    assert "high" in result and "low" in result
    for regime_name in ["high", "low"]:
        assert "aic_lag" in result[regime_name]
        assert "BTC-USD" in result[regime_name]["granger"]


# ── run_phase7 ────────────────────────────────────────────────────────────────
def test_run_phase7_structure():
    from src.analysis import run_phase7
    # Use 400 rows so rolling(30) inside run_phase7 leaves 371 rows,
    # and median split gives ~185 per regime (>= 100 obs assertion).
    np.random.seed(7)
    n = 400
    dates = pd.bdate_range("2021-01-01", periods=n)
    returns = pd.DataFrame({
        "BTC-USD": np.random.normal(0, 0.03, n),
        "ETH-USD": np.random.normal(0, 0.04, n),
        "^GSPC": np.random.normal(0, 0.01, n),
    }, index=dates)
    fake_vix = pd.Series(np.random.uniform(10, 40, n), index=dates, name="VIX")
    result = run_phase7(returns, fake_vix, lags=[1, 2])
    assert "VIX" in result
    assert "Rolling Vol" in result
    for label in ["VIX", "Rolling Vol"]:
        assert "lag_analysis" in result[label]
        assert "granger" in result[label]
        assert "regimes" in result[label]
        assert "signal" in result[label]
