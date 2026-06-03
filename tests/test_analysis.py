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
