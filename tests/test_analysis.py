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
