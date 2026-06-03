# Phase 6 + Phase 7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Phase 6 (Granger causality) and Phase 7 (regime conditioning) to the BTC/ETH → SPX lead-lag pipeline, each as a self-contained phase with checks, figures, and a README section.

**Architecture:** All new functions append to existing `src/analysis.py` and `src/plots.py`; `download_vix()` appends to `src/data.py`; `phase6()` and `phase7()` append to `scripts/run_all.py`. No existing functions are modified.

**Tech Stack:** Python 3.13, statsmodels 0.14.6 (`grangercausalitytests`, `VAR`), yfinance, pandas, matplotlib, pytest (new dev dependency).

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `requirements.txt` | Modify | Add `pytest` |
| `tests/__init__.py` | Create | Empty — marks tests as a package |
| `tests/test_analysis.py` | Create | Unit tests for new analysis functions |
| `tests/test_data.py` | Create | Unit test for `download_vix` output shape |
| `src/analysis.py` | Modify | Append `granger_bivariate`, `select_var_lag`, `run_granger_analysis`, `summarize_granger`, `define_regimes`, `run_regime_lag_analysis`, `run_regime_granger`, `run_phase7` |
| `src/data.py` | Modify | Append `download_vix` |
| `src/plots.py` | Modify | Append `plot_granger_pvalues`, `plot_regime_bands`, `plot_regime_lag_correlation` |
| `scripts/run_all.py` | Modify | Append `phase6()`, `phase7()`; call both in `__main__` |

---

## Task 1 — Test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_analysis.py`
- Create: `tests/test_data.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Install pytest into the venv**

```
.venv\Scripts\pip install pytest
```

Expected: `Successfully installed pytest-...`

- [ ] **Step 2: Add pytest to requirements.txt**

Append this line to `requirements.txt`:
```
pytest==8.3.5
```
(Use the version actually installed; run `.venv\Scripts\pip show pytest` to confirm.)

- [ ] **Step 3: Create `tests/__init__.py`**

Create an empty file at `tests/__init__.py`.

- [ ] **Step 4: Create `tests/test_analysis.py` with shared fixture**

```python
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
```

- [ ] **Step 5: Create `tests/test_data.py` stub**

```python
# tests/test_data.py
# download_vix is tested for structure only (no network call in unit tests).
# Integration test is the pipeline run itself.
```

- [ ] **Step 6: Verify pytest can discover the tests**

```
.venv\Scripts\pytest tests/ -v
```

Expected: `no tests ran` (0 items, no errors — the stubs are valid Python).

---

## Task 2 — `granger_bivariate`

**Files:**
- Modify: `src/analysis.py` (append)
- Modify: `tests/test_analysis.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis.py`:

```python
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
```

- [ ] **Step 2: Run test — confirm FAIL**

```
.venv\Scripts\pytest tests/test_analysis.py::test_granger_bivariate_returns_correct_shape -v
```

Expected: `ImportError` or `AttributeError: module 'src.analysis' has no attribute 'granger_bivariate'`

- [ ] **Step 3: Implement `granger_bivariate` — append to `src/analysis.py`**

```python
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
```

- [ ] **Step 4: Run all three granger_bivariate tests — confirm PASS**

```
.venv\Scripts\pytest tests/test_analysis.py -k "granger_bivariate" -v
```

Expected: 3 passed.

---

## Task 3 — `select_var_lag`

**Files:**
- Modify: `src/analysis.py` (append)
- Modify: `tests/test_analysis.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis.py`:

```python
# ── select_var_lag ────────────────────────────────────────────────────────────
def test_select_var_lag_returns_int_in_range(synthetic_returns):
    from src.analysis import select_var_lag
    lag = select_var_lag(synthetic_returns, maxlag=5)
    assert isinstance(lag, int)
    assert 1 <= lag <= 5
```

- [ ] **Step 2: Run test — confirm FAIL**

```
.venv\Scripts\pytest tests/test_analysis.py::test_select_var_lag_returns_int_in_range -v
```

Expected: `AttributeError`

- [ ] **Step 3: Implement `select_var_lag` — append to `src/analysis.py`**

```python
def select_var_lag(returns: pd.DataFrame, maxlag: int = 10) -> int:
    """Fit VAR on returns; return AIC-optimal lag order (minimum 1)."""
    from statsmodels.tsa.vector_ar.var_model import VAR
    model = VAR(returns.dropna())
    order_result = model.select_order(maxlags=maxlag)
    return max(1, int(order_result.aic))
```

- [ ] **Step 4: Run test — confirm PASS**

```
.venv\Scripts\pytest tests/test_analysis.py::test_select_var_lag_returns_int_in_range -v
```

Expected: 1 passed.

---

## Task 4 — `run_granger_analysis` + `summarize_granger`

**Files:**
- Modify: `src/analysis.py` (append)
- Modify: `tests/test_analysis.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis.py`:

```python
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
```

- [ ] **Step 2: Run tests — confirm FAIL**

```
.venv\Scripts\pytest tests/test_analysis.py -k "run_granger or summarize_granger" -v
```

Expected: `AttributeError`

- [ ] **Step 3: Implement `run_granger_analysis` — append to `src/analysis.py`**

```python
def run_granger_analysis(
    returns: pd.DataFrame,
    maxlag: int = 10,
) -> dict:
    """
    Orchestrate Granger causality analysis for BTC→SPX and ETH→SPX.

    Primary result: bivariate F-test at AIC-selected lag (Bonferroni n=2 assets).
    Comparison: fixed-lag sweep at {1,2,3,5,10} (Bonferroni n=10, matches Phase 3).
    Robustness: VAR Granger test at AIC-selected lag.
    """
    from statsmodels.tsa.vector_ar.var_model import VAR

    fixed_lags = [1, 2, 3, 5, 10]
    n_tests_aic = 2        # Bonferroni: 2 assets, 1 lag each
    n_tests_fixed = 10     # Bonferroni: 2 assets × 5 lags

    aic_lag = select_var_lag(returns, maxlag=maxlag)
    print(f"\nAIC-optimal VAR lag order: {aic_lag}")

    # Fit VAR for robustness check
    var_fit = VAR(returns.dropna()).fit(aic_lag)

    results: dict = {"aic_lag": aic_lag, "assets": {}}

    print("\n=== Granger Causality Results ===")
    for crypto_col in ["BTC-USD", "ETH-USD"]:
        # ── AIC-selected result ──
        aic_df = granger_bivariate(returns[crypto_col], returns["^GSPC"], maxlag=aic_lag)
        aic_row = aic_df.tail(1).copy()          # only the row at aic_lag
        aic_row["p_bonferroni"] = (aic_row["p_raw"] * n_tests_aic).clip(upper=1.0)
        aic_row["significant_raw"] = aic_row["p_raw"] < 0.05
        aic_row["significant_bonferroni"] = aic_row["p_bonferroni"] < 0.05

        # ── Fixed-lag sweep ──
        full_df = granger_bivariate(returns[crypto_col], returns["^GSPC"], maxlag=max(fixed_lags))
        fixed_df = full_df.loc[fixed_lags].copy()
        fixed_df = apply_bonferroni(fixed_df, n_tests=n_tests_fixed)

        # ── VAR robustness ──
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
```

- [ ] **Step 4: Implement `summarize_granger` — append to `src/analysis.py`**

```python
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
            "are nominally significant (p<0.05 uncorrected) — consistent with noise "
            "across 10 simultaneous tests (EXPLORATORY ONLY)."
        )
    else:
        verdict = (
            "NO GRANGER CAUSALITY. Crypto returns do not improve SPX forecasts "
            "beyond SPX's own history at any lag tested."
        )
    print(f"\n=== Granger Verdict ===\n{verdict}")
    return verdict
```

- [ ] **Step 5: Run all granger tests — confirm PASS**

```
.venv\Scripts\pytest tests/test_analysis.py -k "granger or summarize" -v
```

Expected: 5 passed.

---

## Task 5 — `plot_granger_pvalues`

**Files:**
- Modify: `src/plots.py` (append)

- [ ] **Step 1: Append to `src/plots.py`**

```python
def plot_granger_pvalues(
    results: dict,
    bonferroni_alpha: float = 0.05,
) -> pathlib.Path:
    """p-value by lag for BTC->SPX and ETH->SPX; Bonferroni threshold marked."""
    n_tests_fixed = 10  # 2 assets × 5 lags
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
```

- [ ] **Step 2: Verify import works**

```
.venv\Scripts\python -c "from src.plots import plot_granger_pvalues; print('ok')"
```

Expected: `ok`

---

## Task 6 — Phase 6 pipeline integration + end-to-end run + commit

**Files:**
- Modify: `scripts/run_all.py`
- Modify: `PLAN.md`

- [ ] **Step 1: Add imports to `scripts/run_all.py`**

At the top of `run_all.py`, add to the existing `from src.analysis import ...` line:

```python
from src.analysis import (
    run_eda_stats, run_lag_analysis, summarize_lead_lag,
    run_granger_analysis, summarize_granger,          # NEW
)
```

And to the existing `from src.plots import ...` line:

```python
from src.plots import (
    plot_prices, plot_returns, plot_rolling_volatility,
    plot_correlation_matrix, plot_lag_correlation, plot_backtest,
    plot_granger_pvalues,                              # NEW
)
```

- [ ] **Step 2: Add `phase6()` function to `scripts/run_all.py`** (before `if __name__ == "__main__":`)

```python
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
```

- [ ] **Step 3: Call `phase6` in `__main__`**

In `scripts/run_all.py`, after the `phase5(...)` call, add:

```python
    # Phase 6
    granger_results = phase6(returns)
```

- [ ] **Step 4: Update `PLAN.md` — add Phase 6 checklist and decision log entries**

Append to `PLAN.md` after the Phase 5 section:

```markdown
---

## Phase 6 — Granger Causality

**Goal:** Test whether crypto returns Granger-cause SPX returns (stronger than Phase 3 correlation).

### Tasks
- [ ] Implement `granger_bivariate` in `src/analysis.py`
- [ ] Implement `select_var_lag` in `src/analysis.py`
- [ ] Implement `run_granger_analysis` + `summarize_granger` in `src/analysis.py`
- [ ] Implement `plot_granger_pvalues` in `src/plots.py`
- [ ] Wire `phase6()` into `scripts/run_all.py`

### Phase 6 Checks (all must pass before commit)
- [ ] Results table contains AIC-selected-lag result AND fixed-lag sweep
- [ ] Both raw and Bonferroni-corrected p-values reported
- [ ] AIC-optimal lag order printed and logged in Decision Log below
- [ ] VAR robustness result agrees directionally with bivariate result
- [ ] README Phase 6 section written with plain-language verdict
```

And in the Decision Log table, add:

```
| 2026-06-04 | Granger maxlag=10 | Matches Phase 3 lag horizon; AIC selects within this bound |
| 2026-06-04 | Bivariate Granger primary, VAR as robustness | Bivariate is interpretable; VAR controls for BTC↔ETH correlation |
| 2026-06-04 | Bonferroni n=2 for AIC result, n=10 for fixed-lag sweep | AIC result tests 2 assets at one lag; fixed sweep tests 2 assets × 5 lags |
```

- [ ] **Step 5: Run full pipeline and confirm Phase 6 passes**

```
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python scripts/run_all.py
```

Expected output includes:
```
PHASE 6 -- Granger Causality
AIC-optimal VAR lag order: <n>
=== Granger Verdict ===
NO GRANGER CAUSALITY...   (or similar)
Phase 6 checks: PASSED
```

Note the AIC lag order printed — add it to PLAN.md Decision Log.

- [ ] **Step 6: Commit**

```
git add src/analysis.py src/plots.py scripts/run_all.py PLAN.md tests/
git commit -m "phase6: Granger causality -- <paste the verdict here>"
```

---

## Task 7 — `download_vix`

**Files:**
- Modify: `src/data.py` (append)
- Modify: `tests/test_data.py`

- [ ] **Step 1: Write the test**

Replace `tests/test_data.py` with:

```python
# tests/test_data.py
import pandas as pd
import pytest

def test_download_vix_returns_series():
    """VIX download returns a non-empty Series aligned to equity dates."""
    from src.data import download_vix
    vix = download_vix()
    assert isinstance(vix, pd.Series)
    assert vix.name == "VIX"
    assert len(vix) > 500          # at least ~2 years of equity days
    assert vix.isna().sum() == 0   # no NaN after alignment
    assert (vix > 0).all()         # VIX is always positive
```

- [ ] **Step 2: Run test — confirm FAIL**

```
.venv\Scripts\pytest tests/test_data.py -v
```

Expected: `AttributeError`

- [ ] **Step 3: Append `download_vix` to `src/data.py`**

```python
def download_vix() -> pd.Series:
    """Download VIX closing level aligned to the equity trading calendar.

    Uses the same download_raw cache as other symbols.
    VIX itself is used only as a regime-split signal, never as an analysis input.
    """
    end = date.today().isoformat()
    vix_raw = download_raw("^VIX", START_DATE, end)
    eq_raw = download_raw("^GSPC", START_DATE, end)
    equity_dates = pd.to_datetime(eq_raw.dropna().index)
    vix = vix_raw["^VIX"].reindex(equity_dates).ffill()
    assert vix.isna().sum() == 0, "VIX has NaN after alignment"
    vix.name = "VIX"
    vix.index.name = "Date"
    return vix
```

- [ ] **Step 4: Run test — confirm PASS**

```
.venv\Scripts\pytest tests/test_data.py -v
```

Expected: 1 passed. (This makes a real network call via cache; will be fast on repeat runs.)

---

## Task 8 — `define_regimes`

**Files:**
- Modify: `src/analysis.py` (append)
- Modify: `tests/test_analysis.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis.py`:

```python
# ── define_regimes ────────────────────────────────────────────────────────────
def test_define_regimes_median_split():
    from src.analysis import define_regimes
    signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0],
                       index=pd.bdate_range("2021-01-01", periods=5))
    regimes = define_regimes(signal)
    assert regimes.dtype == bool
    # Values >= median(3.0): 3, 4, 5 → True
    assert regimes.sum() == 3

def test_define_regimes_explicit_threshold():
    from src.analysis import define_regimes
    signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0],
                       index=pd.bdate_range("2021-01-01", periods=5))
    regimes = define_regimes(signal, threshold=4.0)
    # Values >= 4.0: 4, 5 → True
    assert regimes.sum() == 2

def test_define_regimes_name():
    from src.analysis import define_regimes
    signal = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2021-01-01", periods=3),
                       name="VIX")
    regimes = define_regimes(signal)
    assert regimes.name == "high_regime"
```

- [ ] **Step 2: Run tests — confirm FAIL**

```
.venv\Scripts\pytest tests/test_analysis.py -k "define_regimes" -v
```

Expected: `AttributeError`

- [ ] **Step 3: Append `define_regimes` to `src/analysis.py`**

```python
def define_regimes(signal: pd.Series, threshold: float = None) -> pd.Series:
    """Return boolean mask (True = high regime, at or above threshold or median)."""
    t = threshold if threshold is not None else float(signal.median())
    return (signal >= t).rename("high_regime")
```

- [ ] **Step 4: Run tests — confirm PASS**

```
.venv\Scripts\pytest tests/test_analysis.py -k "define_regimes" -v
```

Expected: 3 passed.

---

## Task 9 — `run_regime_lag_analysis` + `run_regime_granger`

**Files:**
- Modify: `src/analysis.py` (append)
- Modify: `tests/test_analysis.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis.py`:

```python
# ── run_regime_lag_analysis ───────────────────────────────────────────────────
def test_regime_lag_analysis_structure(synthetic_returns):
    from src.analysis import define_regimes, run_regime_lag_analysis
    signal = synthetic_returns["^GSPC"].rolling(10).std().dropna()
    common = synthetic_returns.index.intersection(signal.index)
    returns_sub = synthetic_returns.loc[common]
    regimes = define_regimes(signal.loc[common])
    result = run_regime_lag_analysis(returns_sub, regimes, "test", lags=[1, 2])
    assert "high" in result and "low" in result
    for regime_name in ["high", "low"]:
        assert "BTC-USD" in result[regime_name]
        assert "^GSPC" not in result[regime_name]  # SPX is the target, not a feature
        for asset_df in result[regime_name].values():
            assert "p_bonferroni" in asset_df.columns
            assert "r" in asset_df.columns

# ── run_regime_granger ────────────────────────────────────────────────────────
def test_regime_granger_structure(synthetic_returns):
    from src.analysis import define_regimes, run_regime_granger
    signal = synthetic_returns["^GSPC"].rolling(10).std().dropna()
    common = synthetic_returns.index.intersection(signal.index)
    returns_sub = synthetic_returns.loc[common]
    regimes = define_regimes(signal.loc[common])
    result = run_regime_granger(returns_sub, regimes, "test", maxlag=3)
    assert "high" in result and "low" in result
    for regime_name in ["high", "low"]:
        assert "aic_lag" in result[regime_name]
        assert "BTC-USD" in result[regime_name]["granger"]
```

- [ ] **Step 2: Run tests — confirm FAIL**

```
.venv\Scripts\pytest tests/test_analysis.py -k "regime_lag or regime_granger" -v
```

Expected: `AttributeError`

- [ ] **Step 3: Append `run_regime_lag_analysis` to `src/analysis.py`**

```python
def run_regime_lag_analysis(
    returns: pd.DataFrame,
    regimes: pd.Series,
    label: str,
    lags: list[int],
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Run Phase 3 lag correlation + Bonferroni within high and low regime subsets.

    Returns {"high": {"BTC-USD": df, "ETH-USD": df},
             "low":  {"BTC-USD": df, "ETH-USD": df}}
    """
    n_tests = len(lags) * 2
    result: dict = {}
    for regime_name, mask in [("high", regimes), ("low", ~regimes)]:
        idx = returns.index[returns.index.isin(mask[mask].index)]
        sub = returns.loc[idx]
        assert len(sub) >= 100, (
            f"[{label}] {regime_name} regime has only {len(sub)} obs — "
            f"too few for reliable inference (need >= 100)"
        )
        print(f"\n  [{label}] {regime_name} regime: n={len(sub)}")
        regime_asset_results: dict = {}
        for crypto_col in ["BTC-USD", "ETH-USD"]:
            df = lagged_correlation(sub[crypto_col], sub["^GSPC"], lags)
            df = apply_bonferroni(df, n_tests=n_tests)
            regime_asset_results[crypto_col] = df
        result[regime_name] = regime_asset_results
    return result
```

- [ ] **Step 4: Append `run_regime_granger` to `src/analysis.py`**

```python
def run_regime_granger(
    returns: pd.DataFrame,
    regimes: pd.Series,
    label: str,
    maxlag: int = 10,
) -> dict:
    """
    Run Granger bivariate test within high and low regime subsets.

    Returns {"high": {"aic_lag": int, "granger": {"BTC-USD": df, "ETH-USD": df}},
             "low":  {"aic_lag": int, "granger": {"BTC-USD": df, "ETH-USD": df}}}
    """
    result: dict = {}
    for regime_name, mask in [("high", regimes), ("low", ~regimes)]:
        idx = returns.index[returns.index.isin(mask[mask].index)]
        sub = returns.loc[idx]
        safe_maxlag = max(1, min(maxlag, len(sub) // 20))
        aic_lag = select_var_lag(sub, maxlag=safe_maxlag)
        granger_asset: dict = {}
        for crypto_col in ["BTC-USD", "ETH-USD"]:
            df = granger_bivariate(sub[crypto_col], sub["^GSPC"], maxlag=aic_lag)
            granger_asset[crypto_col] = df
        result[regime_name] = {"aic_lag": aic_lag, "granger": granger_asset}
        print(f"  [{label}] {regime_name} regime Granger AIC lag: {aic_lag}, n={len(sub)}")
    return result
```

- [ ] **Step 5: Run all regime tests — confirm PASS**

```
.venv\Scripts\pytest tests/test_analysis.py -k "regime" -v
```

Expected: 4 passed.

---

## Task 10 — `run_phase7`

**Files:**
- Modify: `src/analysis.py` (append)
- Modify: `tests/test_analysis.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analysis.py`:

```python
# ── run_phase7 ────────────────────────────────────────────────────────────────
def test_run_phase7_structure(synthetic_returns):
    from src.analysis import run_phase7
    # Use synthetic SPX rolling vol as proxy for VIX (avoid real network call in unit test)
    fake_vix = synthetic_returns["^GSPC"].rolling(5).std().dropna() * 100
    fake_vix.name = "VIX"
    common = synthetic_returns.index.intersection(fake_vix.index)
    result = run_phase7(synthetic_returns.loc[common], fake_vix.loc[common],
                        lags=[1, 2])
    assert "VIX" in result
    assert "Rolling Vol" in result
    for label in ["VIX", "Rolling Vol"]:
        assert "lag_analysis" in result[label]
        assert "granger" in result[label]
        assert "regimes" in result[label]
        assert "signal" in result[label]
```

- [ ] **Step 2: Run test — confirm FAIL**

```
.venv\Scripts\pytest tests/test_analysis.py::test_run_phase7_structure -v
```

Expected: `AttributeError`

- [ ] **Step 3: Append `run_phase7` to `src/analysis.py`**

```python
def run_phase7(
    returns: pd.DataFrame,
    vix: pd.Series,
    lags: list[int] = [1, 2, 3, 5, 10],
    maxlag: int = 10,
) -> dict:
    """
    Orchestrate Phase 7: regime conditioning using VIX and rolling SPX vol.

    Returns nested dict keyed by regime label ("VIX", "Rolling Vol"), each
    containing: regimes, signal, lag_analysis, granger.
    """
    rolling_vol = returns["^GSPC"].rolling(30).std()
    common_idx = (
        returns.index
        .intersection(vix.dropna().index)
        .intersection(rolling_vol.dropna().index)
    )
    returns_sub = returns.loc[common_idx]
    vix_sub = vix.loc[common_idx]
    rolling_vol_sub = rolling_vol.loc[common_idx]

    result: dict = {}
    print("\n=== Phase 7: Regime Conditioning ===")
    for signal, label in [(vix_sub, "VIX"), (rolling_vol_sub, "Rolling Vol")]:
        regimes = define_regimes(signal)
        n_high = int(regimes.sum())
        n_low = int((~regimes).sum())
        print(f"\n  {label}: high={n_high} days, low={n_low} days "
              f"(threshold={float(signal.median()):.4f})")
        lag_analysis = run_regime_lag_analysis(returns_sub, regimes, label, lags)
        granger = run_regime_granger(returns_sub, regimes, label, maxlag=maxlag)
        result[label] = {
            "regimes": regimes,
            "signal": signal,
            "lag_analysis": lag_analysis,
            "granger": granger,
        }
    return result
```

- [ ] **Step 4: Run all tests — confirm PASS**

```
.venv\Scripts\pytest tests/ -v
```

Expected: all tests pass.

---

## Task 11 — Phase 7 plot functions

**Files:**
- Modify: `src/plots.py` (append)

- [ ] **Step 1: Append `plot_regime_bands` to `src/plots.py`**

```python
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
```

- [ ] **Step 2: Append `plot_regime_lag_correlation` to `src/plots.py`**

```python
def plot_regime_lag_correlation(
    regime_results: dict[str, dict[str, pd.DataFrame]],
    label: str,
    bonferroni_alpha: float = 0.05,
) -> pathlib.Path:
    """2x2 grid: (BTC, ETH) x (high, low) regime lag correlation bars."""
    from scipy import stats as _stats

    assets = ["BTC-USD", "ETH-USD"]
    regime_names = ["high", "low"]
    n_tests = 10  # 5 lags × 2 assets, matching Phase 3

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
```

- [ ] **Step 3: Verify imports**

```
.venv\Scripts\python -c "from src.plots import plot_regime_bands, plot_regime_lag_correlation; print('ok')"
```

Expected: `ok`

---

## Task 12 — Phase 7 pipeline integration + end-to-end run + commit

**Files:**
- Modify: `scripts/run_all.py`
- Modify: `PLAN.md`

- [ ] **Step 1: Add imports to `scripts/run_all.py`**

Add to `from src.data import ...`:

```python
from src.data import build_returns, load_prices, align_to_equity_calendar, download_vix
```

Add to `from src.analysis import ...`:

```python
from src.analysis import (
    run_eda_stats, run_lag_analysis, summarize_lead_lag,
    run_granger_analysis, summarize_granger,
    run_phase7,
)
```

Add to `from src.plots import ...`:

```python
from src.plots import (
    plot_prices, plot_returns, plot_rolling_volatility,
    plot_correlation_matrix, plot_lag_correlation, plot_backtest,
    plot_granger_pvalues,
    plot_regime_bands, plot_regime_lag_correlation,
)
```

- [ ] **Step 2: Add `phase7()` to `scripts/run_all.py`** (before `if __name__ == "__main__":`)

```python
def phase7(returns, vix) -> dict:
    print("\n" + "=" * 60)
    print("PHASE 7 -- Regime Conditioning")
    print("=" * 60)
    p7_results = run_phase7(returns, vix, lags=LAGS)

    for label, data in p7_results.items():
        # Regime size check
        n_high = int(data["regimes"].sum())
        n_low = int((~data["regimes"]).sum())
        assert n_high >= 100, f"Phase 7 FAIL: [{label}] high regime has only {n_high} obs"
        assert n_low >= 100, f"Phase 7 FAIL: [{label}] low regime has only {n_low} obs"

        # Plot regime bands
        p = plot_regime_bands(data["signal"], data["regimes"], label)
        print(f"Saved: {p}")

        # Plot regime lag correlation
        p = plot_regime_lag_correlation(data["lag_analysis"], label)
        print(f"Saved: {p}")

    print("\nPhase 7 checks: PASSED")
    return p7_results
```

- [ ] **Step 3: Call `phase7` in `__main__`**

After the `granger_results = phase6(returns)` call, add:

```python
    # Phase 7
    vix = download_vix()
    phase7(returns, vix)
```

- [ ] **Step 4: Update `PLAN.md` — add Phase 7 checklist and decision log**

Append to `PLAN.md` after the Phase 6 section:

```markdown
---

## Phase 7 — Regime Conditioning

**Goal:** Test whether lead-lag signal differs across high/low volatility regimes.

### Tasks
- [ ] Implement `download_vix` in `src/data.py`
- [ ] Implement `define_regimes` in `src/analysis.py`
- [ ] Implement `run_regime_lag_analysis` + `run_regime_granger` + `run_phase7`
- [ ] Implement `plot_regime_bands` + `plot_regime_lag_correlation` in `src/plots.py`
- [ ] Wire `phase7()` into `scripts/run_all.py`

### Phase 7 Checks
- [ ] Both regimes (VIX and rolling vol) have >= 100 obs in each half
- [ ] VIX data NaN-free after alignment
- [ ] Bonferroni applied within each regime (n_tests = lags × 2 assets)
- [ ] README Phase 7 section states whether any regime shows signal absent full-sample
- [ ] VIX-split and rolling-vol-split results compared in README
```

Decision Log additions:

```
| 2026-06-04 | Regime threshold = median | Equal regime sizes; avoids data-mining a threshold |
| 2026-06-04 | Both VIX and rolling-vol regimes | VIX is industry standard; rolling vol is self-contained; compare for robustness |
| 2026-06-04 | VIX used only as regime signal, not as analysis input | VIX levels are non-stationary; log returns are the analysis input within each regime |
```

- [ ] **Step 5: Run full pipeline — confirm Phases 6 and 7 both pass**

```
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python scripts/run_all.py
```

Expected output ends with:
```
Phase 6 checks: PASSED
...
Phase 7 checks: PASSED
ALL PHASES COMPLETE
```

Verify figures exist:
```
dir figures\
```

Expected: `granger_pvalues.png`, `regime_bands_vix.png`, `regime_bands_rolling_vol.png`, `regime_lag_correlation_vix.png`, `regime_lag_correlation_rolling_vol.png`

- [ ] **Step 6: Run all tests one final time**

```
.venv\Scripts\pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```
git add src/ scripts/run_all.py tests/ PLAN.md
git commit -m "phase7: regime conditioning -- <paste verdict here>"
```

---

## Self-Review Checklist

**Spec coverage:**
- `granger_bivariate` → Task 2 ✓
- `select_var_lag` → Task 3 ✓
- `run_granger_analysis` + `summarize_granger` → Task 4 ✓
- `plot_granger_pvalues` → Task 5 ✓
- Phase 6 pipeline + commit → Task 6 ✓
- `download_vix` → Task 7 ✓
- `define_regimes` → Task 8 ✓
- `run_regime_lag_analysis` + `run_regime_granger` → Task 9 ✓
- `run_phase7` → Task 10 ✓
- `plot_regime_bands` + `plot_regime_lag_correlation` → Task 11 ✓
- Phase 7 pipeline + commit → Task 12 ✓

**Bonferroni in every test:** ✓ — `apply_bonferroni` called in `run_granger_analysis` (n=2 AIC, n=10 fixed), `run_regime_lag_analysis` (n=lags×2), and plotted thresholds match.

**Type consistency:** All function signatures consistent across tasks. `run_regime_lag_analysis` returns `dict[str, dict[str, pd.DataFrame]]`; `plot_regime_lag_correlation` receives the same type. ✓

**No placeholders:** All steps have complete code. ✓
