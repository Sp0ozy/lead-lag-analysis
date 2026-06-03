# Design: Phase 6 (Granger Causality) + Phase 7 (Regime Conditioning)

**Date:** 2026-06-04  
**Project:** BTC/ETH → S&P 500 Lead-Lag Analysis  
**Extends:** Phases 1–5 (daily log-return pipeline, lag correlation, logistic regression backtest)

---

## Context

Phases 1–5 established that BTC and ETH daily returns show **no significant lead effect on SPX** at lags {1, 2, 3, 5, 10} using Pearson cross-correlation and a logistic regression backtest. Two natural extensions deepen this finding:

1. **Granger causality** — a formally stronger test: does crypto's past add predictive power *beyond SPX's own past*?
2. **Regime conditioning** — does the null result hold in all market environments, or does a signal appear during high-volatility periods?

Both extensions operate on the existing daily data. No new data sources are required beyond ^VIX (downloaded via the same `yfinance` + `download_raw` infrastructure already in place).

---

## Phase 6 — Granger Causality

### Research question

Does knowing past BTC or ETH returns improve the forecast of SPX returns beyond what SPX's own past returns already explain?

Granger causality is strictly stronger than Phase 3's correlation test. A series X "Granger-causes" Y if lagged X significantly improves a regression of Y on its own lags. It does not imply economic causation — it is a predictability test.

### Methodology

**Primary result — AIC-selected lag, bivariate F-test:**
1. Fit a VAR on [BTC, ETH, SPX] jointly; select lag order by AIC (tested up to maxlag=10).
2. Run `statsmodels.tsa.stattools.grangercausalitytests` for BTC→SPX and ETH→SPX at the AIC-selected lag order.
3. Extract F-statistic and p-value for each asset at the selected lag.
4. Apply Bonferroni correction across 2 assets (n_tests=2 for the AIC-selected result).

**Comparison result — fixed-lag sweep:**
5. Run the bivariate Granger test at each of lags {1, 2, 3, 5, 10} (matching Phase 3).
6. Apply Bonferroni across 2 assets × 5 lags = 10 tests (matching Phase 3's correction).
7. Report alongside Phase 3 correlation results so the two tests are directly comparable.

**Robustness check — VAR:**
8. Fit the full VAR on [BTC, ETH, SPX] at the AIC-selected order.
9. Report whether VAR Granger tests agree with the bivariate results.
10. Print AIC-optimal lag order; log it in PLAN.md decision log.

### New functions in `src/analysis.py`

| Function | Signature | Purpose |
|----------|-----------|---------|
| `granger_bivariate` | `(crypto, equity, maxlag) → pd.DataFrame` | Wraps `grangercausalitytests`; returns lag, F-stat, p-value for lags 1..maxlag |
| `select_var_lag` | `(returns_df, maxlag=10) → int` | Fits VAR, returns AIC-optimal lag order |
| `run_granger_analysis` | `(returns_df, maxlag=10) → dict` | Orchestrates AIC selection, bivariate tests, fixed-lag sweep, Bonferroni, VAR robustness |
| `summarize_granger` | `(results) → str` | Plain-English verdict consistent with Phase 3 format |

### New figure in `src/plots.py`

| Function | Output |
|----------|--------|
| `plot_granger_pvalues` | p-value by lag for BTC→SPX and ETH→SPX; Bonferroni threshold marked; mirrors Phase 3 chart for visual comparison |

### Phase 6 checks (all must pass before commit)

- [ ] Results table contains AIC-selected-lag result AND fixed-lag sweep
- [ ] Both raw and Bonferroni-corrected p-values reported
- [ ] AIC-optimal lag order printed to stdout and logged in PLAN.md
- [ ] VAR robustness result agrees directionally with bivariate result (or discrepancy explained)
- [ ] README Phase 6 section written with plain-language verdict
- [ ] No result reported as "significant" without Bonferroni correction passing

### Commit message format

```
phase6: Granger causality — <verdict summary>
```

---

## Phase 7 — Regime Conditioning

### Research question

Does the lead-lag relationship (or absence thereof) hold uniformly across market environments? Specifically: does the Pearson correlation or Granger signal appear in high-volatility regimes but not low, or vice versa?

### Regime definitions (both applied, results compared)

| Regime signal | Definition | Threshold | Source |
|---------------|------------|-----------|--------|
| VIX level | Daily VIX closing level | Median VIX over full sample | ^VIX downloaded via `download_raw` |
| Rolling SPX vol | 30-day rolling std of SPX log returns | Median over full sample | Already computed in Phase 2 |

Both use the same `define_regimes(signal, threshold=None)` function. Median split is used by default (equal-sized regimes). No fixed threshold is imposed to avoid data mining.

### New data in `src/data.py`

| Function | Signature | Purpose |
|----------|-----------|---------|
| `download_vix` | `() → pd.Series` | Calls `download_raw("^VIX", ...)`, aligns to equity calendar, returns VIX closing level as a Series |

### New functions in `src/analysis.py`

| Function | Signature | Purpose |
|----------|-----------|---------|
| `define_regimes` | `(signal, threshold=None) → pd.Series[bool]` | Returns boolean mask (True = high regime); splits on median if no threshold given |
| `run_regime_lag_analysis` | `(returns_df, regimes, label, lags) → dict` | Runs Phase 3 lag correlation + Bonferroni within high and low subsets; returns `{"high": df, "low": df}` |
| `run_regime_granger` | `(returns_df, regimes, label, maxlag) → dict` | Runs Phase 6 Granger bivariate test within each regime subset |
| `run_phase7` | `(returns_df, vix) → dict` | Orchestrates both regime signals, both analyses; returns nested results dict |

### New figures in `src/plots.py`

| Function | Output |
|----------|--------|
| `plot_regime_bands` | Time series of regime signal (VIX or rolling vol) with high/low periods shaded; shows regime sizes |
| `plot_regime_lag_correlation` | 2×2 grid: BTC and ETH × high and low regime; mirrors Phase 3 figure |

### Phase 7 checks (all must pass before commit)

- [ ] Both VIX and rolling-vol regimes have ≥ 100 observations in each half
- [ ] VIX data is NaN-free after alignment
- [ ] Bonferroni applied within each regime using the same n_tests as Phase 3 (lags × 2 assets)
- [ ] README Phase 7 section states explicitly whether any regime shows signal absent in the full-sample result
- [ ] Results for VIX-split and rolling-vol-split are compared in README (do they agree?)

### Commit message format

```
phase7: regime conditioning — <verdict summary>
```

---

## Architecture — What Changes vs What Stays

### Files modified

| File | Change |
|------|--------|
| `src/data.py` | Add `download_vix()` |
| `src/analysis.py` | Add Phase 6 and Phase 7 functions as new independent blocks |
| `src/plots.py` | Add `plot_granger_pvalues`, `plot_regime_bands`, `plot_regime_lag_correlation` |
| `scripts/run_all.py` | Add `phase6(returns)` and `phase7(returns, vix)` functions; call them in `__main__` |
| `README.md` | Add Phase 6 and Phase 7 sections (auto-generated by `_write_readme`) |
| `PLAN.md` | Add Phase 6 and Phase 7 checklists; log AIC lag order in Decision Log |

### Files unchanged

`src/model.py`, existing Phase 1–5 functions in all files, `.gitignore`, `requirements.txt` (no new deps).

### No new dependencies

`statsmodels.tsa.stattools.grangercausalitytests` and `statsmodels.tsa.vector_ar.var_model.VAR` are both in the already-pinned `statsmodels==0.14.6`. ^VIX uses existing `yfinance` infrastructure.

---

## Statistical rules inherited from Phases 1–5 (no exceptions)

- All analysis on log returns, never prices
- Bonferroni correction on all reported significance claims
- ADF stationarity confirmed for VIX in regime sub-samples before Granger testing
- Chronological split boundary (2024-12-31 / 2025-01-01) respected if any in-sample vs out-of-sample distinction is made
- Null results reported honestly; no p-hacking across regime definitions

---

## Decision log entries to add to PLAN.md

| Decision | Rationale |
|----------|-----------|
| Regime threshold = median (not fixed value) | Avoids data mining a threshold; ensures equal regime sizes for comparable test power |
| Granger maxlag = 10 | Matches Phase 3; covers the same horizon; AIC selects within this bound |
| Both VIX and rolling-vol regimes | VIX is the industry standard; rolling vol is self-contained; comparing them checks robustness of the regime definition itself |
| Bivariate Granger as primary, VAR as robustness | Bivariate is more interpretable; VAR controls for the BTC↔ETH correlation but is harder to explain |
