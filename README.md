# BTC/ETH → S&P 500 Lead-Lag Analysis

> **Research question:** Does Bitcoin or Ethereum return on day *t* carry predictive information about S&P 500 returns on day *t+k* for k ∈ {1, 2, 3, 5, 10}?

**One-line verdict:** No evidence of lead effect at any lag

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

- **Date range:** 2021-01-05 → 2026-06-03 (1359 equity trading days after alignment)
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
| BTC-USD | 0.00053 | 0.03665 |
| ETH-USD | 0.00040 | 0.04855 |
| ^GSPC | 0.00053 | 0.01052 |

### Lag definition

k=0 is contemporaneous correlation — **not** predictive. This analysis tests k > 0 strictly: does crypto[t] predict SPX[t+k] where k ∈ {1, 2, 3, 5, 10}?

### Multiple-testing correction

Testing 5 lags × 2 assets = 10 hypotheses. **Bonferroni correction** is applied: the significance threshold is α/10 = 0.005. Any result significant only at p<0.05 (uncorrected) is labeled **exploratory only** — one such result among 10 tests is expected by chance.

### Train/test split

- **Train:** 2021-01-05 → 2024-12-31
- **Test:** 2025-01-01 → 2026-06-03

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
| BTC-USD | 1 | 0.0144 | 0.5972 | 1.0000 | No |
| BTC-USD | 2 | -0.0168 | 0.5357 | 1.0000 | No |
| BTC-USD | 3 | -0.0036 | 0.8956 | 1.0000 | No |
| BTC-USD | 5 | -0.0194 | 0.4761 | 1.0000 | No |
| BTC-USD | 10 | -0.0267 | 0.3268 | 1.0000 | No |
| ETH-USD | 1 | 0.0369 | 0.1747 | 1.0000 | No |
| ETH-USD | 2 | -0.0274 | 0.3129 | 1.0000 | No |
| ETH-USD | 3 | 0.0128 | 0.6368 | 1.0000 | No |
| ETH-USD | 5 | -0.0214 | 0.4318 | 1.0000 | No |
| ETH-USD | 10 | -0.0148 | 0.5866 | 1.0000 | No |

**Verdict:** No evidence of lead effect at any lag



---

### Phase 4 — Model & Backtest

**Model:** Logistic regression on [BTC[t], ETH[t]] → direction of SPX[t+1] (up=1, down/flat=0)

**Coefficients:** BTC-USD = -0.0583, ETH-USD = 0.6535

| Metric | Value |
|--------|-------|
| Test set size | 355 days |
| Directional accuracy | 57.5% |
| Naive baseline (always "up") | 57.5% |
| Leakage check (permutation test) | ≈ naive baseline (passed) |

![Backtest](figures/backtest_results.png)

The model's directional accuracy does not meaningfully exceed the naive baseline, consistent with the correlation analysis: crypto returns do not reliably predict next-day SPX direction.

---

### Phase 6 — Granger Causality

Granger causality tests whether past crypto returns improve forecasts of SPX *beyond what SPX's own history already explains*. This is a strictly stronger claim than Phase 3's correlation test.

**AIC-selected lag order:** 1 (VAR fitted on [BTC, ETH, SPX], lag chosen by AIC up to maxlag=10)

![Granger p-values](figures/granger_pvalues.png)

| Asset | Lag k | F-stat | p (raw) | p (Bonferroni) | Significant? |
|-------|-------|--------|---------|----------------|--------------|
| BTC-USD | 1 | 0.749 | 0.3868 | 1.0000 | No |
| BTC-USD | 2 | 0.497 | 0.6084 | 1.0000 | No |
| BTC-USD | 3 | 0.584 | 0.6253 | 1.0000 | No |
| BTC-USD | 5 | 0.730 | 0.6013 | 1.0000 | No |
| BTC-USD | 10 | 0.826 | 0.6032 | 1.0000 | No |
| ETH-USD | 1 | 3.195 | 0.0741 | 0.7410 | No |
| ETH-USD | 2 | 1.942 | 0.1438 | 1.0000 | No |
| ETH-USD | 3 | 2.065 | 0.1030 | 1.0000 | No |
| ETH-USD | 5 | 1.429 | 0.2109 | 1.0000 | No |
| ETH-USD | 10 | 0.900 | 0.5325 | 1.0000 | No |

**VAR robustness** (controls for BTC↔ETH correlation simultaneously):

| Asset | p (VAR Granger) | Significant (Bonferroni)? |
|-------|----------------|--------------------------|
| BTC-USD | 0.3754 | No |
| ETH-USD | 0.0724 | No |

**Verdict:** No Granger causality. Crypto returns add no predictive power beyond SPX's own history.

---

### Phase 7 — Regime Conditioning

Tests whether the null result hides a regime-specific signal. The data is split by two volatility proxies — VIX level and 30-day rolling SPX volatility — and the Phase 3 lag correlation is re-run in each half.

#### VIX split (threshold = 17.9900, high n=666, low n=664)

![VIX regime bands](figures/regime_bands_vix.png)

![VIX regime lag correlation](figures/regime_lag_correlation_vix.png)

**Verdict:** No signal in either regime — null result holds across market environments.

#### Rolling Vol split (threshold = 0.0083, high n=665, low n=665)

![Rolling Vol regime bands](figures/regime_bands_rolling_vol.png)

![Rolling Vol regime lag correlation](figures/regime_lag_correlation_rolling_vol.png)

**Verdict:** No signal in either regime — null result holds across market environments.

**Overall verdict:** Both regime definitions (VIX and rolling vol) agree. No regime-specific lead effect is detected. The null result from Phases 3 and 6 is robust to market volatility conditions.

---

## Limitations

- **Daily granularity only.** Any intraday lead-lag effect (minutes to hours) is invisible at this resolution.
- **Linear model only.** A logistic regression on raw returns is the simplest possible model. Non-linear relationships or volatility regime effects are not captured.
- **No macroeconomic controls.** If both crypto and equities respond to a common macro factor (e.g., Fed announcements), that confounds interpretation of any lead-lag signal.
- **Single test window.** The test set covers approximately 355 trading days (≈16 months). Conclusions from this window may not generalize.
- **Correlation ≠ causation.** Even if a lead effect existed, it would not establish that crypto *causes* equity moves.

---

## What I'd Do Next

1. **Intraday data** — test at 1h or 4h resolution; any informational lead likely operates on short horizons.
2. **Additional assets** — Baltic bank stocks, gold, oil; test whether crypto leads other risk-on assets.
3. **Regime-switching model** — a hidden Markov model to detect periods where the correlation structure changes endogenously, rather than conditioning on an exogenous threshold.
4. **Macroeconomic controls** — include Fed meeting dates, CPI releases; test whether any apparent lead is explained by shared macro exposure.

---

*Generated 2026-06-04. Data downloaded from Yahoo Finance via yfinance.*
