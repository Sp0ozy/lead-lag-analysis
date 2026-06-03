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

## Limitations

- **Daily granularity only.** Any intraday lead-lag effect (minutes to hours) is invisible at this resolution.
- **Linear model only.** A logistic regression on raw returns is the simplest possible model. Non-linear relationships or volatility regime effects are not captured.
- **No macroeconomic controls.** If both crypto and equities respond to a common macro factor (e.g., Fed announcements), that confounds interpretation of any lead-lag signal.
- **Single test window.** The test set covers approximately 355 trading days (≈16 months). Conclusions from this window may not generalize.
- **Correlation ≠ causation.** Even if a lead effect existed, it would not establish that crypto *causes* equity moves.

---

## What I'd Do Next

1. **Intraday data** — test at 1h or 4h resolution; any informational lead likely operates on short horizons.
2. **Volatility regime conditioning** — split into high-VIX and low-VIX periods; the relationship may differ.
3. **Additional assets** — Baltic bank stocks, gold, oil; test whether crypto leads other risk-on assets.
4. **Granger causality** — a more formal test of predictive causality than simple cross-correlation.
5. **Regime-switching model** — a hidden Markov model to detect periods where the correlation structure changes.

---

*Generated 2026-06-04. Data downloaded from Yahoo Finance via yfinance.*
