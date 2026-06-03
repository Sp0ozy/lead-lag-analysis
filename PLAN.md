# Project Plan: BTC/ETH → S&P 500 Lead-Lag Analysis

> **Implementation:** All 5 phases are completed in a single Claude Code session. The checklist below tracks in-session progress — phases are checkpoints, not session restart points.
> **Rule:** Do not start a phase before the prior phase's checks all pass.
> **Rule:** All decisions that are non-obvious or deviate from the spec go in the Decision Log below.

---

## Phase 1 — Data & Setup

**Goal:** Working data pipeline; clean, aligned log-return series ready for analysis.

### Tasks
- [ ] Create virtual environment (`python -m venv .venv`)
- [ ] Write `requirements.txt` with pinned versions
- [ ] Create directory structure (`data/raw/`, `data/processed/`, `src/`, `scripts/`, `figures/`)
- [ ] Write `src/data.py`:
  - [ ] `download_raw(symbol, start, end)` → downloads from yfinance, saves to `data/raw/<symbol>.csv`, loads from cache if present
  - [ ] `load_prices()` → loads all three symbols from cache, returns raw DataFrame
  - [ ] `align_to_equity_calendar(prices_df)` → resamples crypto to daily, aligns to equity trading days, documents mapping rule
  - [ ] `compute_log_returns(prices_df)` → returns DataFrame of log returns, drops first row (NaN from diff)
  - [ ] `save_processed(returns_df)` → writes `data/processed/returns.csv`
- [ ] Write `scripts/run_all.py` as a thin driver that calls the above in sequence

### Phase 1 Checks (all must pass before Phase 2)
- [ ] `returns.csv` contains zero NaN values after alignment
- [ ] Date range for all three assets matches (same start/end after alignment)
- [ ] Return stats are sane:
  - BTC daily log returns: mean near 0, std roughly 3–5%
  - ETH daily log returns: mean near 0, std roughly 4–6%
  - SPX daily log returns: mean near 0, std roughly 0.8–1.2%
- [ ] No future data in returns (last date ≤ today)
- [ ] Crypto and equity calendar alignment is documented (see Decision Log)

---

## Phase 2 — Exploratory Data Analysis

**Goal:** Visual and statistical understanding of the data. Findings written to README.

### Tasks
- [ ] Write `src/plots.py`:
  - [ ] `plot_prices(prices_df)` → normalized price chart (context only; labeled as not for analysis)
  - [ ] `plot_returns(returns_df)` → time series of daily log returns, all assets
  - [ ] `plot_rolling_volatility(returns_df, window=30)` → 30-day rolling std
  - [ ] `plot_correlation_matrix(returns_df)` → contemporaneous correlation heatmap (k=0)
- [ ] Write `src/analysis.py`:
  - [ ] `adf_test(series)` → runs ADF stationarity test, returns dict with statistic, p-value, lags, conclusion
  - [ ] `run_eda_stats(returns_df)` → runs ADF on each series, prints summary
- [ ] Update `scripts/run_all.py` to call EDA functions and save figures
- [ ] Write Phase 2 findings section in README (what the data looks like, stationarity confirmed)

### Phase 2 Checks
- [ ] All figures saved to `figures/` with descriptive names
- [ ] ADF tests confirm returns are stationary (p < 0.05) for all three series
- [ ] README updated with EDA findings

---

## Phase 3 — Lagged Cross-Correlation (Core Test)

**Goal:** Answer the research question with statistical rigor.

### Tasks
- [ ] Extend `src/analysis.py`:
  - [ ] `lagged_correlation(crypto_series, equity_series, lags)` → for each lag k, computes Pearson r and p-value of `crypto_t` vs `equity_{t+k}`; returns DataFrame
  - [ ] `apply_bonferroni(results_df)` → adds Bonferroni-corrected p-values and significance flags
  - [ ] `summarize_lead_lag(results_df)` → prints/returns plain-English verdict
- [ ] Extend `src/plots.py`:
  - [ ] `plot_lag_correlation(results_df)` → lag (x) vs correlation (y) with error bars or p-value color coding; corrected threshold marked
- [ ] Run for BTC→SPX and ETH→SPX for k ∈ {1, 2, 3, 5, 10}
- [ ] Save figures and write Phase 3 section in README with a plain-language verdict

### Phase 3 Checks
- [ ] Results table includes raw p-values AND Bonferroni-corrected p-values
- [ ] README states clearly: "evidence of lead effect" or "no evidence of lead effect (noise consistent with chance)"
- [ ] Any nominally significant uncorrected result is labeled exploratory and placed in context

---

## Phase 4 — Model & Backtest

**Goal:** One interpretable model, evaluated on out-of-sample data only.

### Tasks
- [ ] Write `src/model.py`:
  - [ ] `make_features(returns_df)` → creates feature matrix: `[btc_t, eth_t]` predicting `spx_{t+1}_direction`; labels are sign of next-day SPX return; **strictly no future data in features**
  - [ ] `chronological_split(X, y, split_date)` → splits on date, no shuffling
  - [ ] `train_model(X_train, y_train)` → fits logistic regression (or equivalent simple model)
  - [ ] `backtest(model, X_test, y_test)` → reports directional accuracy, F1, comparison to buy-and-hold baseline
  - [ ] `check_for_leakage(X, y)` → sanity check: shuffle y and confirm accuracy ≈ 50%
- [ ] Update `scripts/run_all.py`
- [ ] Write Phase 4 section in README

### Phase 4 Checks
- [ ] Train set: data through 2024-12-31 only
- [ ] Test set: data from 2025-01-01 only — never used during training or tuning
- [ ] Leakage check passes (shuffled labels → ~50% accuracy)
- [ ] Results reported vs. naive baseline (predict "up" every day)

---

## Phase 5 — Writeup

**Goal:** README is a complete, standalone portfolio document.

### Tasks
- [ ] Complete README sections:
  - [ ] Research question and motivation
  - [ ] Data description (sources, dates, alignment approach)
  - [ ] Methodology (returns, lag definition, multiple-testing correction, train/test split)
  - [ ] Results with figures (EDA, lag correlations, model performance)
  - [ ] Honest limitations ("what this analysis cannot tell us")
  - [ ] "What I'd do next" (Baltic tickers, intraday, volatility regime conditioning, etc.)
- [ ] Final review: no claims exceed what the statistics support
- [ ] All figures render correctly in README

### Phase 5 Checks
- [ ] README reads end-to-end without assuming the reader has seen the code
- [ ] Every figure is referenced in the text
- [ ] Limitations section is honest, not defensive

---

## Decision Log

*Record non-obvious choices here as they arise, so future readers understand why the code is the way it is.*

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06-03 | Crypto-to-equity calendar mapping: use same calendar-date UTC close for crypto as the equity session date | Simplest defensible rule; avoids ambiguity about overnight crypto moves. If equity session is 2021-03-15, we use the BTC/ETH daily close for 2021-03-15 UTC. Documented here per CLAUDE.md requirement. |
| 2026-06-03 | Train cutoff: 2024-12-31; test start: 2025-01-01 | Gives ~4 years of training data and ~6 months of test data (at time of writing). Clean calendar-year split is easy to explain and audit. |
| 2026-06-03 | Lags: {1, 2, 3, 5, 10} | Covers next-day, short-week, and two-week horizons. Avoids data dredging over too many lags while sampling the plausible predictive window. |
| 2026-06-03 | Model: logistic regression on [btc_t, eth_t] → spx direction_{t+1} | Maximally interpretable; coefficients directly answer "does crypto return predict equity direction." A black box would obscure the answer. |
