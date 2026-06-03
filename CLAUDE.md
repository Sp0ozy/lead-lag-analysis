# BTC/ETH → S&P 500 Lead-Lag Analysis

## Project Purpose

**Research question:** Does Bitcoin/Ethereum return at day *t* carry predictive information about S&P 500 returns at day *t+k* for k ∈ {1, 2, 3, 5, 10}?

**Deliverable:** A statistically defensible portfolio/interview project. A correctly-demonstrated null result is a **success**. A positive result caused by methodology artifacts is a **failure**. Optimize for truth, not apparent profitability.

---

## Quick Start

```bash
# First-time setup
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

# Run the full pipeline
python scripts/run_all.py

# Force fresh data download (delete cache, then re-run)
del data\raw\*.csv              # Windows
# rm data/raw/*.csv             # macOS/Linux
python scripts/run_all.py
```

---

## NON-NEGOTIABLE STATISTICAL RULES

These rules apply to every file, every function, every plot. There are no exceptions.

### 1. Work in log returns, not price levels
Prices are non-stationary and produce spurious correlation. All analysis uses log returns:
```
r_t = ln(P_t / P_{t-1})
```
Prices may appear in plots **for visual context only**, never as analysis inputs.

### 2. Calendar alignment is explicit and documented
- Crypto trades 24/7; equities do not.
- Resample to daily, align on the **equity trading calendar** (business days, US market holidays excluded).
- **Mapping rule:** The crypto close used for equity day *t* is the UTC midnight-to-midnight close of the *same calendar date* as the equity session. Document any deviation from this rule in PLAN.md's decision log.
- After alignment, verify date ranges match and NaN count is zero.

### 3. "Leads" means k > 0 strictly
- k=0 is **contemporaneous** correlation — not predictive.
- k>0 is the lead effect being tested.
- Never describe a k=0 result as evidence of predictive power.

### 4. Multiple testing requires correction
- Testing 5 lags × 2 assets = 10 hypotheses minimum.
- Apply **Bonferroni correction** (or equivalent) to any p-values reported as "significant."
- Results without correction must be labeled **exploratory only**.
- A single uncorrected p<0.05 among 10 tests is expected noise. Do not report it as a finding.

### 5. Chronological train/test split — no exceptions
- Train: data through 2024-12-31
- Test: data from 2025-01-01 onward
- **No shuffling.** No k-fold on time series. No fitting on test data. No parameter tuning that touches the test set.
- If a result on the test set looks good, look for leakage before claiming it's real.

### 6. Too-good results are bugs until proven otherwise
If a result looks surprising (high correlation, strong predictive accuracy), stop and audit:
- Check for off-by-one lag errors
- Check that test labels weren't computed from future data
- Verify the alignment mapping
- Only after ruling out bugs should a strong result be reported.

---

## Tech Stack

- **Python:** 3.11+
- **Core:** pandas, numpy
- **Stats:** scipy, statsmodels
- **ML:** scikit-learn (minimal — one simple model)
- **Data:** yfinance
- **Plots:** matplotlib
- **Environment:** virtualenv + pinned `requirements.txt`

## Data Sources

| Symbol  | Asset              | Via      |
|---------|--------------------|----------|
| BTC-USD | Bitcoin            | yfinance |
| ETH-USD | Ethereum           | yfinance |
| ^GSPC   | S&P 500 index      | yfinance |

- **Date range:** 2021-01-01 → present (today's date at time of download)
- **Field:** Adjusted close (daily)
- **Cache:** Raw downloads → `data/raw/<symbol>.csv`. Load from cache; re-download only if file is absent or stale.

---

## Repo Structure

```
btc-sp500/
├── CLAUDE.md              # This file
├── PLAN.md                # Phase checklist + decision log
├── README.md              # Audience-facing writeup (built up each phase)
├── requirements.txt       # Pinned deps
├── data/
│   ├── raw/               # Cached yfinance downloads (CSV, never commit large files)
│   └── processed/         # returns.csv and other derived outputs
├── src/
│   ├── data.py            # download, cache, align, compute returns
│   ├── analysis.py        # correlation, lag tests, statistical corrections
│   ├── model.py           # train/test split, simple classifier, backtest
│   └── plots.py           # all figure-generating functions
├── scripts/
│   └── run_all.py         # thin driver — calls src/ functions in sequence
└── figures/               # saved matplotlib outputs
```

**Code organization rule:** All reusable logic lives in `src/`. Scripts and notebooks are thin drivers that import from `src/` — they contain no analysis logic themselves.

---

## Scope Discipline

Do not expand scope until all 5 phases are complete end-to-end on the minimal version:
- No additional assets (Baltic bank tickers, etc.) until Phase 5 is done
- No intraday data
- No complex ML (LSTM, transformers, ensembles)
- No volatility signals beyond what Phase 3–4 require
- One simple, explainable model for Phase 4 (e.g. logistic regression on crypto return → equity direction)

**Rule:** A regression you can explain beats a black box you can't.

---

## Git Workflow

Commit after each meaningful unit of work — not after everything, not after every line.

**Commit cadence by phase:**
- Phase 1: one commit per completed function in `src/data.py`, one commit when `returns.csv` passes all checks
- Phase 2–5: one commit per completed phase (all figures saved + README section written)
- Never commit broken pipeline state — `scripts/run_all.py` must run clean before committing

**Commit message format:**
```
phase1: add download_raw() with yfinance caching
phase1: add align_to_equity_calendar() — maps UTC crypto close to equity date
phase1: returns.csv passes all Phase 1 checks
phase2: EDA figures saved, ADF stationarity confirmed
phase3: lag correlation results — no significant lead effect after Bonferroni
```

**What not to commit:**
- `data/raw/*.csv` and `data/processed/*.csv` — add to `.gitignore`
- `.venv/` — add to `.gitignore`
- `figures/*.png` — optional; commit only final presentation figures if desired

**Branch strategy (optional but recommended):**
- `main` — clean, phase-complete snapshots only
- `phase/1-data`, `phase/2-eda`, etc. — work branches; merge to main when phase checks pass

---

## Coding Standards

- Functions have clear single responsibilities
- No hardcoded paths — use `pathlib.Path` and a single `ROOT` constant
- No hardcoded date strings outside `data.py` config
- All figures saved to `figures/` with descriptive filenames; `plt.show()` is never called in src/
- Return DataFrames/Series from functions; let the caller decide what to do with them
- Type hints on all public functions
