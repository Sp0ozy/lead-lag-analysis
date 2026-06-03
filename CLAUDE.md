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

## Claude Code Setup

### Permissions
Create `.claude/settings.json` so Claude Code doesn't prompt on every tool call:

```json
{
  "permissions": {
    "allow": [
      "Bash(python *)",
      "Bash(.venv/Scripts/python *)",
      "Bash(.venv/Scripts/pip *)",
      "Bash(pip *)",
      "Bash(git *)",
      "Bash(mkdir *)",
      "Bash(del *)"
    ]
  }
}
```

### Skills to invoke during implementation

| Skill | When to use |
|-------|-------------|
| `/superpowers:systematic-debugging` | When the pipeline errors or a phase check fails |
| `/code-review` | After all 5 phases are complete |
| `/verify` | To confirm figures and CSV outputs look correct |

The `context7` MCP server is available — if statsmodels, scikit-learn, or yfinance API questions come up, it fetches current docs rather than relying on training data.

### Working style
- All 5 phases are implemented in a **single Claude Code session** — phases are progress checkpoints, not restart points
- Press `#` at any point to have Claude update this file with anything non-obvious discovered during implementation

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

Commit once per phase, after all that phase's checks pass. Never commit a broken pipeline.

**Commit message format:**
```
phase1: data pipeline passes all checks — returns.csv ready
phase2: EDA figures saved, ADF stationarity confirmed
phase3: lag correlation results — no significant lead effect after Bonferroni
phase4: backtest complete, leakage check passed
phase5: README writeup complete
```

**What not to commit** (already in `.gitignore`):
- `data/raw/*.csv`, `data/processed/*.csv` — regenerable from pipeline
- `.venv/` — local environment
- `figures/*.png` — optional; commit final presentation figures only

---

## Coding Standards

- Functions have clear single responsibilities
- No hardcoded paths — use `pathlib.Path` and a single `ROOT` constant
- No hardcoded date strings outside `data.py` config
- All figures saved to `figures/` with descriptive filenames; `plt.show()` is never called in src/
- Return DataFrames/Series from functions; let the caller decide what to do with them
- Type hints on all public functions
