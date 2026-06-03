"""Data download, caching, alignment, and return computation."""

from __future__ import annotations

import pathlib
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

SYMBOLS = ["BTC-USD", "ETH-USD", "^GSPC"]
START_DATE = "2021-01-01"


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "")


def download_raw(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close from yfinance; load from cache if data is current.

    Cache is considered stale if its last date is more than 1 calendar day before
    the requested end date (allowing for the fact that today's data may not yet
    be published).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"{_safe_symbol(symbol)}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cache_last = pd.Timestamp(df.index.max()).date()
        end_date = pd.Timestamp(end).date()
        if (end_date - cache_last).days <= 1:
            return df
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, auto_adjust=True)[["Close"]]
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.index.name = "Date"
    df.columns = [symbol]
    df.to_csv(cache_path)
    return df


def load_prices() -> pd.DataFrame:
    """Load all symbols from cache; download missing ones."""
    end = date.today().isoformat()
    frames = [download_raw(sym, START_DATE, end) for sym in SYMBOLS]
    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    return prices


def align_to_equity_calendar(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Align all series to the equity trading calendar (^GSPC non-NaN dates).

    Mapping rule (per PLAN.md decision log):
      The crypto close used for equity day t is the UTC midnight-to-midnight
      close of the same calendar date as the equity session.
    """
    equity_dates = prices["^GSPC"].dropna().index
    aligned = prices.reindex(equity_dates)
    # Forward-fill any rare gaps (holidays where crypto traded but equity did not
    # create NaN in equity; this reindex already drops those via equity_dates).
    # Crypto may have NaN on equity holidays — shouldn't happen after reindex
    # to equity dates, but forward-fill as a safety net for data gaps.
    aligned = aligned.ffill()
    assert aligned.isna().sum().sum() == 0, "NaN values remain after alignment"
    return aligned


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns; drop the first row (NaN from diff)."""
    log_ret = np.log(prices).diff().dropna()
    return log_ret


def save_processed(returns: pd.DataFrame) -> pathlib.Path:
    """Write returns to data/processed/returns.csv."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "returns.csv"
    returns.to_csv(path)
    return path


def build_returns() -> pd.DataFrame:
    """Full pipeline: download → align → compute log returns → save."""
    prices = load_prices()
    prices_aligned = align_to_equity_calendar(prices)
    returns = compute_log_returns(prices_aligned)
    save_processed(returns)
    return returns


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
