# tests/test_data.py
import pandas as pd
import pytest

def test_download_vix_returns_series():
    """VIX download returns a non-empty Series aligned to equity dates."""
    from src.data import download_vix
    vix = download_vix()
    assert isinstance(vix, pd.Series)
    assert vix.name == "VIX"
    assert len(vix) > 500
    assert vix.isna().sum() == 0
    assert (vix > 0).all()
