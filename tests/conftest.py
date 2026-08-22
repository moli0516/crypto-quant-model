"""Pytest 共用 Fixtures（模擬數據）"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_ohlcv():
    """產生簡單的 OHLCV 模擬資料"""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 20
    volume = np.random.randint(10, 1000, size=n).astype(float)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
