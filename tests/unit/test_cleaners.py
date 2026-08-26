"""
BinanceOHLCVCleaner 單元測試
"""
import pytest
import pandas as pd
import numpy as np
from src.cleaners.binance_cleaner import BinanceOHLCVCleaner


def test_binance_cleaner_transform(sample_ohlcv):
    """驗證正常 OHLCV 數據清洗流程"""
    cleaner = BinanceOHLCVCleaner(fill_method="ffill")
    cleaned_df = cleaner(sample_ohlcv)

    assert not cleaned_df.empty
    assert isinstance(cleaned_df.index, pd.DatetimeIndex)
    assert cleaned_df["close"].dtype == np.float32


def test_binance_cleaner_deduplication():
    """驗證重複時間戳記過濾邏輯（預設 keep='last' 保留最後出現的一筆）"""
    dates = pd.date_range("2024-01-01", periods=3, freq="1h")
    df = pd.DataFrame(
        {
            "open": [10, 11, 12],
            "high": [12, 13, 14],
            "low": [9, 10, 11],
            "close": [11, 12, 13],  # 重複點 close 分別為 12 和 13
            "volume": [100, 100, 100],
        },
        index=[dates[0], dates[1], dates[1]],  # 重複的 dates[1]
    )

    cleaner = BinanceOHLCVCleaner()
    cleaned = cleaner(df)

    assert len(cleaned) == 2
    # 重複時間戳記使用 keep='last'，應保留最後一筆 close = 13
    assert cleaned.loc[dates[1]]["close"] == 13.0


def test_binance_cleaner_invalid_prices():
    """驗證無效價格（<=0）的過濾機制"""
    dates = pd.date_range("2024-01-01", periods=3, freq="1h")
    df = pd.DataFrame(
        {
            "open": [10, -5, 12],
            "high": [12, 0, 14],
            "low": [9, -10, 11],
            "close": [11, 12, 13],
            "volume": [100, 100, 100],
        },
        index=dates,
    )

    cleaner = BinanceOHLCVCleaner()
    cleaned = cleaner(df)

    assert len(cleaned) == 2
    assert dates[1] not in cleaned.index


def test_binance_cleaner_gap_filling():
    """驗證時間序列缺失空缺對齊與 ffill 填補機制"""
    dates = [
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2024-01-01 02:00:00"),  # 缺少 01:00:00
        pd.Timestamp("2024-01-01 03:00:00"),
        pd.Timestamp("2024-01-01 04:00:00"),
    ]
    df = pd.DataFrame(
        {
            "open": [10.0, 20.0, 30.0, 40.0],
            "high": [15.0, 25.0, 35.0, 45.0],
            "low": [5.0, 15.0, 25.0, 35.0],
            "close": [12.0, 22.0, 32.0, 42.0],
            "volume": [100.0, 200.0, 300.0, 400.0],
        },
        index=dates,
    )

    # 顯式指定 freq="1h"，使 reindex 能夠精準補齊缺失的時間點
    cleaner = BinanceOHLCVCleaner(fill_method="ffill", freq="1h")
    cleaned = cleaner(df)

    # 原始 4 筆 + 補齊 01:00 共 5 筆
    assert len(cleaned) == 5
    assert pd.Timestamp("2024-01-01 01:00:00") in cleaned.index
    # 01:00:00 應從 00:00:00 ffill 填補 close=12.0
    assert cleaned.loc["2024-01-01 01:00:00"]["close"] == 12.0