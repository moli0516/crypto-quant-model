"""
LeakageGuard 防洩漏與特徵驗證單元測試
"""
import pytest
import pandas as pd
import numpy as np
from src.features.utils.leak_guard import LeakageGuard


def test_check_future_leakage_warning():
    """驗證當特徵與 Target 相關係數過高 (>= 0.90) 時發出 Warning"""
    target = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    leaked_feature = target * 1.0  # 完全洩漏

    df = pd.DataFrame({"feat_leaked": leaked_feature, "target": target})

    with pytest.warns(UserWarning, match="LEAKAGE WARNING"):
        corr = LeakageGuard.check_future_leakage(
            df=df, feature_col="feat_leaked", target_col="target", threshold=0.90
        )
    assert abs(corr - 1.0) < 1e-5


def test_assert_no_null_keys_pass_and_fail():
    """驗證 Primary Keys 缺失值檢查"""
    valid_df = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT"], "val": [1, 2]})
    invalid_df = pd.DataFrame({"symbol": ["BTCUSDT", None], "val": [1, 2]})

    # 正常情況
    LeakageGuard.assert_no_null_keys(valid_df, ["symbol"])

    # 包含 Null 應丟出 AssertionError
    with pytest.raises(AssertionError, match="包含 1 個 Null/NaN 缺失值"):
        LeakageGuard.assert_no_null_keys(invalid_df, ["symbol"])


def test_validate_feature_dataframe():
    """驗證特徵矩陣整體規格檢查"""
    df = pd.DataFrame({"symbol": ["BTCUSDT"], "feat_1": [0.5]})
    assert LeakageGuard.validate_feature_dataframe(df, required_keys=["symbol"]) is True

    empty_df = pd.DataFrame(columns=["symbol", "feat_1"])
    with pytest.raises(AssertionError, match="筆數為 0"):
        LeakageGuard.validate_feature_dataframe(empty_df, required_keys=["symbol"])