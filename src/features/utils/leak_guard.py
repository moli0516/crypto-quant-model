import warnings
from typing import List
import numpy as np
import pandas as pd


class LeakageGuard:
    """自動化檢查產出的 Feature DataFrame 是否存在 Data Leakage 或異常值。"""

    @staticmethod
    def check_future_leakage(
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        threshold: float = 0.90,
    ) -> float:
        """檢查特徵與 Target 的相關性，若絕對值過高則自動發出警告。"""
        valid_df = df[[feature_col, target_col]].dropna()
        if len(valid_df) < 2:
            return 0.0

        corr = valid_df[feature_col].corr(valid_df[target_col])

        # 處理 Pandas corr() 回傳 NaN 的情況
        if pd.isna(corr):
            return 0.0

        corr_val = float(corr)

        if abs(corr_val) >= threshold:
            warnings.warn(
                f"🚨 [LEAKAGE WARNING] 特徵 `{feature_col}` 與 Target `{target_col}` 的相關係數高達 {corr_val:.4f}！請檢查是否漏寫 `.shift(1)`！",
                UserWarning,
            )
        return corr_val

    @staticmethod
    def assert_no_null_keys(df: pd.DataFrame, key_cols: List[str]):
        """確保 Primary Keys 完全沒有缺失值。"""
        for col in key_cols:
            null_count = df[col].isnull().sum()
            assert (
                null_count == 0
            ), f"❌ Key 欄位 `{col}` 包含 {null_count} 個 Null/NaN 缺失值！"

    @staticmethod
    def validate_feature_dataframe(
        df: pd.DataFrame, required_keys: List[str]
    ) -> bool:
        """檢驗生成後的 Feature DataFrame 是否符合管道規格。"""
        LeakageGuard.assert_no_null_keys(df, required_keys)
        assert len(df) > 0, "❌ 產出的 Feature DataFrame 筆數為 0！"
        return True