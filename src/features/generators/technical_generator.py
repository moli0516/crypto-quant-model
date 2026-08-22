import numpy as np
import pandas as pd
from typing import List

class TechnicalFeatureGenerator:
    """
    計算經典技術指標與報酬率的特徵生成器。
    執行順序建議：排在較前段 (EXECUTION_ORDER = 1)
    """
    EXECUTION_ORDER = 1

    def __init__(self, windows: List[int] = [1, 6, 24, 168]) -> None:
        self.windows = windows

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        
        # 1. 基礎對數報酬率 (Log Returns)
        feat["log_return_1h"] = np.log(df["close"] / df["close"].shift(1))

        # 2. 多時間窗口的滾動報酬率與波動率
        for w in self.windows:
            feat[f"return_{w}h"] = df["close"].pct_change(w)
            feat[f"volatility_{w}h"] = feat["log_return_1h"].rolling(window=w).std()

        # 3. 簡單移動平均線乖離率 (BIAS)
        sma_24 = df["close"].rolling(window=24).mean()
        feat["bias_24h"] = (df["close"] - sma_24) / sma_24

        return feat