import logging
import pandas as pd
from src.features.base_target import BaseTarget

logger = logging.getLogger(__name__)

class BinaryReturnTarget(BaseTarget):
    """
    未來報酬率二元分類目標生成器。
    預測未來 H 小時後是否能獲得大於 threshold 的正報酬。
    """

    def __init__(self, horizon: int = 1, threshold: float = 0.0) -> None:
        self.horizon = horizon
        self.threshold = threshold
        logger.info(f"🎯 初始化 BinaryReturnTarget | 預測步長: {horizon}h | 門檻值: {threshold}")

    def generate_target(self, df: pd.DataFrame) -> pd.Series:
        # 計算未來 H 小時的價格變化
        # shift(-self.horizon) 將未來 H 小時的價格拉回到當前行
        future_close = df["close"].shift(-self.horizon)
        future_return = (future_close - df["close"]) / df["close"]

        # 轉換為二元標籤 (1: 漲幅超過門檻, 0: 未超過或下跌)
        target = (future_return > self.threshold).astype(int)

        # ⚠️ 注意：最後 H 筆資料因為沒有未來價格，會產生 NaN，必須將其設為 NaN 或在訓練時捨棄
        target.iloc[-self.horizon:] = pd.NA
        
        target.name = f"target_class_{self.horizon}h"
        return target