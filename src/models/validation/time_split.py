import logging
from typing import Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TimeSeriesSplitter:
    """
    時間序列資料切分工具。
    專門用來依據日期進行切分，確保訓練集在過去、驗證集在未來，防範 Data Leakage。
    """
    def __init__(self, date_col: str = "date"):
        self.date_col = date_col

    def split_by_days(
        self, 
        df: pd.DataFrame, 
        val_days: int = 30
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        以最後一天往前推指定天數作為驗證集，其餘為訓練集。
        """
        if self.date_col not in df.columns:
            raise ValueError(f"【錯誤】DataFrame 中找不到指定的日期欄位: '{self.date_col}'")

        df_sorted = df.copy()
        df_sorted[self.date_col] = pd.to_datetime(df_sorted[self.date_col])
        df_sorted = df_sorted.sort_index().reset_index() # 確保時間順序

        max_date = df_sorted[self.date_col].max()
        split_date = max_date - pd.Timedelta(days=val_days)

        logger.info(f"📅 資料集日期範圍: {df_sorted[self.date_col].min().strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
        logger.info(f"✂️ 切分點: 驗證集為最近 {val_days} 天 (大於 {split_date.strftime('%Y-%m-%d')})")

        # 重新設定 Index 為 timestamp
        df_sorted.set_index("timestamp", inplace=True)
        
        train_df = df_sorted[df_sorted[self.date_col] <= split_date].copy()
        val_df = df_sorted[df_sorted[self.date_col] > split_date].copy()

        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError("【錯誤】切分後的訓練集或驗證集為空！請檢查資料量或 val_days 設定。")

        logger.info(f"📊 切分完成 | 訓練集樣本數: {len(train_df)} | 驗證集樣本數: {len(val_df)}")
        return train_df, val_df