import logging
import pandas as pd
from src.cleaners.base_cleaner import BaseCleaner

logger = logging.getLogger(__name__)

class BinanceOHLCVCleaner(BaseCleaner):
    """
    針對幣安 OHLCV 歷史資料的具體清洗器。
    """

    def __init__(self, fill_method: str = "ffill") -> None:
        super().__init__()
        self.fill_method = fill_method

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 建立副本，避免修改到原始 DataFrame
        cleaned = df.copy()

        # 1. 確保 Index 為 Datetime 格式且按時間排序
        if not isinstance(cleaned.index, pd.DatetimeIndex):
            cleaned.index = pd.to_datetime(cleaned.index)
        cleaned = cleaned.sort_index()

        # 2. 移除重複的時間戳記（保留最後一筆）
        initial_count = len(cleaned)
        cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
        if len(cleaned) < initial_count:
            logger.warning(f"⚠️ 偵測並移除 {initial_count - len(cleaned)} 筆重複的時間戳記。")

        # 3. 檢查並處理價格異常（例如價格 <= 0）
        invalid_price_mask = (cleaned["low"] <= 0) | (cleaned["high"] <= 0) | (cleaned["close"] <= 0)
        if invalid_price_mask.sum() > 0:
            logger.warning(f"⚠️ 發現 {invalid_price_mask.sum()} 筆價格小於或等於 0 的異常數據，將予以過濾。")
            cleaned = cleaned[~invalid_price_mask]

        # 4. 確保時間序列連續性 (針對 1h 等固定頻率進行對齊與空缺填補)
        # 偵測最常見的時間間隔
        inferred_freq = pd.infer_freq(cleaned.index[:100])
        if inferred_freq:
            cleaned = cleaned.asfreq(inferred_freq)
            
            # 針對缺失的時間點進行填補
            missing_count = cleaned["close"].isna().sum()
            if missing_count > 0:
                logger.warning(f"⚠️ 偵測到 {missing_count} 個時間序列空缺，使用 '{self.fill_method}' 進行填補。")
                if self.fill_method == "ffill":
                    cleaned = cleaned.ffill()
                elif self.fill_method == "bfill":
                    cleaned = cleaned.bfill()
                else:
                    cleaned = cleaned.fillna(0)

        # 5. 確保所有數值型態皆為 float32
        numeric_cols = cleaned.select_dtypes(include=["number"]).columns
        cleaned[numeric_cols] = cleaned[numeric_cols].astype("float32")

        return cleaned