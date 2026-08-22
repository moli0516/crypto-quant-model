import abc
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class BaseCleaner(abc.ABC):
    """
    資料清洗器的抽象基底類別。
    規範所有清洗管線必須實作 transform 方法，接收並回傳 pd.DataFrame。
    """

    def __init__(self) -> None:
        logger.info(f"🚀 初始化清洗模組: {self.__class__.__name__}")

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [介面合約] 執行資料清洗的核心方法。
        """
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        支援以函式化風格直接調用清洗器 (e.g., cleaner(df))。
        """
        if df.empty:
            logger.warning("⚠️ 傳入的 DataFrame 為空，跳過清洗流程。")
            return df
            
        original_len = len(df)
        cleaned_df = self.transform(df)
        logger.info(f"✨ 清洗完畢 | 原始筆數: {original_len} -> 清洗後筆數: {len(cleaned_df)}")
        return cleaned_df