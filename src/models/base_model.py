import abc
import logging
from typing import List, Tuple, Any
import pandas as pd
import numpy as np
import joblib

logger = logging.getLogger(__name__)

class BaseModel(abc.ABC):
    """
    所有機器學習模型的抽象基底類別 (Abstract Base Class)
    """
    def __init__(self, model_params: dict = None):
        self.model_params = model_params or {}
        self.model = None
        self.feature_cols: List[str] = []

    @abc.abstractmethod
    def fit(self, train_df: pd.DataFrame, feature_cols: List[str], target_col: str, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        pass

    def save(self, filepath: str) -> None:
        try:
            joblib.dump(self, filepath)
            logger.info(f"✅ 模型已成功儲存至: {filepath}")
        except Exception as e:
            logger.error(f"❌ 模型儲存失敗 ({filepath}): {e}")
            raise e

    @classmethod
    def load(cls, filepath: str) -> Any:
        try:
            model_instance = joblib.load(filepath)
            logger.info(f"✅ 模型已成功從 {filepath} 載入")
            return model_instance
        except Exception as e:
            logger.error(f"❌ 模型載入失敗 ({filepath}): {e}")
            raise e