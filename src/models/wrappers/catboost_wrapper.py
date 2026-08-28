import logging
import os
from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.models.base_model import BaseModel
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

@ModelRegistry.register("cat_classifier")
class CatBoostClassifierWrapper(BaseModel):
    """
    基於 CatBoost Classifier 的加密貨幣二元分類模型封裝 (效能優化版)
    """
    def __init__(self, model_params: Optional[dict] = None):
        super().__init__(model_params)
        
        # 🛡️ 優化 CPU 資源分配：限制單一 CatBoost 實例佔用的最大核心數，避免與 Joblib 多進程相撞
        max_threads = min(4, os.cpu_count() or 4)

        default_params = {
            "iterations": 300,            # 由 500 調整為 300，維持表現同時提升訓練速度
            "learning_rate": 0.06,        # 配合 iterations 稍微調高學習率
            "depth": 6,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": 42,
            "verbose": False,
            "thread_count": max_threads   # 鎖定合理 CPU 核心數，解除鎖定效能瓶頸
        }
        if self.model_params:
            default_params.update(self.model_params)
            
        self.model_params = default_params
        self.model = CatBoostClassifier(**self.model_params)

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        eval_set: Optional[Tuple[pd.DataFrame, List[str], str]] = None,
        **kwargs,
    ) -> None:
        self.feature_cols = feature_cols
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        fit_kwargs = kwargs.copy()
        if "verbose" not in fit_kwargs:
            fit_kwargs["verbose"] = False

        fit_params = {}
        if eval_set is not None:
            val_df, val_feature_cols, val_target_col = eval_set
            X_val = val_df[val_feature_cols]
            y_val = val_df[val_target_col]
            fit_params["eval_set"] = (X_val, y_val)
            fit_params["early_stopping_rounds"] = 20  # 縮短早停檢查圈數以加速訓練

        logger.info(f"🚀 開始訓練 CatBoostClassifier | 特徵數: {len(feature_cols)} | 樣本數: {len(X_train)}")
        self.model.fit(X_train, y_train, **fit_params, **kwargs)
        logger.info("✅ CatBoostClassifier 訓練成功！")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("【錯誤】模型尚未訓練或載入！")
        X = df[self.feature_cols]
        return self.model.predict_proba(X)[:, 1]