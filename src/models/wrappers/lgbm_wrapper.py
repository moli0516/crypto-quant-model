import logging
from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.models.base_model import BaseModel
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

@ModelRegistry.register("lgb_classifier")
class LGBMClassifierWrapper(BaseModel):
    """
    基於 LightGBM Classifier 的加密貨幣二元分類模型封裝。
    """
    def __init__(self, model_params: Optional[dict] = None):
        super().__init__(model_params)
        
        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1
        }
        if self.model_params:
            default_params.update(self.model_params)
            
        self.model_params = default_params
        self.model = lgb.LGBMClassifier(**self.model_params)

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

        # 🛡️ 安全過濾：LightGBM fit() 不接受 verbose 參數，將其過濾以避免 TypeError
        fit_kwargs = kwargs.copy()
        verbose_flag = fit_kwargs.pop("verbose", False)

        fit_params = {}
        callbacks = []

        if eval_set is not None:
            val_df, val_feature_cols, val_target_col = eval_set
            X_val = val_df[val_feature_cols]
            y_val = val_df[val_target_col]
            fit_params["eval_set"] = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))

        if callbacks:
            fit_params["callbacks"] = callbacks

        logger.info(f"🚀 開始訓練 LGBMClassifier | 特徵數: {len(feature_cols)} | 樣本數: {len(X_train)}")
        self.model.fit(X_train, y_train, **fit_params, **fit_kwargs)
        logger.info("✅ LGBMClassifier 訓練成功！")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("【錯誤】模型尚未訓練或載入！")
        X = df[self.feature_cols]
        return self.model.predict_proba(X)[:, 1]