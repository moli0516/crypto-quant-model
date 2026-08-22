import logging
from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

from src.models.base_model import BaseModel
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

@ModelRegistry.register("xgb_classifier")
class XGBClassifierWrapper(BaseModel):
    """
    基於 XGBoost Classifier 的加密貨幣二元分類模型封裝。
    """
    def __init__(self, model_params: Optional[dict] = None):
        super().__init__(model_params)
        
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,
            "learning_rate": 0.03,
            "n_estimators": 100,
            "random_state": 42
        }
        if self.model_params:
            default_params.update(self.model_params)
            
        self.model_params = default_params
        self.model = XGBClassifier(**self.model_params)

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

        fit_params = {}
        if eval_set is not None:
            val_df, val_feature_cols, val_target_col = eval_set
            X_val = val_df[val_feature_cols]
            y_val = val_df[val_target_col]
            fit_params["eval_set"] = [(X_val, y_val)]
            if "verbose" not in kwargs:
                kwargs["verbose"] = False

        logger.info(f"🚀 開始訓練 XGBClassifier | 特徵數: {len(feature_cols)} | 樣本數: {len(X_train)}")
        self.model.fit(X_train, y_train, **fit_params, **kwargs)
        logger.info("✅ XGBClassifier 訓練成功！")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("【錯誤】模型尚未訓練或載入！")
        X = df[self.feature_cols]
        # 回傳預測為正類（漲）的機率
        return self.model.predict_proba(X)[:, 1]