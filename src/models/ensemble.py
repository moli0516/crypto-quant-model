"""
src/models/ensemble.py
==============================================================================
Ensemble Prediction Engine (XGBoost + LightGBM Soft Voting)
==============================================================================
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional

# 🎯 統一從 src.config 讀取配置
from src.config import MODEL_PATHS, ENSEMBLE_WEIGHTS

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    雙模型集成推論引擎 (Soft Voting)
    防禦性設計：自動處理欄位對齊、缺失值 (NaNs) 及權重歸一化。
    """

    def __init__(
        self,
        model_paths: Optional[Dict[str, str]] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.model_paths = model_paths or MODEL_PATHS
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.models: Dict[str, object] = {}

        self._validate_and_normalize_weights()
        self._load_models()

    def _validate_and_normalize_weights(self) -> None:
        total_weight = sum(self.weights.values())
        if not abs(total_weight - 1.0) < 1e-5:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def _load_models(self) -> None:
        for name, path in self.model_paths.items():
            if not os.path.exists(path):
                logger.error(f"❌ 找不到模型檔案 [{name}]: {path}")
                raise FileNotFoundError(f"Model file not found: {path}")
            
            try:
                self.models[name] = joblib.load(path)
                logger.info(f"✅ 成功載入模型 [{name}] (Weight: {self.weights[name]:.2f}) 來自: {path}")
            except Exception as e:
                logger.error(f"❌ 載入模型 [{name}] 失敗: {e}")
                raise e

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        if features_df.empty:
            logger.warning("⚠️ 傳入特徵矩陣為空，回傳空預測結果。")
            return np.array([])

        ignore_cols = ["timestamp", "datetime", "symbol", "close_price", "close", "dt"]
        feature_cols = [c for c in features_df.columns if c not in ignore_cols]
        X = features_df[feature_cols].copy()

        if X.isna().sum().sum() > 0:
            X = X.fillna(0.0)

        weighted_probs = np.zeros(len(X))

        for name, model in self.models.items():
            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X)[:, 1]
                else:
                    prob = model.predict(X)
                
                weighted_probs += prob * self.weights[name]
            except Exception as e:
                logger.error(f"❌ 模型 [{name}] 推論過程發生錯誤: {e}")
                raise e

        return weighted_probs