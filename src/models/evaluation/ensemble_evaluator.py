"""
src/models/evaluation/ensemble_evaluator.py
==============================================================================
Ensemble Walk-Forward Evaluator (XGBoost + LightGBM Blended Predictions)
完全對齊 walk_forward.py 介面與 ModelRegistry 封裝規範，解決 KeyError 問題。
==============================================================================
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class EnsembleWalkForwardEvaluator:
    """
    雙模型集成 (Ensemble: XGB + LGB) Walk-Forward 滾動交叉驗證器
    嚴格遵循 ModelRegistry 封裝介面，使用標準 'label' 欄位進行訓練。
    """

    def __init__(
        self,
        feature_cols: List[str],
        model_a: str = "xgb_classifier",
        model_b: str = "lgb_classifier",
        weights: List[float] = [0.5, 0.5],
        min_train_days: int = 180,
        step_days: int = 30,
        horizon: int = 12,
        threshold: float = 0.0
    ):
        self.feature_cols = feature_cols
        self.model_a_name = model_a
        self.model_b_name = model_b
        
        # 權重歸一化
        total_w = sum(weights)
        self.weights = [w / total_w for w in weights] if total_w > 0 else [0.5, 0.5]
        
        self.min_train_days = min_train_days
        self.step_days = step_days
        self.horizon = horizon
        self.threshold = threshold
        self.registry = ModelRegistry()

    def evaluate_and_blend(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        對齊 walk_forward.py 的執行流程，進行多幣種 Walk-Forward 預測融合。
        """
        logger.info("📈 啟動雙模型集成 (XGB + LGB) Walk-Forward 滾動交叉驗證...")

        # 1. 檢查訓練標籤 'label' 是否存在
        target_col = "label"
        if target_col not in dataset.columns:
            # 防禦性退化檢查：若不存在 label 則尋找 target_ret_Xh 欄位並重命名
            matching_targets = [c for c in dataset.columns if "target" in c or "label" in c]
            if matching_targets:
                target_col = matching_targets[0]
                logger.warning(f"⚠️ Dataset 缺失原生 'label' 欄位，自動對齊使用 [{target_col}] 作為訓練目標。")
            else:
                raise KeyError(f"❌ Dataset 缺少標籤欄位 'label'，無法訓練模型！現有欄位: {list(dataset.columns[:8])}")

        dataset = dataset.sort_index()
        unique_timestamps = np.sort(dataset.index.unique())
        
        start_date = unique_timestamps[0]
        end_date = unique_timestamps[-1]
        
        train_window_delta = pd.Timedelta(days=self.min_train_days)
        step_delta = pd.Timedelta(days=self.step_days)

        current_train_end = start_date + train_window_delta
        
        all_predictions = []
        fold_count = 0

        while current_train_end < end_date:
            test_end = current_train_end + step_delta
            
            train_mask = dataset.index < current_train_end
            test_mask = (dataset.index >= current_train_end) & (dataset.index < test_end)
            
            train_df = dataset[train_mask]
            test_df = dataset[test_mask]

            if len(train_df) == 0 or len(test_df) == 0:
                current_train_end = test_end
                continue

            # 2. 對齊 walk_forward.py 的模型訓練機制
            # 訓練 Model A (XGBoost)
            model_a = ModelRegistry.create(self.model_a_name)
            model_a.fit(
                train_df=train_df,
                feature_cols=self.feature_cols,
                target_col=target_col,
                verbose=False
            )
            prob_a = model_a.predict(test_df)

            # 訓練 Model B (LightGBM)
            model_b = ModelRegistry.create(self.model_b_name)
            model_b.fit(
                train_df=train_df,
                feature_cols=self.feature_cols,
                target_col=target_col,
                verbose=False
            )
            prob_b = model_b.predict(test_df)

            # 3. 機率融合 (Soft Voting)
            blended_prob = (prob_a * self.weights[0]) + (prob_b * self.weights[1])

            # 4. 構建此 Fold 輸出，完美對齊 backtest_engine 期待格式
            close_col = "close" if "close" in test_df.columns else "close_price"
            
            fold_pred_df = test_df[[close_col, target_col]].copy()
            if close_col != "close":
                fold_pred_df = fold_pred_df.rename(columns={close_col: "close"})
                
            fold_pred_df["pred_proba"] = blended_prob
            fold_pred_df["prob_a"] = prob_a
            fold_pred_df["prob_b"] = prob_b
            fold_pred_df["fold"] = fold_count + 1

            if "symbol" in test_df.columns:
                fold_pred_df["symbol"] = test_df["symbol"]

            all_predictions.append(fold_pred_df)
            
            fold_count += 1
            current_train_end = test_end

        if not all_predictions:
            raise ValueError("❌ Ensemble Walk-Forward 未能產生任何有效預測，請檢查時間參數與 Dataset！")

        combined_preds = pd.concat(all_predictions).sort_index()
        logger.info(f"✨ Ensemble Walk-Forward 驗證完畢！總折數: {fold_count} | 總預測點數: {len(combined_preds)}")
        
        return combined_preds