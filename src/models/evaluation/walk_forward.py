import logging
import pandas as pd
import numpy as np
from src.models.registry import ModelRegistry


logger = logging.getLogger(__name__)

class WalkForwardEvaluator:
    def __init__(
        self, 
        feature_cols: list[str], 
        model_name: str = "xgb_classifier", 
        min_train_days: int = 180, 
        step_days: int = 30, 
        horizon: int = 12, 
        threshold: float = 0.0
    ):
        self.feature_cols = feature_cols
        self.model_name = model_name  # 確保這裡有正確接收與儲存
        self.min_train_days = min_train_days
        self.step_days = step_days
        self.horizon = horizon
        self.threshold = threshold
        self.registry = ModelRegistry()

    def evaluate(self, dataset: pd.DataFrame) -> dict:
        """
        執行基於時間軸的多幣種 Walk-Forward 滾動交叉驗證
        """
        logger.info("📈 啟動多幣種 Walk-Forward 滾動交叉驗證...")
        
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

            # 建立模型實例
            model = ModelRegistry.create(self.model_name)
            
            # 訓練模型：完整對接 XGBClassifierWrapper.fit 的參數
            model.fit(
                train_df=train_df,
                feature_cols=self.feature_cols,
                target_col="label",
                verbose=False
            )

            # 預測外樣本外機率：直接傳入 test_df，因為 wrapper 已經封裝好只取特徵欄位並回傳機率
            preds_proba = model.predict(test_df)

            # 記錄預測結果
            fold_pred_df = test_df[["close", "label"]].copy()
            fold_pred_df["pred_proba"] = preds_proba
            if "symbol" in test_df.columns:
                fold_pred_df["symbol"] = test_df["symbol"]

            all_predictions.append(fold_pred_df)
            
            fold_count += 1
            current_train_end = test_end

        if not all_predictions:
            raise ValueError("❌ Walk-Forward 未能產生任何有效預測，請檢查時間參數！")

        combined_preds = pd.concat(all_predictions).sort_index()

        metrics = {
            "total_folds": fold_count,
            "total_predictions": len(combined_preds)
        }

        logger.info(f"✨ Walk-Forward 驗證完畢！總折數: {fold_count} | 總預測點數: {len(combined_preds)}")
        return {
            "metrics": metrics,
            "predictions": combined_preds
        }