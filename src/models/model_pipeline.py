import logging
import os
import pandas as pd
from src.models.data_loader import CryptoDataLoader
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

class ModelPipeline:
    def __init__(self):
        self.loader = CryptoDataLoader()
        self.registry = ModelRegistry()

    def run_train_pipeline(self, model_name: str = "xgb_classifier", val_days: int = 30, horizon: int = 12, threshold: float = 0.0):
        """
        執行多幣種全域模型訓練
        """
        logger.info("🚀 開始執行多幣種模型訓練管線...")
        
        # 1. 載入多幣種 Parquet 總表與特徵欄位
        dataset, feature_cols = self.loader.load_dataset(horizon=horizon, threshold=threshold)
        
        if dataset.empty:
            raise ValueError("❌ 訓練資料集為空，請確認 Parquet 檔案是否存在！")

        # 2. 依據時間進行訓練集與驗證集切分 (Time-based split)
        max_date = dataset.index.max()
        val_start_date = max_date - pd.Timedelta(days=val_days)
        
        train_df = dataset[dataset.index < val_start_date]
        val_df = dataset[dataset.index >= val_start_date]

        logger.info(f"📊 訓練集樣本數: {len(train_df)} | 驗證集樣本數: {len(val_df)}")

        # 3. 取得模型實例並訓練 (修正為 ModelRegistry.create)
        model = ModelRegistry.create(model_name)
        
        # 對接 XGBClassifierWrapper.fit 的正確簽章
        model.fit(
            train_df=train_df,
            feature_cols=feature_cols,
            target_col="label",
            eval_set=(val_df, feature_cols, "label"),
            verbose=False
        )

        # 4. 計算驗證集準確率或指標
        preds_proba = model.predict(val_df)
        
        # 將機率轉換為 0 或 1 的硬標籤以計算分類指標
        val_preds = (preds_proba > 0.5).astype(int) 
        y_val = val_df["label"]
        
        from sklearn.metrics import accuracy_score, precision_score
        metrics = {
            "accuracy": float(accuracy_score(y_val, val_preds)),
            "precision": float(precision_score(y_val, val_preds, zero_division=0))
        }

        logger.info(f"✨ 模型訓練完成！驗證指標: {metrics}")
        
        # 5. 儲存模型至實體檔案供實盤/推論載入
        os.makedirs("models", exist_ok=True)
        model.save("models/best_xgb_model.pkl")

        return model, metrics