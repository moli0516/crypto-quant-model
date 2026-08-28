import logging
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.evaluation.walk_forward import WalkForwardEvaluator
from src.models.evaluation.backtest_engine import SimpleStrategyBacktester

logger = logging.getLogger(__name__)

class EnsembleWalkForwardEvaluator:
    """
    雙模型集成 (Ensemble) Walk-Forward 評估器。
    透過組合兩種不同的模型 (如 XGBoost + LightGBM)，實現 Soft Voting 訊號融合。
    """
    def __init__(
        self, 
        feature_cols: list[str], 
        model_a: str = "xgb_classifier", 
        model_b: str = "lgb_classifier",
        weights: list[float] = [0.5, 0.5],
        min_train_days: int = 180, 
        step_days: int = 30, 
        horizon: int = 12, 
        threshold: float = 0.0
    ):
        self.feature_cols = feature_cols
        self.model_a = model_a
        self.model_b = model_b
        self.weights = weights
        self.horizon = horizon
        
        # 實例化兩個獨立的 WalkForwardEvaluator (完全沿用固有程式碼)
        self.evaluator_a = WalkForwardEvaluator(
            feature_cols=feature_cols, model_name=model_a, 
            min_train_days=min_train_days, step_days=step_days, 
            horizon=horizon, threshold=threshold
        )
        self.evaluator_b = WalkForwardEvaluator(
            feature_cols=feature_cols, model_name=model_b, 
            min_train_days=min_train_days, step_days=step_days, 
            horizon=horizon, threshold=threshold
        )

    def evaluate_and_blend(self, dataset: pd.DataFrame, cache_dir: str = "local-logs") -> pd.DataFrame:
        """
        對兩個模型分別執行 Walk-Forward 驗證（支援 Parquet 快取），並進行對齊與加權融合。
        """
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        file_a = cache_path / f"wf_preds_cache_{self.model_a}_h{self.horizon}.parquet"
        file_b = cache_path / f"wf_preds_cache_{self.model_b}_h{self.horizon}.parquet"

        # 1. 模型 A 快取機制
        if file_a.exists():
            logger.info(f"⚡ 載入模型 A ({self.model_a}) 快取: {file_a.name}")
            preds_a = pd.read_parquet(file_a)
        else:
            logger.info(f"🤖 執行模型 A ({self.model_a}) Walk-Forward 訓練...")
            preds_a = self.evaluator_a.evaluate(dataset)["predictions"]
            preds_a.to_parquet(file_a)

        # 2. 模型 B 快取機制
        if file_b.exists():
            logger.info(f"⚡ 載入模型 B ({self.model_b}) 快取: {file_b.name}")
            preds_b = pd.read_parquet(file_b)
        else:
            logger.info(f"🤖 執行模型 B ({self.model_b}) Walk-Forward 訓練...")
            preds_b = self.evaluator_b.evaluate(dataset)["predictions"]
            preds_b.to_parquet(file_b)

        # 3. 雙模型預測結果對齊與 Soft Voting
        logger.info(f"🔀 正在融合雙模型訊號 ({self.model_a} * {self.weights[0]} + {self.model_b} * {self.weights[1]})...")
        
        merged = pd.merge(
            preds_a.reset_index(),
            preds_b.reset_index()[["timestamp", "symbol", "pred_proba"]],
            on=["timestamp", "symbol"],
            suffixes=("_a", "_b")
        ).set_index("timestamp")

        blended_df = merged[["close", "label", "symbol"]].copy()
        blended_df["pred_proba"] = (merged["pred_proba_a"] * self.weights[0]) + (merged["pred_proba_b"] * self.weights[1])
        
        return blended_df