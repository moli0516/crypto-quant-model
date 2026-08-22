import logging
import pandas as pd
from typing import List, Any
from src.features.utils.leak_guard import LeakageGuard
from src.features.generators import load_all_generators

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """
    特徵工程總管線。
    動態載入所有生成器，依據 EXECUTION_ORDER 排序並依序執行。
    """

    def __init__(self, generators: List[Any] = None, **generator_kwargs) -> None:
        # 若未指定生成器，則透過動態掃描 generators/ 目錄載入
        self.generators = generators if generators is not None else load_all_generators(**generator_kwargs)
        logger.info(f"🚀 FeaturePipeline 初始化成功，共載入 {len(self.generators)} 個生成器。")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        依序執行所有特徵生成器，並將結果合併。
        """
        logger.info("⚙️ 開始執行 Feature Pipeline 轉換...")
        feature_dfs = []

        for gen in self.generators:
            gen_name = gen.__class__.__name__
            order = getattr(gen, "EXECUTION_ORDER", 500)
            logger.info(f"🔄 執行生成器: {gen_name} (Order: {order})")
            
            feat_df = gen.generate(df)
            feature_dfs.append(feat_df)

        # 合併所有特徵
        if not feature_dfs:
            return pd.DataFrame(index=df.index)

        combined_features = pd.concat(feature_dfs, axis=1)

        # 🛡️ 驗證特徵與原始資料的長度一致性
        assert len(combined_features) == len(df), "❌ 產出的特徵矩陣長度與原始資料不符！"
        
        # 檢查並處理 NaNs
        combined_features = combined_features.bfill().fillna(0)

        logger.info(f"✨ Feature Pipeline 執行完畢 | 總特徵維度: {combined_features.shape[1]} 欄")
        return combined_features