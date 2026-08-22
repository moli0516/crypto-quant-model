import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CryptoDataLoader:
    def __init__(self, processed_dir: str = "data/processed", interim_dir: str = "data/interim"):
        self.processed_dir = Path(processed_dir)
        self.interim_dir = Path(interim_dir)

    def load_dataset(self, horizon: int = 12, threshold: float = 0.0) -> tuple[pd.DataFrame, list[str]]:
        parquet_files = list(self.processed_dir.glob("*_features.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"❌ 在 {self.processed_dir} 找不到任何特徵 Parquet 檔案！請先執行 batch-feature。")

        dfs = []
        for file_path in parquet_files:
            symbol = file_path.name.replace("_features.parquet", "").upper()
            df = pd.read_parquet(file_path)
            
            if "symbol" not in df.columns:
                df["symbol"] = symbol

            # 確保 close 欄位存在（若特徵矩陣裡沒有，從 interim 清洗檔補回來）
            if "close" not in df.columns:
                clean_file = self.interim_dir / f"{symbol.lower()}_clean_record.csv"
                if clean_file.exists():
                    clean_df = pd.read_csv(clean_file, index_col="timestamp", parse_dates=True)
                    if "close" in clean_df.columns:
                        df["close"] = clean_df["close"]

            dfs.append(df)

        full_df = pd.concat(dfs, axis=0).sort_index()

        # 排除非特徵欄位
        exclude_cols = {"open", "high", "low", "close", "volume", "symbol", "label", "target", "future_return"}
        feature_cols = [col for col in full_df.columns if col not in exclude_cols and not col.startswith("target_")]

        # 動態生成標籤與未來報酬率
        if "label" not in full_df.columns or "future_return" not in full_df.columns:
            if "close" not in full_df.columns:
                raise KeyError("❌ 無法計算標籤：缺少 'close' 欄位，請檢查 interim 清洗資料夾是否完整！")
            
            full_df["future_return"] = full_df.groupby("symbol")["close"].shift(-horizon) / full_df["close"] - 1.0
            full_df["label"] = (full_df["future_return"] > threshold).astype(int)
            full_df = full_df.dropna(subset=["future_return", "close"])

        logger.info(f"✨ 多幣種總資料集載入完畢！總筆數: {len(full_df)} | 特徵數量: {len(feature_cols)}")
        return full_df, feature_cols