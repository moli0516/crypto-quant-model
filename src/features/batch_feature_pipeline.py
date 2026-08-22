import logging
import pandas as pd
from pathlib import Path
from src.features.feature_pipeline import FeaturePipeline

logger = logging.getLogger(__name__)

def run_batch_feature_engineering(input_dir: str = "data/interim", output_dir: str = "data/processed"):
    """
    自動掃描 data/interim/ 底下所有 *_clean_record.csv，
    逐一執行 FeaturePipeline 特徵工程，並以 Parquet 格式高效儲存。
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(in_path.glob("*_clean_record.csv"))
    if not csv_files:
        logger.warning(f"⚠️ 在 {input_dir} 找不到任何清洗後的資料檔案！")
        return

    logger.info(f"⚙️ 開始為 {len(csv_files)} 個幣種進行特徵工程與 Parquet 轉換...")
    pipeline = FeaturePipeline()

    for file_path in csv_files:
        symbol_name = file_path.name.replace("_clean_record.csv", "").upper()
        logger.info(f"📊 正在處理幣種特徵: {symbol_name}")

        try:
            # 1. 讀取清洗後的 CSV
            df = pd.read_csv(file_path, index_col="timestamp", parse_dates=True)
            if df.empty:
                logger.warning(f"⚠️ 檔案 {file_path.name} 內容為空，跳過。")
                continue

            # 2. 執行特徵工程
            feature_df = pipeline.fit_transform(df)

            # 3. 輸出為高效的 Parquet 格式
            output_file = out_path / f"{symbol_name}_features.parquet"
            feature_df.to_parquet(output_file, compression="snappy")
            
            logger.info(f"✅ {symbol_name} 特徵矩陣已儲存至: {output_file} | 維度: {feature_df.shape}")

        except Exception as e:
            logger.error(f"❌ 處理 {symbol_name} 特徵時發生錯誤: {e}")

    logger.info("🎉 所有幣種的特徵工程與 Parquet 轉檔全部完成！")