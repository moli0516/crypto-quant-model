import logging
import os
from datetime import datetime
from typing import List
import pandas as pd
from src.collectors.binance.rest_client import BinanceAsyncRESTClient
from src.cleaners.binance_cleaner import BinanceOHLCVCleaner
from src.features.feature_pipeline import FeaturePipeline

logger = logging.getLogger(__name__)

class LiveDataPipeline:
    """即時資料與特徵管線"""
    def __init__(self, symbols: List[str], timeframe: str = "1h", lookback_limit: int = 200):
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback_limit = lookback_limit 
        self.cleaner = BinanceOHLCVCleaner(fill_method="ffill")
        self.feature_pipeline = FeaturePipeline()

    async def fetch_and_process(self) -> pd.DataFrame:
        """非同步抓取所有幣種最新資料，並產出最新一筆特徵矩陣"""
        all_latest_features = []

        async with BinanceAsyncRESTClient() as client:
            for symbol in self.symbols:
                try:
                    # 1. 抓取最新資料
                    raw_df = await client.fetch_historical_ohlcv(
                        symbol=symbol, timeframe=self.timeframe, limit=self.lookback_limit
                    )
                    
                    if raw_df.empty:
                        logger.warning(f"⚠️ {symbol} 抓取失敗")
                        continue

                    if not raw_df.empty:
                        raw_df = raw_df.iloc[:-1]

                    # 2. 清洗資料
                    cleaned_df = self.cleaner(raw_df)

                    # 3. 生成特徵
                    feature_df = self.feature_pipeline.fit_transform(cleaned_df)
                    
                    # 4. 只取最新（最後一筆）未完成或剛完成的 K 線特徵
                    latest_feature = feature_df.iloc[[-1]].copy()
                    latest_feature["symbol"] = symbol
                    latest_feature["close"] = cleaned_df["close"].iloc[-1]
                    
                    all_latest_features.append(latest_feature)
                    
                except Exception as e:
                    logger.error(f"❌ 處理 {symbol} 時發生錯誤: {e}")

        if not all_latest_features:
            return pd.DataFrame()
            
        return pd.concat(all_latest_features)

    def save_inference_log(self, result_df: pd.DataFrame, prob_threshold: float = 0.55):
        """將每次推論的結果寫入獨立的 CSV Log 檔案中"""
        if result_df.empty:
            return

        os.makedirs("logs", exist_ok=True)
        log_file = "logs/inference_history.csv"

        log_records = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for _, row in result_df.iterrows():
            prob = row.get("pred_proba", 0.0)
            status = "LONG" if prob >= prob_threshold else "WAIT"
            
            log_records.append({
                "timestamp": current_time,
                "symbol": row["symbol"],
                "close_price": row["close"],
                "pred_proba": prob,
                "status": status
            })

        log_df = pd.DataFrame(log_records)

        file_exists = os.path.exists(log_file)
        log_df.to_csv(log_file, mode="a", header=not file_exists, index=False)
        logger.info(f"📁 預測結果已成功寫入至日誌檔: {log_file}")