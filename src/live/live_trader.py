import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd

from src.live.live_pipeline import LiveDataPipeline
from src.models.registry import ModelRegistry
from src.live.paper_trader import PaperTrader
# 👈 匯入專門的訊號檢查推播函式
from src.live.utils.notifier import notify_inference_signals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveTrader:
    def __init__(self, model_path: str, prob_threshold: float = 0.53):
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT", "LINKUSDT",
            "DOTUSDT", "MATICUSDT", "NEARUSDT", "UNIUSDT", "LTCUSDT",
            "ATOMUSDT", "ICPUSDT", "APTUSDT", "RENDERUSDT", "FETUSDT"
        ]
        self.pipeline = LiveDataPipeline(symbols=self.symbols)
        self.prob_threshold = prob_threshold
        
        logger.info(f"載入模型: {model_path}")
        self.model = ModelRegistry.create("xgb_classifier")
        self.model = self.model.load(model_path) 
        
        self.paper_trader = PaperTrader(
            initial_balance=200.0,
            prob_threshold=self.prob_threshold,
            position_pct=0.10,
            holding_hours=12
        )
        
    async def _execute_inference_cycle(self):
        logger.info("🔄 開始執行即時推論與模擬交易週期...")
        
        latest_features_df = await self.pipeline.fetch_and_process()
        if latest_features_df.empty:
            logger.error("❌ 無法取得最新特徵，略過估算。")
            return

        X_live = latest_features_df[self.model.feature_cols]
        preds_proba = self.model.predict(X_live)
        latest_features_df["pred_proba"] = preds_proba
        
        # 1. 寫入 Inference History Log
        self.pipeline.save_inference_log(latest_features_df, prob_threshold=self.prob_threshold)
        
        # 2. 觸發推播器發送高勝率訊號卡片 (僅當包含 ≥ prob_threshold 的標的時)
        try:
            notify_inference_signals("logs/inference_history.csv", prob_threshold=self.prob_threshold)
        except Exception as e:
            logger.error(f"❌ 訊號推播器出錯: {e}")

        # 3. 驅動 PaperTrader 進行模擬交易開/平倉並處理交易通知
        try:
            self.paper_trader.update_and_process(latest_features_df)
        except Exception as e:
            logger.error(f"❌ 模擬交易模組執行錯誤: {e}")

    async def run_scheduler(self):
        logger.info("🚀 Live Trader 啟動，等待下一個整點...")
        while True:
            now = datetime.now()
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=1, microsecond=0)
            wait_seconds = (next_hour - now).total_seconds()
            
            logger.info(f"⏳ 系統休眠中，預計將於 {wait_seconds:.0f} 秒後 wake up...")
            await asyncio.sleep(wait_seconds)
            
            try:
                await self._execute_inference_cycle()
            except Exception as e:
                logger.error(f"❌ 推論週期發生非預期錯誤: {e}")

if __name__ == "__main__":
    trader = LiveTrader(model_path="models/best_xgb_model.pkl", prob_threshold=0.53)
    asyncio.run(trader.run_scheduler())