import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd

from src.live.live_pipeline import LiveDataPipeline
from src.models.registry import ModelRegistry
from src.live.utils.notifier import check_and_notify
from src.live.paper_trader import PaperTrader  # 👈 匯入 PaperTrader 模組

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
        
        # 👈 初始化模擬交易引擎
        self.paper_trader = PaperTrader(
            initial_balance=200.0,
            prob_threshold=self.prob_threshold,
            position_pct=0.10,
            holding_hours=12
        )
        
    async def _execute_inference_cycle(self):
        """執行單次完整的推論與模擬交易處理"""
        logger.info("🔄 開始執行即時推論與模擬交易週期...")
        
        # 1. 取得最新特徵
        latest_features_df = await self.pipeline.fetch_and_process()
        
        if latest_features_df.empty:
            logger.error("❌ 無法取得最新特徵，略過此次推論。")
            return

        # 2. 確保只傳入模型看過的特徵欄位
        X_live = latest_features_df[self.model.feature_cols]
        
        # 3. 執行推論
        preds_proba = self.model.predict(X_live)
        
        # 4. 整理訊號
        latest_features_df["pred_proba"] = preds_proba
        
        print("\n" + "="*50)
        print(f"🕒 即時預測報告 | 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        for _, row in latest_features_df.iterrows():
            sym = row['symbol']
            prob = row['pred_proba']
            price = row['close']
            
            signal_marker = "🔥 進場做多" if prob >= self.prob_threshold else "➖ 觀望"
            print(f"{sym:<10} | 價格: {price:>10.4f} | 看漲機率: {prob*100:>5.2f}% | 狀態: {signal_marker}")
        print("="*50 + "\n")

        # 5. 寫入推論 Log 與觸發通知
        self.pipeline.save_inference_log(latest_features_df, prob_threshold=self.prob_threshold)
        
        # 6. 👈 驅動模擬交易引擎更新持倉與帳戶狀態
        try:
            self.paper_trader.update_and_process(latest_features_df)
        except Exception as e:
            logger.error(f"❌ 模擬交易模組執行錯誤: {e}")

        # 7. 檢查推播
        try:
            check_and_notify("logs/inference_history.csv")
        except Exception as e:
            logger.error(f"推播器發生錯誤: {e}")

    async def run_scheduler(self):
        """執行每小時整點排程迴圈"""
        logger.info("🚀 Live Trader 啟動，等待下一個整點...")
        
        while True:
            now = datetime.now()
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=1, microsecond=0)
            wait_seconds = (next_hour - now).total_seconds()
            
            logger.info(f"⏳ 系統休眠中，預計將於 {wait_seconds:.0f} 秒後 ({next_hour.strftime('%H:%M:%S')}) 醒來...")
            await asyncio.sleep(wait_seconds)
            
            try:
                await self._execute_inference_cycle()
            except Exception as e:
                logger.error(f"❌ 推論週期發生非預期錯誤: {e}")

if __name__ == "__main__":
    trader = LiveTrader(model_path="models/best_xgb_model.pkl", prob_threshold=0.53)
    asyncio.run(trader.run_scheduler())