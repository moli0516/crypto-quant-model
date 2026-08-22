import asyncio
import logging
import pandas as pd
from pathlib import Path
from src.collectors.binance.rest_client import BinanceAsyncRESTClient
from src.cleaners.binance_cleaner import BinanceOHLCVCleaner

logger = logging.getLogger(__name__)

TOP_20_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT", "LINKUSDT",
    "DOTUSDT", "MATICUSDT", "NEARUSDT", "UNIUSDT", "LTCUSDT",
    "ATOMUSDT", "ICPUSDT", "APTUSDT", "RENDERUSDT", "FETUSDT"
]

async def _fetch_and_clean_symbol(symbol: str, timeframe: str, limit: int):
    logger.info(f"🚀 開始處理幣種: {symbol}")
    async with BinanceAsyncRESTClient() as client:
        try:
            df = await client.fetch_historical_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
            if df.empty:
                logger.warning(f"⚠️ 幣種 {symbol} 抓取到的資料為空，跳過。")
                return

            cleaner = BinanceOHLCVCleaner(fill_method="ffill")
            cleaned_df = cleaner(df)

            output_dir = Path("data/interim")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{symbol.lower()}_clean_record.csv"
            
            cleaned_df.to_csv(output_path)
            logger.info(f"✅ {symbol} 儲存成功至: {output_path} | 筆數: {len(cleaned_df)}")
        except Exception as e:
            logger.error(f"❌ 處理 {symbol} 時發生錯誤: {e}")

async def run_batch_collection(timeframe: str = "1h", limit: int = 15000):
    logger.info(f"🌍 開始批量收集前 {len(TOP_20_SYMBOLS)} 大熱門加密貨幣歷史數據...")
    for symbol in TOP_20_SYMBOLS:
        await _fetch_and_clean_symbol(symbol, timeframe, limit)
        await asyncio.sleep(0.3)
    logger.info("🎉 所有幣種的歷史數據批量收集與清洗完畢！")