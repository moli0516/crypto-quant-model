import asyncio
import logging
import pandas as pd
from pathlib import Path
from src.collectors.binance.rest_client import BinanceAsyncRESTClient
from src.cleaners.binance_cleaner import BinanceOHLCVCleaner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 幣安市值與流動性前 20 大熱門 USDT 交易對
TOP_20_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT", "LINKUSDT",
    "DOTUSDT", "MATICUSDT", "NEARUSDT", "UNIUSDT", "LTCUSDT",
    "ATOMUSDT", "ICPUSDT", "APTUSDT", "RENDERUSDT", "FETUSDT"
]

async def fetch_and_clean_symbol(symbol: str, timeframe: str = "1h", limit: int = 15000):
    """
    非同步批次抓取單一幣種並執行標準化清洗
    """
    logger.info(f"🚀 開始處理幣種: {symbol}")
    async with BinanceAsyncRESTClient() as client:
        try:
            # 1. 抓取歷史 OHLCV
            df = await client.fetch_historical_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
            if df.empty:
                logger.warning(f"⚠️ 幣種 {symbol} 抓取到的資料為空，跳過。")
                return

            # 2. 執行清洗
            cleaner = BinanceOHLCVCleaner(fill_method="ffill")
            cleaned_df = cleaner(df)

            # 3. 儲存至 interim 資料夾，以幣種命名
            output_dir = Path("data/interim")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{symbol.lower()}_clean_record.csv"
            
            cleaned_df.to_csv(output_path)
            logger.info(f"✅ {symbol} 儲存成功至: {output_path} | 筆數: {len(cleaned_df)}")

        except Exception as e:
            logger.error(f"❌ 處理 {symbol} 時發生錯誤: {e}")

async def main():
    logger.info(f"🌍 開始批量收集前 {len(TOP_20_SYMBOLS)} 大熱門加密貨幣歷史數據...")
    
    # 為了避免觸發幣安 Rate Limit，我們可以分批或依序執行非同步任務
    for symbol in TOP_20_SYMBOLS:
        await fetch_and_clean_symbol(symbol)
        await asyncio.sleep(0.5)  # 溫和停頓

    logger.info("🎉 所有幣種的歷史數據收集與清洗管線執行完畢！")

if __name__ == "__main__":
    asyncio.run(main())