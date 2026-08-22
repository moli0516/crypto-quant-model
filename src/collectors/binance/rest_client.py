import asyncio
import logging
import pandas as pd
from typing import Optional
from src.collectors.base_async_rest import BaseAsyncRESTCollector

# 宣告模組層級的 logger
logger = logging.getLogger(__name__)

class BinanceAsyncRESTClient(BaseAsyncRESTCollector):
    """
    幣安 (Binance) 非同步 REST API 客戶端。
    """
    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__(base_url="https://api.binance.com", api_key=api_key)

    async def fetch_historical_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        獲取幣安歷史 K 線 (Klines) 資料，支援自動雙向分頁與 DataFrame 標準化。
        """
        endpoint = "/api/v3/klines"
        all_raw_data = []
        remaining_limit = limit
        
        # 🧭 決定分頁方向：有指定 start_time 就往「未來」抓，否則往「過去」追溯
        paginate_forward = start_time is not None
        
        current_start_time = start_time
        current_end_time = end_time

        # 🔄 自動分頁迴圈
        while remaining_limit > 0:
            fetch_limit = min(remaining_limit, 1000)
            
            params = {
                "symbol": symbol.upper(),
                "interval": timeframe,
                "limit": fetch_limit
            }
            if current_start_time: params["startTime"] = current_start_time
            if current_end_time: params["endTime"] = current_end_time

            raw_data = await self._request("GET", endpoint, params=params)

            if not raw_data:
                logger.info("已無更多歷史資料可供抓取。")
                break

            # 🪡 根據分頁方向無縫拼接資料
            if paginate_forward:
                all_raw_data.extend(raw_data)
                # 往未來：下一批起點 = 這一批最後一根 K 線時間 + 1 毫秒
                current_start_time = raw_data[-1][0] + 1
            else:
                # 往過去：新抓到的（更舊的）資料必須優先放在前面，確保時間序列正確
                all_raw_data = raw_data + all_raw_data
                # 下一批終點 = 這一批第一根 K 線時間 - 1 毫秒
                current_end_time = raw_data[0][0] - 1

            remaining_limit -= len(raw_data)

            # 如果回傳的數量少於請求數量，代表已經撞到歷史資料的盡頭
            if len(raw_data) < fetch_limit:
                break
            
            # 🛡️ 避免過度頻繁請求觸發 Rate Limit
            if remaining_limit > 0:
                await asyncio.sleep(0.1) 

        if not all_raw_data:
            return pd.DataFrame()

        # 🧹 資料標準化與清洗
        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ]
        
        # 確保回傳數量不多於原本要求的 limit
        if not paginate_forward:
            all_raw_data = all_raw_data[-limit:]
        else:
            all_raw_data = all_raw_data[:limit]
            
        df = pd.DataFrame(all_raw_data, columns=columns)
        
        # 捨棄不需要的欄位
        df = df.drop(columns=["close_time", "ignore"])
        
        # 將時間戳轉為 UTC datetime 物件
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms', utc=True)
        
        # 📉 效能優化：將字串轉換為 float32，節省記憶體
        numeric_cols = df.columns.drop("timestamp")
        df[numeric_cols] = df[numeric_cols].astype("float32")
        
        # 設定時間戳為 Index 方便後續時間序列分析
        df.set_index("timestamp", inplace=True)
        
        return df