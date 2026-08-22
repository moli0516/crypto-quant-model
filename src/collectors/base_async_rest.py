import abc
import asyncio
import logging
from typing import Optional, Dict, Any
import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

class BaseAsyncRESTCollector(abc.ABC):
    """
    非同步 REST API 資料收集器基底類別。
    依賴 aiohttp 進行高效能的非同步 HTTP 請求，並內建容錯機制。
    """

    def __init__(
        self, 
        base_url: str, 
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 10
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """非同步上下文管理器進入點，在此初始化 Session"""
        headers = {"Accept": "application/json", "User-Agent": "CryptoQuant-AsyncCollector/1.0"}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
            
        self.session = aiohttp.ClientSession(headers=headers, timeout=self.timeout)
        logger.info(f"🚀 初始化 Async REST Collector | Base URL: {self.base_url}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同步上下文管理器離開點，確保安全關閉 Session"""
        if self.session:
            await self.session.close()
            logger.info("🛑 Async REST Collector Session 已安全關閉")

    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """核心非同步請求方法，包含指數退避重試邏輯"""
        if not self.session:
            raise RuntimeError("Session 未初始化，請使用 'async with' 語法")

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.session.request(method=method.upper(), url=url, params=params) as response:
                    if response.status in [429, 500, 502, 503, 504]:
                        wait_time = 2 ** attempt
                        logger.warning(f"⚠️ 狀態碼 {response.status} | 嘗試 {attempt}/{self.max_retries} | 暫停 {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    response.raise_for_status()
                    return await response.json()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                wait_time = 2 ** attempt
                logger.warning(f"🌐 網路異常: {str(e)} | 嘗試 {attempt}/{self.max_retries} | 暫停 {wait_time}s")
                await asyncio.sleep(wait_time)
                
        logger.error(f"❌ 達到最大重試次數 | URL: {url}")
        raise Exception(f"API 請求完全失敗: {url}")

    @abc.abstractmethod
    async def fetch_historical_ohlcv(self, symbol: str, timeframe: str, **kwargs) -> pd.DataFrame:
        """[介面合約] 子類別必須實作的歷史數據獲取方法"""
        pass