import abc
import time
import logging
import requests
from typing import Optional, Dict, Any, Union
import pandas as pd
from requests.exceptions import RequestException, HTTPError, Timeout

# 設定模組層級的 Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BaseRESTCollector(abc.ABC):
    """
    REST API 資料收集器的抽象基底類別。
    提供標準化的 HTTP 請求管理、重試機制與統一的介面合約。
    """

    def __init__(
        self, 
        base_url: str, 
        api_key: Optional[str] = None, 
        api_secret: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 10
    ) -> None:
        """
        初始化基礎收集器並建立連線池。
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.api_secret = api_secret
        self.max_retries = max_retries
        self.timeout = timeout
        
        # 🛡️ 建立 Session 以重複利用底層 TCP 連線，提升效能
        self.session = requests.Session()
        self._setup_default_headers()
        
        logger.info(f"🚀 初始化 REST Collector 成功 | Base URL: {self.base_url}")

    def _setup_default_headers(self) -> None:
        """設定預設的 HTTP Headers，可由子類別覆寫以適應不同交易所的認證需求。"""
        headers = {
            "Accept": "application/json",
            "User-Agent": "CryptoQuant-Collector/1.0"
        }
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key  # 以 Binance 標準為例，可於子類別覆寫
            logger.info("🔒 API Key 已掛載至 Headers")
        
        self.session.headers.update(headers)

    def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        核心請求方法，封裝了重試機制與防禦性錯誤處理。
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=data,
                    timeout=self.timeout
                )
                
                # 檢查 HTTP 狀態碼
                response.raise_for_status()
                return response.json()

            except HTTPError as e:
                status_code = e.response.status_code if e.response else None
                # ⚠️ 處理 Rate Limit (429) 或伺服器錯誤 (5xx)
                if status_code in [429, 500, 502, 503, 504]:
                    wait_time = (2 ** attempt)  # 指數退避 (2s, 4s, 8s...)
                    logger.warning(f"⚠️ 請求失敗 (狀態碼: {status_code}) | 嘗試次數: {attempt}/{self.max_retries} | {wait_time} 秒後重試...")
                    time.sleep(wait_time)
                else:
                    # ❌ 客戶端錯誤 (如 400 參數錯誤, 401 授權失敗) 不應重試，直接拋出
                    logger.error(f"❌ 無法恢復的請求錯誤 | URL: {url} | 狀態碼: {status_code}")
                    raise

            except (Timeout, ConnectionError) as e:
                wait_time = (2 ** attempt)
                logger.warning(f"🌐 網路連線或超時異常: {str(e)} | 嘗試次數: {attempt}/{self.max_retries} | {wait_time} 秒後重試...")
                time.sleep(wait_time)
                
        logger.error(f"❌ 達到最大重試次數 ({self.max_retries}) | URL: {url}")
        raise RequestException(f"API 請求失敗: {url}")

    @abc.abstractmethod
    def fetch_historical_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: Optional[Union[int, str]] = None,
        end_time: Optional[Union[int, str]] = None
    ) -> pd.DataFrame:
        """
        [介面合約] 獲取歷史 K 線資料。
        所有繼承此類別的子類別都必須實作此方法，並回傳標準化的 pandas DataFrame。
        """
        pass

    def close(self) -> None:
        """
        關閉 Session，釋放系統資源。
        """
        self.session.close()
        logger.info("🛑 REST Collector Session 已安全關閉並釋放資源")
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()