"""
src/config.py
==============================================================================
Centralized Configuration Module for Crypto Quant Trading Framework
包含交易標的池、風控甜點區、Ensemble 模型路徑、防洩漏參數與 Telegram 監控設定。
==============================================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 載入 .env 環境變數
load_dotenv()

# ==============================================================================
# 1. 專案基礎路徑設定 (Project Base Paths)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# 確保基礎資料夾存在
for directory in [LOGS_DIR, MODELS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. 交易標的池與黑名單 (Symbol Universe & Blacklist)
# ==============================================================================
# 預設 20 個熱門 USDT 交易對
TARGET_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "NEARUSDT", "FETUSDT",
    "ICPUSDT", "UNIUSDT", "SUIUSDT", "ARBUSDT", "OPUSDT",
    "BCHUSDT", "LTCUSDT", "AVAXUSDT", "LINKUSDT", "XRPUSDT",
    "ADAUSDT", "DOTUSDT", "MATICUSDT", "ATOMUSDT", "ETCUSDT"
]

# 🎯 基於實證數據（0% 勝率毒瘤幣種）之黑名單，推論層自動跳過
SYMBOL_BLACKLIST = {
    "AVAXUSDT", "ATOMUSDT", "DOTUSDT", "LINKUSDT", 
    "XRPUSDT", "LTCUSDT", "MATICUSDT"
}

# 經過黑名單過濾後的有效標的池
ACTIVE_SYMBOLS = [sym for sym in TARGET_SYMBOLS if sym not in SYMBOL_BLACKLIST]

# ==============================================================================
# 3. 訊號推論與 Ensemble 模型設定 (Inference & Model Ensemble)
# ==============================================================================
# 訊號發射機率門檻 (Probability Threshold)
PROB_THRESHOLD = 0.54

# 推論與 K 線時間週期
FEATURE_TIMEFRAME = "1h"  # 特徵生成週期
BASELINE_HOLD_HOURS = 12  # Phase 1 Baseline 自然持倉時數

# 模型架構配置：支援單模型或雙模型 Ensemble (Soft Voting)
USE_ENSEMBLE = True

# 模型檔案路徑
MODEL_PATHS = {
    "xgb": str(MODELS_DIR / "best_xgb_model.pkl"),
    "lgb": str(MODELS_DIR / "best_lgb_model.pkl")
}

# Ensemble Soft Voting 加權比例 (總和需為 1.0)
ENSEMBLE_WEIGHTS = {
    "xgb": 0.5,
    "lgb": 0.5
}

# ==============================================================================
# 4. 風控與網格甜點區設定 (Risk Management & OCO Order Parameters)
# ==============================================================================
# 初始資本與倉位控管
INITIAL_CAPITAL = 7287.90  # USD
POSITION_SIZE_RATIO = 0.10  # 單筆下注當下 Equity 的 10%
MAX_CONCURRENT_POSITIONS = 8  # 最大同時持倉數，防範全市場 Beta 連環跌

# 🎯 網格搜尋（1m K 線碰撞）產出之最新風控甜點區
DEFAULT_TAKE_PROFIT_PCT = 0.04  # TP 2.5%
DEFAULT_STOP_LOSS_PCT = 0.01    # SL 4.0%

# 交易手續費率 (Binance Standard Spot Fee: 0.10%)
TRANSACTION_FEE_RATE = 0.0010
TOP_K_SIGNALS = 3           # 🎯 每小時整點最多僅允許開倉前 K 個最高機率幣種 (建議 2~3)
USE_TOP_K_FILTER = True     # 是否開啟 Top-K 篩選機制
IS_SPOT_TRADING = True               # 標記為現貨模式

# ==============================================================================
# 5. 日誌與歷史資料檔案路徑 (Data & Log Paths)
# ==============================================================================
TRADING_ENV = os.getenv("BINANCE_TRADING_ENV", "demo").strip().lower()
if TRADING_ENV not in {"demo", "testnet"}:
    raise ValueError("BINANCE_TRADING_ENV must be 'demo' or 'testnet'")

STATE_FILE = str(LOGS_DIR / f"paper_account_state_{TRADING_ENV}.json")
INFERENCE_LOG_FILE = str(LOGS_DIR / "inference_history.csv")
TRADES_LOG_FILE = str(LOGS_DIR / f"paper_trades_{TRADING_ENV}.csv")
EQUITY_LOG_FILE = str(LOGS_DIR / "paper_equity_daily.csv")
SLTP_REPORT_LOG_FILE = str(LOGS_DIR / "sltp_grid_report.csv")
DIAGNOSTIC_IMG_FILE = str(LOGS_DIR / "diagnostic_report.png")
REPORT_IMG_FILE = str(LOGS_DIR / "equity_chart.png")
SLTP_REPORT_IMG_FILE = str(LOGS_DIR / "sltp_diagnostic_report.png")

# ==============================================================================
# 6. Telegram Bot 與 API 安全設定 (API Credentials & Rate Limits)
# ==============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Binance API 憑證 (僅在 Live Mode 下使用)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

# 防禦性 API 超時與重試設定
API_TIMEOUT = 10  # seconds
MAX_API_RETRIES = 3


# ==============================================================================
# 7. 配置防禦性自動校驗 (Self-Validation Check)
# ==============================================================================
def validate_config():
    """執行配置檔啟動時的防禦性檢查"""
    if USE_ENSEMBLE:
        total_weight = sum(ENSEMBLE_WEIGHTS.values())
        if not abs(total_weight - 1.0) < 1e-5:
            raise ValueError(f"❌ ENSEMBLE_WEIGHTS 權重總和必須為 1.0，目前為: {total_weight}")

    if POSITION_SIZE_RATIO <= 0 or POSITION_SIZE_RATIO > 1.0:
        raise ValueError(f"❌ POSITION_SIZE_RATIO 必須在 (0, 1.0] 之間，目前為: {POSITION_SIZE_RATIO}")

validate_config()