"""
_config_example.py
==============================================================================
Centralized Configuration Template for Crypto Quant Trading Framework
開源專案配置範本。請複製此檔案並重命名為 `src/config.py` 以開始使用。
包含專案路徑、標的池剪枝、Ensemble 雙模型設定、OCO 風控參數與防禦性啟動校驗。
==============================================================================
"""

import os
from pathlib import Path
from typing import Dict, List, Set
from dotenv import load_dotenv

# 載入 .env 環境變數 (若存在)
load_dotenv()

# ==============================================================================
# 1. 專案基礎路徑設定 (Project Base Paths)
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent
LOGS_DIR: Path = BASE_DIR / "logs"
MODELS_DIR: Path = BASE_DIR / "models"
DATA_DIR: Path = BASE_DIR / "data"

# 防禦性目錄自動創建：確保基本 File System 結構完整
for directory in [LOGS_DIR, MODELS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. 交易標的池與黑名單 (Symbol Universe & Blacklist)
# ==============================================================================
# 預設監控之熱門 USDT 交易對
TARGET_SYMBOLS: List[str] = [
    "BTCUSDT"
]

# 🎯 實證數據導向黑名單 (剔除歷史微觀結構差、0% 勝率之毒瘤幣種)
SYMBOL_BLACKLIST: Set[str] = {
    "ETHUSDT"
}

# 動態剪枝過濾後之有效標的池 (Active Universe)
ACTIVE_SYMBOLS: List[str] = [
    sym for sym in TARGET_SYMBOLS if sym not in SYMBOL_BLACKLIST
]

# ==============================================================================
# 3. 訊號推論與 Ensemble 模型設定 (Inference & Model Ensemble)
# ==============================================================================
# LONG 訊號發射門檻
PROB_THRESHOLD: float = 1.0

# 時間週期與 Baseline 設定
FEATURE_TIMEFRAME: str = "12h"  # 特徵採樣週期
BASELINE_HOLD_HOURS: int = int('inf')   # Phase 1 無風控持有上限

# 是否啟用雙模型 Ensemble (Soft Voting)
USE_ENSEMBLE: bool = False

# 模型檔案映射路徑
MODEL_PATHS: Dict[str, str] = {
    "xgb": str(MODELS_DIR / "xgb_baseline.pkl")
}

# Ensemble Soft Voting 加權權重 (權重總和必須等於 1.0)
ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "xgb": 1
}

# ==============================================================================
# 4. 風控與 1m 網格尋優甜點區 (Risk Management & OCO Order Parameters)
# ==============================================================================
# 資金控管設定
INITIAL_CAPITAL: float = 200.0         # 初始模擬/實盤本金 (USD)
POSITION_SIZE_RATIO: float = 1      # 單筆下注比例 
MAX_CONCURRENT_POSITIONS: int = int('inf')      # 最大同時持倉數 (防範大盤 Systemic Beta 崩盤)

# 🎯 每 50 筆網格尋優 (1m High-Freq SL/TP Backtest) 鎖定之最優風控甜點區
DEFAULT_TAKE_PROFIT_PCT: float = 0  # TP
DEFAULT_STOP_LOSS_PCT: float = 0    # SL

# 交易所手續費率 (Binance Standard Fee: 0.10%)
TRANSACTION_FEE_RATE: float = 0.0010

# ==============================================================================
# 5. 日誌與歷史資料檔案路徑 (Data & Log File Paths)
# ==============================================================================
STATE_FILE: str = str(LOGS_DIR / "paper_account_state.json")
INFERENCE_LOG_FILE: str = str(LOGS_DIR / "inference_history.csv")
TRADES_LOG_FILE: str = str(LOGS_DIR / "paper_trades.csv")
EQUITY_LOG_FILE: str = str(LOGS_DIR / "paper_equity_daily.csv")
DIAGNOSTIC_IMG_FILE: str = str(LOGS_DIR / "diagnostic_report.png")
REPORT_IMG_FILE: str = str(LOGS_DIR / "equity_chart.png")
SLTP_REPORT_IMG_FILE: str = str(LOGS_DIR / "sltp_diagnostic_report.png")

# ==============================================================================
# 6. Telegram Bot 與 API 安全憑證 (Credentials & Rate Limits)
# ==============================================================================
# 優先從環境變數讀取，避免將明文 Key 提交至版本控制系統 (Git)
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID_HERE")

BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "YOUR_BINANCE_API_KEY_HERE")
BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "YOUR_BINANCE_SECRET_KEY_HERE")

# 防禦性 API 超時與重試設置
API_TIMEOUT: int = 10         # 秒
MAX_API_RETRIES: int = 3


# ==============================================================================
# 7. 運行時防禦性自動校驗 (Runtime Defensive Self-Validation)
# ==============================================================================
def validate_config() -> None:
    """
    啟動時防禦性檢查：攔截不合法的參數設定，防止系統在交易運行期間崩潰。
    """
    # 1. 校驗標的池與黑名單 logic
    if not ACTIVE_SYMBOLS:
        raise ValueError("❌ [Config Error] 剪枝後之 ACTIVE_SYMBOLS 標的池為空，請檢查 TARGET_SYMBOLS 與 SYMBOL_BLACKLIST。")

    # 2. 校驗 Ensemble 權重歸一化
    if USE_ENSEMBLE:
        total_weight = sum(ENSEMBLE_WEIGHTS.values())
        if not abs(total_weight - 1.0) < 1e-5:
            raise ValueError(f"❌ [Config Error] ENSEMBLE_WEIGHTS 權重總和必須為 1.0，當前總和為: {total_weight:.4f}")
        for name, path in MODEL_PATHS.items():
            if name not in ENSEMBLE_WEIGHTS:
                raise KeyError(f"❌ [Config Error] MODEL_PATHS 中的模型 [{name}] 缺乏相對應的 ENSEMBLE_WEIGHTS 權重設定。")

    # 3. 校驗倉位與風控數值範圍
    if not (0.0 < POSITION_SIZE_RATIO <= 1.0):
        raise ValueError(f"❌ [Config Error] POSITION_SIZE_RATIO 必須在 (0, 1.0] 區間內，當前設定為: {POSITION_SIZE_RATIO}")

    if DEFAULT_TAKE_PROFIT_PCT <= 0 or DEFAULT_STOP_LOSS_PCT <= 0:
        raise ValueError("❌ [Config Error] DEFAULT_TAKE_PROFIT_PCT 與 DEFAULT_STOP_LOSS_PCT 必須為正浮點數。")

    if MAX_CONCURRENT_POSITIONS < 1:
        raise ValueError(f"❌ [Config Error] MAX_CONCURRENT_POSITIONS 必須至少為 1，當前設定為: {MAX_CONCURRENT_POSITIONS}")


# 模組載入時自動執行防禦校驗
validate_config()