"""
scripts/reset_test_account.py
==============================================================================
重置本地 Paper/Live Testnet 帳戶狀態檔與日誌
==============================================================================
"""

import os
import json
import shutil
from datetime import datetime
from src.config import (
    STATE_FILE,
    TRADES_LOG_FILE,
    EQUITY_LOG_FILE,
    INITIAL_CAPITAL,
    LOGS_DIR
)


def reset_local_account_state():
    print("🧹 開始執行本地 Test Account 狀態重置流程...")

    # 1. 備份舊的 logs 資料夾（防止數據意外丟失）
    backup_dir = LOGS_DIR / f"backup_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if os.path.exists(STATE_FILE) or os.path.exists(TRADES_LOG_FILE):
        os.makedirs(backup_dir, exist_ok=True)
        for f in [STATE_FILE, TRADES_LOG_FILE, EQUITY_LOG_FILE]:
            if os.path.exists(f):
                shutil.copy(f, backup_dir)
        print(f"📦 舊帳戶狀態與歷史交易已安全備份至: {backup_dir}")

    # 2. 重置 paper_account_state.json
    clean_state = {
        "cash": INITIAL_CAPITAL,
        "total_equity": INITIAL_CAPITAL,
        "open_positions": []
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_state, f, indent=4, ensure_ascii=False)
    print(f"✅ `paper_account_state.json` 已重置，初始資金: ${INITIAL_CAPITAL:,.2f} USD，持倉已清空")

    # 3. 重置交易與資產 CSV 日誌檔
    for log_path in [TRADES_LOG_FILE, EQUITY_LOG_FILE]:
        if os.path.exists(log_path):
            os.remove(log_path)
            print(f"🗑️ 已清空舊數據檔: {log_path}")

    print("✨ 本地狀態重置完成！\n")


if __name__ == "__main__":
    reset_local_account_state()