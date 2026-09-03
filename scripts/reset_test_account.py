"""
scripts/reset_test_account.py
==============================================================================
Binance Spot Testnet 本地與遠端一鍵重置腳本
==============================================================================
1. 強制重置本地 logs/paper_account_state.json 為 10,000 USDT 初始狀態與 0 持倉。
2. 自動刪除舊的交易與權益歷史日誌 (paper_trades.csv, paper_equity_daily.csv)。
3. 調用 cancel_testnet_orders 撤銷遠端交易所所有掛設的 OCO 訂單。
==============================================================================
"""

import os
import json
import logging
import asyncio
from pathlib import Path

from src.config import (
    STATE_FILE,
    INITIAL_CAPITAL,
    TRADES_LOG_FILE,
    EQUITY_LOG_FILE,
    INFERENCE_LOG_FILE,
    LOGS_DIR
)
from scripts.cancel_testnet_orders import cancel_all_spot_testnet_orders

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def reset_local_state():
    """重置本地狀態檔與歷史日誌"""
    logger.info("🧹 開始重置本地帳戶狀態與日誌...")

    # 1. 重置 paper_account_state.json
    clean_state = {
        "initial_balance": INITIAL_CAPITAL,
        "cash": INITIAL_CAPITAL,
        "total_equity": INITIAL_CAPITAL,
        "open_positions": []
    }

    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_state, f, indent=4, ensure_ascii=False)
    
    logger.info(f" ✅ 本地狀態檔 ({STATE_FILE}) 已成功重置為初始資產: ${INITIAL_CAPITAL:,.2f} USD")

    # 2. 清除舊的交易與權益日誌 (可選保留推論歷史)
    for log_path in [TRADES_LOG_FILE, EQUITY_LOG_FILE]:
        if os.path.exists(log_path):
            try:
                os.remove(log_path)
                logger.info(f" 🗑️ 已清空舊歷史檔: {log_path}")
            except Exception as e:
                logger.warning(f" ⚠️ 清除 {log_path} 失敗: {e}")


async def main():
    print("=" * 70)
    print("🚀 啟動一鍵重置流程：Binance Spot Testnet 本地 + 遠端歸零")
    print("=" * 70)

    # 1. 重置本地 json 與 csv 日誌
    reset_local_state()

    # 2. 呼叫遠端撤單邏輯
    logger.info("📡 正在連接至 Binance Spot Testnet 撤銷遠端掛單...")
    await cancel_all_spot_testnet_orders()

    print("=" * 70)
    print("🎉 [SUCCESS] 重置完成！目前系統狀態：")
    print(f"   • 本地可用現金 (Cash): ${INITIAL_CAPITAL:,.2f} USD")
    print("   • 本地持倉 (Positions): 0 筆 (All Cash)")
    print("   • 遠端 Testnet 掛單: 全數撤銷")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())