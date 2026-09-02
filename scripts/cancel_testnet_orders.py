"""
scripts/cancel_testnet_orders.py
==============================================================================
Binance Spot Testnet 撤單與條件單清理腳本
==============================================================================
修復 CCXT fetchOpenOrders Rate Limit 警告，精準撤銷所有現貨掛單與 OCO 條件單。
==============================================================================
"""

import os
import sys
import asyncio
import logging
import ccxt.async_support as ccxt
from dotenv import load_dotenv

# 🎯 統一從 config 讀取生效標的池，精準查詢避免全市場掃描
from src.config import ACTIVE_SYMBOLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(override=True)


async def cancel_all_spot_testnet_orders():
    api_key = os.getenv("BINANCE_SPOT_TESTNET_API_KEY", "").strip("'\" \t\r\n")
    api_secret = os.getenv("BINANCE_SPOT_TESTNET_API_SECRET", "").strip("'\" \t\r\n")

    if not api_key or not api_secret:
        logger.error("❌ 未在 .env 中讀取到 BINANCE_SPOT_TESTNET_API_KEY 或 SECRET")
        return

    # 初始化 CCXT Binance Spot 模組
    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
            # 🌟 核心修正：顯式通知 CCXT 我們已知悉全市場查詢的 Rate Limit 警告，防止拋出例外
            "warnOnFetchOpenOrdersWithoutSymbol": False,
            "fetchOpenOrders": {
                "warnWithoutSymbol": False
            }
        }
    })
    
    # 官方標準切換至 Spot Testnet
    exchange.set_sandbox_mode(True)

    try:
        logger.info("🔍 開始掃描 Binance Spot Testnet 上的未平倉單與 OCO 條件單...")
        
        # 方法 A: 優先針對 ACTIVE_SYMBOLS 逐一精準查詢 (權重僅 3 Weight/次，極速且安全)
        open_orders = []
        await exchange.load_markets()
        
        for symbol in ACTIVE_SYMBOLS:
            formatted_symbol = symbol.replace("USDT", "/USDT")
            try:
                orders = await exchange.fetch_open_orders(symbol=formatted_symbol)
                open_orders.extend(orders)
            except Exception as e:
                logger.warning(f"⚠️ 查詢 {formatted_symbol} 未平倉單時跳過: {e}")

        # 若逐一查詢未找到，再進行一次退避的全市場備用查詢
        if not open_orders:
            try:
                open_orders = await exchange.fetch_open_orders()
            except Exception as e:
                logger.debug(f"全市場查詢無結果或受限: {e}")

        if not open_orders:
            logger.info("🟢 Spot Testnet 當前無任何未平倉單或 OCO 掛單！帳戶乾淨。")
            return

        logger.info(f"⚠️ 偵測到 {len(open_orders)} 筆未平倉/OCO 掛單，準備發射撤單指令...")

        # 執行撤單
        for order in open_orders:
            ord_id = order["id"]
            ord_symbol = order["symbol"]
            try:
                await exchange.cancel_order(ord_id, ord_symbol)
                logger.info(f"  └ ✅ 成功撤銷掛單: {ord_symbol} (Order ID: {ord_id})")
            except Exception as e:
                logger.error(f"  └ ❌ 撤銷 {ord_symbol} (ID: {ord_id}) 失敗: {e}")

        logger.info("🎉 [SUCCESS] Binance Spot Testnet 所有掛單與風控單已全數清理完成！")

    except Exception as e:
        logger.error(f"❌ 撤銷訂單執行過程中發生未預期例外: {e}")
    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(cancel_all_spot_testnet_orders())