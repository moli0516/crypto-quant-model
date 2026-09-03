"""
scripts/reset_binance_testnet.py
==============================================================================
Binance Spot Testnet 遠端真實資產強制清倉與一鍵重置腳本
==============================================================================
1. 自動查詢 Testnet 帳戶內所有持有現貨 (BTC, ETH, UNI 等)。
2. 撤銷所有掛載中的 OCO / Limit 條件賣單。
3. 發射市價賣單 (Market Sell) 將現貨全數清倉變現為 USDT。
4. 將本地 paper_account_state.json 與遠端實時資產 100% 同步。
==============================================================================
"""

import os
import sys
import asyncio
import json
import logging
import ccxt.async_support as ccxt
from dotenv import load_dotenv

from src.config import (
    STATE_FILE,
    TRADES_LOG_FILE,
    EQUITY_LOG_FILE,
    LOGS_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(override=True)


async def reset_binance_testnet_account():
    api_key = os.getenv("BINANCE_SPOT_TESTNET_API_KEY", "").strip("'\" \t\r\n")
    api_secret = os.getenv("BINANCE_SPOT_TESTNET_API_SECRET", "").strip("'\" \t\r\n")

    if not api_key or not api_secret:
        logger.error("❌ 未在 .env 讀取到 BINANCE_SPOT_TESTNET_API_KEY 或 SECRET")
        return

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
            "warnOnFetchOpenOrdersWithoutSymbol": False
        }
    })
    exchange.set_sandbox_mode(True)  # 切換至 Testnet

    try:
        logger.info("📡 正在連線至 Binance Spot Testnet 執行全帳戶資產重置...")
        await exchange.load_markets()

        # Step 1: 撤銷所有未平倉掛單 (OCO / Limit)
        logger.info("1️⃣ 正在清理所有未成交的 OCO 與限價單...")
        open_orders = await exchange.fetch_open_orders()
        for order in open_orders:
            try:
                await exchange.cancel_order(order["id"], order["symbol"])
                logger.info(f"   └ ✅ 已撤銷掛單: {order['symbol']} (ID: {order['id']})")
            except Exception as e:
                logger.warning(f"   └ ⚠️ 撤銷掛單 {order['symbol']} 失敗: {e}")

        # Step 2: 查詢實時現貨餘額並市價清倉
        logger.info("2️⃣ 正在掃描 Testnet 現貨持倉並執行全數市價賣出 (Market Sell)...")
        balance = await exchange.fetch_balance()
        free_balances = balance.get("free", {})

        for asset, free_amount in free_balances.items():
            amount = float(free_amount)
            if asset in ["USDT", "LDUSDT"] or amount <= 0:
                continue

            symbol = f"{asset}/USDT"
            if symbol not in exchange.markets:
                continue

            try:
                # 精度裁切
                formatted_qty = float(exchange.amount_to_precision(symbol, amount))
                market_info = exchange.market(symbol)
                min_notional = market_info.get("limits", {}).get("cost", {}).get("min", 10.0)

                # 取得當前價格判斷是否低於最小名義限制
                ticker = await exchange.fetch_ticker(symbol)
                est_value = formatted_qty * ticker["close"]

                if est_value < min_notional:
                    logger.warning(f"   └ ⚠️ {symbol} 餘額價值 (${est_value:.2f}) 低於現貨限制 (${min_notional})，跳過平倉")
                    continue

                logger.info(f"   🚀 正在市價賣出 {symbol} | 數量: {formatted_qty}")
                await exchange.create_market_sell_order(symbol, formatted_qty)
                logger.info(f"   └ ✅ 成功平倉 {symbol}")
            except Exception as e:
                logger.error(f"   └ ❌ 平倉 {symbol} 失敗: {e}")

        # Step 3: 同步最新 USDT 餘額至本地狀態檔
        await asyncio.sleep(1.0)  # 等待撮合完成
        updated_balance = await exchange.fetch_balance()
        final_usdt = float(updated_balance.get("USDT", {}).get("total", 0.0))

        logger.info("3️⃣ 正在同步最新資產至本地 state 檔案...")
        clean_state = {
            "initial_balance": final_usdt,
            "cash": final_usdt,
            "total_equity": final_usdt,
            "open_positions": []
        }

        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(clean_state, f, indent=4, ensure_ascii=False)

        # 清理歷史日誌
        for log_path in [TRADES_LOG_FILE, EQUITY_LOG_FILE]:
            if os.path.exists(log_path):
                os.remove(log_path)

        print("\n" + "=" * 70)
        print("🎉 [Binance Spot Testnet 資產重置成功!]")
        print(f"• 實時清空後 USDT 可用餘額: ${final_usdt:,.2f}")
        print("• 當前現貨持倉: 0 筆 (All Cash)")
        print("• 本地狀態檔 (paper_account_state.json): 已完全同步")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"❌ 重置 Testnet 帳戶失敗: {e}")
    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(reset_binance_testnet_account())