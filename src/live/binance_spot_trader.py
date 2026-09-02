"""
src/live/binance_spot_trader.py
==============================================================================
Binance Spot Testnet Order Execution & OCO Engine (CCXT)
專為現貨市場設計：市價買入 USDT 顆數、精準數量精度裁切與現貨 OCO (TP/SL) 條件單發射。
==============================================================================
"""

import os
import json
import logging
import asyncio
import ccxt.async_support as ccxt
import pandas as pd
from datetime import datetime
from typing import Dict, Any

from src.config import (
    INITIAL_CAPITAL,
    PROB_THRESHOLD,
    POSITION_SIZE_RATIO,
    DEFAULT_TAKE_PROFIT_PCT,
    DEFAULT_STOP_LOSS_PCT,
    STATE_FILE,
    TRADES_LOG_FILE,
    EQUITY_LOG_FILE,
    SYMBOL_BLACKLIST
)
from src.live.utils.notifier import send_telegram_alert

logger = logging.getLogger(__name__)


class BinanceSpotTrader:
    """
    幣安現貨 Testnet 實盤級下單執行器
    """

    def __init__(self):
        self.api_key = os.getenv("BINANCE_SPOT_TESTNET_API_KEY")
        self.api_secret = os.getenv("BINANCE_SPOT_TESTNET_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("❌ 未在環境變數中找到 BINANCE_SPOT_TESTNET_API_KEY / SECRET")

        # 初始化 CCXT Binance Spot (Testnet 模式)
        self.exchange = ccxt.binance({
            "apiKey": self.api_key.strip(),
            "secret": self.api_secret.strip(),
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
                "adjustForTimeDifference": True,
            },
        })
        self.exchange.set_sandbox_mode(True)   # ← 官方推薦方式

        self.state_file = STATE_FILE
        self.trades_log_file = TRADES_LOG_FILE
        self.equity_log_file = EQUITY_LOG_FILE
        
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_file):
            init_state = {"cash": INITIAL_CAPITAL, "total_equity": INITIAL_CAPITAL, "open_positions": []}
            self._save_state(init_state)
            return init_state
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 讀取狀態檔失敗: {e}")
            return {"cash": INITIAL_CAPITAL, "total_equity": INITIAL_CAPITAL, "open_positions": []}

    def _save_state(self, state: Dict[str, Any]) -> None:
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ 儲存狀態檔失敗: {e}")

    async def sync_account_balance(self) -> float:
        """同步幣安 Spot Testnet 可用 USDT 餘額"""
        try:
            balance = await self.exchange.fetch_balance()
            usdt_free = float(balance["USDT"]["free"])
            usdt_total = float(balance["USDT"]["total"])
            self.state["cash"] = usdt_free
            self.state["total_equity"] = usdt_total
            self._save_state(self.state)
            return usdt_total
        except Exception as e:
            logger.error(f"⚠️ [Spot Testnet] 無法獲取帳戶餘額: {e}")
            return self.state["total_equity"]

    async def execute_spot_buy_with_oco(self, symbol: str, current_price: float, prob: float) -> bool:
        """
        現貨市價買入 + 現貨原生 OCO (TP Limit / SL Stop-Limit)
        """
        formatted_symbol = symbol.replace("USDT", "/USDT")  # BTC/USDT
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        await self.exchange.load_markets()
        market = self.exchange.market(formatted_symbol)

        # 1. 計算下單金額與數量
        equity = await self.sync_account_balance()
        trade_amount_usd = equity * POSITION_SIZE_RATIO

        if trade_amount_usd < 10.0:
            logger.warning(f"⚠️ {symbol} 開倉金額 ${trade_amount_usd:.2f} 低於現貨 $10 Limit，跳過")
            return False

        raw_qty = trade_amount_usd / current_price
        qty = float(self.exchange.amount_to_precision(formatted_symbol, raw_qty))

        # 2. 計算 OCO 價格
        tp_price = float(self.exchange.price_to_precision(
            formatted_symbol, current_price * (1.0 + DEFAULT_TAKE_PROFIT_PCT)
        ))
        sl_stop_price = float(self.exchange.price_to_precision(
            formatted_symbol, current_price * (1.0 - DEFAULT_STOP_LOSS_PCT)
        ))
        # SL Limit 稍微再低一點，確保能成交
        sl_limit_price = float(self.exchange.price_to_precision(
            formatted_symbol, sl_stop_price * 0.998
        ))

        try:
            # 3. 市價買入
            logger.info(f"⚡ [Spot Buy Order] {symbol} Qty: {qty} @ ~${current_price:.4f}")
            buy_order = await self.exchange.create_market_buy_order(formatted_symbol, qty)

            filled_qty = float(buy_order.get("filled") or qty)
            filled_qty = float(self.exchange.amount_to_precision(formatted_symbol, filled_qty))

            # 4. 掛 OCO 賣單（隱式 API）
            oco_params = {
                "symbol": market["id"],
                "side": "SELL",
                "quantity": self.exchange.amount_to_precision(formatted_symbol, filled_qty),
                "price": self.exchange.price_to_precision(formatted_symbol, tp_price),
                "stopPrice": self.exchange.price_to_precision(formatted_symbol, sl_stop_price),
                "stopLimitPrice": self.exchange.price_to_precision(formatted_symbol, sl_limit_price),
                "stopLimitTimeInForce": "GTC",
            }

            logger.info(f"🛡️ [Spot OCO Order] 掛載 OCO 賣單 | TP: ${tp_price} | SL Stop: ${sl_stop_price}")
            oco_order = await self.exchange.private_post_order_oco(oco_params)

            # 5. 更新本地狀態
            new_pos = {
                "trade_id": f"SPOT-{datetime.now().strftime('%Y%m%d%H%M')}-{symbol}",
                "symbol": symbol,
                "entry_time": now_str,
                "entry_price": current_price,
                "qty": filled_qty,
                "margin": trade_amount_usd,
                "prob": prob,
                "tp_price": tp_price,
                "sl_price": sl_stop_price,
                "buy_order_id": buy_order.get("id"),
                "oco_order_list_id": oco_order.get("orderListId"),  # OCO 回傳的是 orderListId
            }
            self.state["open_positions"].append(new_pos)
            self._save_state(self.state)

            # 6. Telegram 通知
            msg = (
                f"🚀 *[BINANCE SPOT TESTNET] 現貨買入成功*\n"
                f"-----------------------------------\n"
                f"• *標的*: `{symbol}`\n"
                f"• *買入均價*: `${current_price:.4f}`\n"
                f"• *數量*: `{filled_qty}`\n"
                f"• *信心度*: `{prob:.4f}`\n"
                f"• *止盈價 (TP)*: `${tp_price}` (+{DEFAULT_TAKE_PROFIT_PCT*100:.1f}%)\n"
                f"• *止損價 (SL)*: `${sl_stop_price}` (-{DEFAULT_STOP_LOSS_PCT*100:.1f}%)\n"
            )
            send_telegram_alert(msg)
            return True

        except Exception as e:
            logger.error(f"❌ [Spot Testnet Failed] {symbol}: {e}")
            send_telegram_alert(f"🚨 *[SPOT TESTNET ERROR]* {symbol} 開倉失敗: `{e}`")
            return False
        
    async def close(self):
        """安全關閉 CCXT exchange 連線，釋放 aiohttp connector"""
        if hasattr(self, "exchange") and self.exchange is not None:
            try:
                await self.exchange.close()
                logger.info("🛑 BinanceSpotTrader exchange 已安全關閉")
            except Exception as e:
                logger.warning(f"⚠️ 關閉 exchange 時發生例外: {e}")