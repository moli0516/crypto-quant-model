"""
src/live/binance_spot_trader.py
==============================================================================
 Binance Spot Demo/Testnet 純交易所驅動交易器
- 資金與持倉以交易所為唯一真實來源
- 本地只保留輕量 OCO 訂單記錄（方便追蹤與 Telegram 顯示）
- 開倉：市價買入 + 現貨原生 OCO
==============================================================================
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

import ccxt.async_support as ccxt
import pandas as pd
from dotenv import load_dotenv

from src.config import (
    POSITION_SIZE_RATIO,
    DEFAULT_TAKE_PROFIT_PCT,
    DEFAULT_STOP_LOSS_PCT,
    STATE_FILE,               # 我們改用這個檔案存輕量訂單記錄
    LOGS_DIR,
    TRADES_LOG_FILE,
    TRANSACTION_FEE_RATE,
)
from src.live.utils.notifier import send_telegram_alert

load_dotenv(override=True)
logger = logging.getLogger(__name__)


class BinanceSpotTrader:
    """
    純 Testnet 驅動的現貨交易器
    - 現金 / 餘額 → 即時從交易所讀取
    - 持倉判斷 → 只看本地記錄的「我們自己發出的 OCO」
    """

    def __init__(self):
        self.trading_environment = os.getenv("BINANCE_TRADING_ENV", "demo").strip().lower()
        if self.trading_environment not in {"demo", "testnet"}:
            raise ValueError("BINANCE_TRADING_ENV must be 'demo' or 'testnet'")
        credential_prefix = "BINANCE_SPOT_DEMO" if self.trading_environment == "demo" else "BINANCE_SPOT_TESTNET"
        self.api_key = os.getenv(f"{credential_prefix}_API_KEY", "").strip("'\" \t\r\n")
        self.api_secret = os.getenv(f"{credential_prefix}_API_SECRET", "").strip("'\" \t\r\n")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                f"Missing {credential_prefix}_API_KEY / {credential_prefix}_API_SECRET"
            )

        self.exchange = ccxt.binance({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
                "adjustForTimeDifference": True,
            },
        })
        if self.trading_environment == "demo":
            self.exchange.enable_demo_trading(True)
        else:
            self.exchange.set_sandbox_mode(True)

        # 本地輕量訂單記錄（只存我們自己發出的 OCO）
        self.orders_file = STATE_FILE          # 沿用原本路徑，但內容改為訂單記錄
        os.makedirs(LOGS_DIR, exist_ok=True)
        self.local_orders: List[Dict[str, Any]] = self._load_local_orders()
        self._sync_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # 本地輕量訂單記錄
    # ------------------------------------------------------------------
    def _load_local_orders(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.orders_file):
            return []
        try:
            with open(self.orders_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 相容舊格式
                if isinstance(data, dict) and "open_positions" in data:
                    return data.get("open_positions", [])
                if isinstance(data, list):
                    return data
                return []
        except Exception as e:
            logger.error(f"❌ 讀取本地訂單記錄失敗: {e}")
            return []

    def _save_local_orders(self) -> None:
        try:
            with open(self.orders_file, "w", encoding="utf-8") as f:
                json.dump(self.local_orders, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ 儲存本地訂單記錄失敗: {e}")

    def get_active_symbols(self) -> List[str]:
        """回傳目前本地記錄中仍有效的持倉幣種"""
        return [o["symbol"] for o in self.local_orders if o.get("status") == "OPEN"]

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """回傳目前本地記錄的 OPEN 訂單"""
        return [o for o in self.local_orders if o.get("status") == "OPEN"]

    # ------------------------------------------------------------------
    # 交易所即時資訊
    # ------------------------------------------------------------------
    async def get_usdt_balance(self) -> Dict[str, float]:
        """即時從交易所取得 USDT 餘額"""
        try:
            balance = await self.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            return {
                "free": float(usdt.get("free", 0.0)),
                "used": float(usdt.get("used", 0.0)),
                "total": float(usdt.get("total", 0.0)),
            }
        except Exception as e:
            logger.error(f"⚠️ 取得 USDT 餘額失敗: {e}")
            return {"free": 0.0, "used": 0.0, "total": 0.0}

    async def get_account_snapshot(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """Return exchange cash and mark-to-market non-USDT holdings."""
        balance = await self.exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        if not isinstance(usdt, dict):
            usdt = {}
        holding_value = 0.0
        holdings: Dict[str, float] = {}
        for asset, wallet in balance.items():
            if asset in {"info", "free", "used", "total"} or asset == "USDT":
                continue
            if not isinstance(wallet, dict):
                continue
            amount = float(wallet.get("total") or 0.0)
            symbol = f"{asset}USDT"
            price = prices.get(symbol, 0.0)
            if amount and price:
                holdings[symbol] = amount
                holding_value += amount * price
        return {
            "available_cash": float(usdt.get("free") or 0.0),
            "usdt_total": float(usdt.get("total") or 0.0),
            "holding_value": holding_value,
            "holdings": holdings,
        }

    async def sync_and_cleanup_orders(self) -> None:
        """Serialize exchange reconciliation so concurrent polls cannot duplicate closes."""
        async with self._sync_lock:
            await self._sync_and_cleanup_orders()

    async def monitor_order_status(self, interval_seconds: float = 15.0) -> None:
        """Continuously reconcile filled/canceled orders independently of inference."""
        logger.info("📡 Order-status monitor started; polling every %.1f seconds", interval_seconds)
        while True:
            try:
                await self.sync_and_cleanup_orders()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("⚠️ Order-status polling failed")
            await asyncio.sleep(interval_seconds)

    async def _sync_and_cleanup_orders(self) -> None:
        """
        同步本地記錄與交易所實際 OCO 狀態。
        OCO 成交後，以交易所成交資料補齊實際出場價格、數量、手續費與損益。
        """
        try:
            # 取得所有未成交訂單
            open_orders = await self.exchange.fetch_open_orders()
            open_order_ids = {str(o["id"]) for o in open_orders}
            open_list_ids = {str(o.get("info", {}).get("orderListId", "")) for o in open_orders}

            changed = False
            for order in self.local_orders:
                if order.get("status") != "OPEN":
                    continue

                oco_id = str(order.get("oco_order_list_id", ""))
                buy_id = str(order.get("buy_order_id", ""))

                # 如果 OCO 已經不在交易所未成交列表，視為已結束
                if oco_id and oco_id not in open_list_ids and buy_id not in open_order_ids:
                    symbol = str(order.get("symbol", "")).upper()
                    closed_orders = await self.exchange.fetch_closed_orders(
                        symbol.replace("USDT", "/USDT")
                    )
                    exit_order = self._find_exit_order(order, closed_orders)
                    if exit_order is None or float(exit_order.get("filled") or 0.0) <= 0.0:
                        logger.warning(
                            "⚠️ %s OCO 已離開未成交列表，但找不到已成交 SELL；保留 OPEN 狀態",
                            symbol,
                        )
                        continue

                    self._apply_close_data(order, exit_order)
                    self._save_local_orders()
                    self._append_realized_trade(order)
                    self._notify_closed_order(order)
                    changed = True
                    logger.info(
                        "📝 本地記錄更新: %s OCO 已成交 | PnL $%.4f",
                        symbol,
                        order["pnl_usd"],
                    )

            if changed:
                self._save_local_orders()

        except Exception as e:
            logger.warning(f"⚠️ 同步訂單狀態時發生錯誤: {e}")

    def _notify_closed_order(self, order: Dict[str, Any]) -> None:
        pnl = float(order.get("pnl_usd") or 0.0)
        pnl_pct = float(order.get("pnl_pct") or 0.0)
        mark = "🟢" if pnl >= 0 else "🔴"
        environment = self.trading_environment.title()
        message = (
            f"📉 *[SPOT {environment}] OCO 平倉通知*\n"
            "-----------------------------------\n"
            f"• *標的*: `{order.get('symbol', '-')}`\n"
            f"• *進場價*: `${float(order.get('entry_price') or 0.0):.8f}` ➔ "
            f"*出場價*: `${float(order.get('exit_price') or 0.0):.8f}`\n"
            f"• *出場數量*: `{float(order.get('exit_qty') or 0.0):.8f}`\n"
            f"• *已實現損益*: {mark} `${pnl:+.2f} USD` (`{pnl_pct * 100:+.2f}%`)\n"
            f"• *手續費*: `${float(order.get('fee_usd') or 0.0):.4f}`\n"
            f"• *平倉時間*: `{order.get('close_time', '-')}`"
        )
        send_telegram_alert(message)

    @staticmethod
    def _find_exit_order(
        order: Dict[str, Any], closed_orders: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Select the filled OCO SELL child order for a local position."""
        known_ids = {str(item) for item in order.get("oco_order_ids", [])}
        entry_time = datetime.strptime(
            str(order.get("entry_time")), "%Y-%m-%d %H:%M:%S"
        ).timestamp() * 1000
        candidates = []
        for exchange_order in closed_orders:
            if str(exchange_order.get("side", "")).lower() != "sell":
                continue
            if known_ids and str(exchange_order.get("id")) not in known_ids:
                continue
            timestamp = exchange_order.get("timestamp")
            if timestamp is not None and float(timestamp) < entry_time:
                continue
            if float(exchange_order.get("filled") or 0.0) <= 0.0:
                continue
            candidates.append(exchange_order)
        return max(candidates, key=lambda item: item.get("timestamp") or 0) if candidates else None

    @staticmethod
    def _order_fee_usd(exchange_order: Dict[str, Any], price: float) -> float:
        fees = exchange_order.get("fees") or []
        if not fees and exchange_order.get("fee"):
            fees = [exchange_order["fee"]]
        total = 0.0
        symbol = str(exchange_order.get("symbol", "")).split("/")
        quote = symbol[1] if len(symbol) == 2 else "USDT"
        base = symbol[0] if symbol else ""
        for fee in fees:
            amount = float(fee.get("cost") or fee.get("amount") or 0.0)
            currency = str(fee.get("currency") or quote).upper()
            total += amount if currency == quote else amount * price if currency == base else 0.0
        return total

    def _apply_close_data(self, order: Dict[str, Any], exit_order: Dict[str, Any]) -> None:
        entry_price = float(order.get("entry_price") or 0.0)
        exit_qty = float(exit_order.get("filled") or 0.0)
        exit_price = float(exit_order.get("average") or exit_order.get("price") or 0.0)
        if exit_price <= 0.0:
            exit_price = float(exit_order.get("cost") or 0.0) / exit_qty

        entry_fee = float(order.get("entry_fee_usd") or 0.0)
        if entry_fee <= 0.0:
            entry_fee = entry_price * exit_qty * TRANSACTION_FEE_RATE
        exit_fee = self._order_fee_usd(exit_order, exit_price)
        if exit_fee <= 0.0:
            exit_fee = exit_price * exit_qty * TRANSACTION_FEE_RATE
        pnl = (exit_price - entry_price) * exit_qty - entry_fee - exit_fee
        margin = float(order.get("margin_usd") or entry_price * exit_qty)
        order.update({
            "status": "CLOSED",
            "close_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "close_reason": "OCO_FILLED",
            "exit_price": exit_price,
            "exit_qty": exit_qty,
            "fee_usd": round(entry_fee + exit_fee, 8),
            "pnl_usd": round(pnl, 8),
            "pnl_pct": round(pnl / margin, 8) if margin else 0.0,
        })

    @staticmethod
    def _append_realized_trade(order: Dict[str, Any]) -> None:
        record = {
            key: order.get(key)
            for key in (
                "trade_id", "symbol", "entry_time", "entry_price", "close_time",
                "exit_price", "exit_qty", "margin_usd", "fee_usd", "pnl_usd",
                "pnl_pct", "prob", "status",
            )
        }
        frame = pd.DataFrame([record])
        os.makedirs(os.path.dirname(TRADES_LOG_FILE) or ".", exist_ok=True)
        header_needed = not os.path.exists(TRADES_LOG_FILE)
        frame.to_csv(TRADES_LOG_FILE, mode="a", header=header_needed, index=False)

    # ------------------------------------------------------------------
    # 核心下單邏輯
    # ------------------------------------------------------------------
    async def execute_spot_buy_with_oco(
        self,
        symbol: str,
        current_price: float,
        prob: float,
    ) -> bool:
        """
        市價買入 + 掛現貨 OCO（TP Limit + SL Stop-Limit）
        成功後把訂單寫入本地輕量記錄
        """
        formatted_symbol = symbol.replace("USDT", "/USDT")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. 檢查是否已有該幣種的 OPEN 訂單
        if symbol in self.get_active_symbols():
            logger.info(f"⏭️ {symbol} 已有未結束的 OCO，跳過開倉")
            return False

        await self.exchange.load_markets()
        market = self.exchange.market(formatted_symbol)

        # 2. 計算下單金額
        usdt = await self.get_usdt_balance()
        trade_amount_usd = usdt["free"] * POSITION_SIZE_RATIO

        if trade_amount_usd < 11.0:  # 留一點緩衝
            logger.warning(f"⚠️ {symbol} 可用金額不足 (${trade_amount_usd:.2f})，跳過")
            return False

        raw_qty = trade_amount_usd / current_price
        qty = float(self.exchange.amount_to_precision(formatted_symbol, raw_qty))

        # 3. 計算 OCO 價格
        tp_price = float(self.exchange.price_to_precision(
            formatted_symbol, current_price * (1.0 + DEFAULT_TAKE_PROFIT_PCT)
        ))
        sl_stop_price = float(self.exchange.price_to_precision(
            formatted_symbol, current_price * (1.0 - DEFAULT_STOP_LOSS_PCT)
        ))
        sl_limit_price = float(self.exchange.price_to_precision(
            formatted_symbol, sl_stop_price * 0.998
        ))

        try:
            # 4. 市價買入
            logger.info(f"⚡ [Spot Buy] {symbol} Qty: {qty} @ ~${current_price:.4f}")
            buy_order = await self.exchange.create_market_buy_order(formatted_symbol, qty)

            filled_qty = float(buy_order.get("filled") or qty)
            filled_qty = float(self.exchange.amount_to_precision(formatted_symbol, filled_qty))
            avg_price = float(buy_order.get("average") or current_price)
            entry_fee = self._order_fee_usd(buy_order, avg_price)
            if entry_fee <= 0.0:
                entry_fee = avg_price * filled_qty * TRANSACTION_FEE_RATE

            # 5. 掛 OCO
            oco_params = {
                "symbol": market["id"],
                "side": "SELL",
                "quantity": self.exchange.amount_to_precision(formatted_symbol, filled_qty),
                "price": self.exchange.price_to_precision(formatted_symbol, tp_price),
                "stopPrice": self.exchange.price_to_precision(formatted_symbol, sl_stop_price),
                "stopLimitPrice": self.exchange.price_to_precision(formatted_symbol, sl_limit_price),
                "stopLimitTimeInForce": "GTC",
            }

            logger.info(f"🛡️ [OCO] {symbol} TP: ${tp_price} | SL: ${sl_stop_price}")
            oco_resp = await self.exchange.private_post_order_oco(oco_params)

            oco_list_id = oco_resp.get("orderListId")
            oco_order_ids = [
                item.get("orderId") or item.get("id")
                for item in oco_resp.get("orders", [])
                if item.get("orderId") or item.get("id")
            ]

            # 6. 寫入本地輕量記錄
            record = {
                "trade_id": f"SPOT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{symbol}",
                "symbol": symbol,
                "status": "OPEN",
                "entry_time": now_str,
                "entry_price": avg_price,
                "qty": filled_qty,
                "margin_usd": round(filled_qty * avg_price, 4),
                "entry_fee_usd": round(entry_fee, 8),
                "prob": prob,
                "tp_price": tp_price,
                "sl_price": sl_stop_price,
                "buy_order_id": buy_order.get("id"),
                "oco_order_list_id": oco_list_id,
                "oco_order_ids": oco_order_ids,
            }
            self.local_orders.append(record)
            self._save_local_orders()

            # 7. Telegram 通知
            msg = (
                f"🚀 *[SPOT TESTNET] 開倉成功*\n"
                f"-----------------------------------\n"
                f"• *標的*: `{symbol}`\n"
                f"• *買入均價*: `${avg_price:.4f}`\n"
                f"• *數量*: `{filled_qty}`\n"
                f"• *投入*: `${record['margin_usd']:.2f} USDT`\n"
                f"• *信心度*: `{prob:.4f}`\n"
                f"• *止盈 (TP)*: `${tp_price}` (+{DEFAULT_TAKE_PROFIT_PCT*100:.1f}%)\n"
                f"• *止損 (SL)*: `${sl_stop_price}` (-{DEFAULT_STOP_LOSS_PCT*100:.1f}%)\n"
            )
            send_telegram_alert(msg)
            return True

        except Exception as e:
            logger.error(f"❌ [Spot Testnet Failed] {symbol}: {e}")
            send_telegram_alert(f"🚨 *[SPOT TESTNET ERROR]* `{symbol}` 開倉失敗:\n`{e}`")
            return False

    async def close(self):
        """安全關閉 CCXT 連線"""
        if hasattr(self, "exchange") and self.exchange is not None:
            try:
                await self.exchange.close()
                logger.info("🛑 BinanceSpotTrader exchange 已安全關閉")
            except Exception as e:
                logger.warning(f"⚠️ 關閉 exchange 時發生例外: {e}")