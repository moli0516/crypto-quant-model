import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from src.live.utils.notifier import send_telegram_alert

logger = logging.getLogger(__name__)

class PaperTrader:
    """
    模擬交易執行器 (Paper Trading Engine)
    管理模擬帳戶資金、倉位生命週期 (H=12)、歷史交易紀錄與淨值追蹤。
    """
    def __init__(
        self,
        initial_balance: float = 200.0,
        prob_threshold: float = 0.53,
        position_pct: float = 0.10,
        holding_hours: int = 12,
        fee_rate: float = 0.001,
        logs_dir: str = "logs"
    ):
        self.initial_balance = initial_balance
        self.prob_threshold = prob_threshold
        self.position_pct = position_pct
        self.holding_hours = holding_hours
        self.fee_rate = fee_rate
        self.logs_dir = logs_dir
        
        self.account_state_file = os.path.join(self.logs_dir, "paper_account_state.json")
        self.trades_log_file = os.path.join(self.logs_dir, "paper_trades.csv")
        self.equity_log_file = os.path.join(self.logs_dir, "paper_equity_daily.csv")
        
        os.makedirs(self.logs_dir, exist_ok=True)
        self.state = self._load_account_state()

    def _load_account_state(self) -> Dict[str, Any]:
        """讀取或初始化帳戶狀態檔"""
        if not os.path.exists(self.account_state_file):
            initial_state = {
                "initial_balance": self.initial_balance,
                "cash": self.initial_balance,
                "total_equity": self.initial_balance,
                "open_positions": []
            }
            self._save_account_state(initial_state)
            logger.info(f"✨ 初始化模擬交易帳戶成功，初始資金: ${self.initial_balance:.2f} USD")
            return initial_state
            
        try:
            with open(self.account_state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 讀取帳戶狀態檔失敗，建立預設狀態: {e}")
            return {
                "initial_balance": self.initial_balance,
                "cash": self.initial_balance,
                "total_equity": self.initial_balance,
                "open_positions": []
            }

    def _save_account_state(self, state: Dict[str, Any]) -> None:
        """寫入帳戶狀態檔"""
        try:
            with open(self.account_state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ 儲存帳戶狀態檔失敗: {e}")

    def update_and_process(self, inference_df: pd.DataFrame) -> Dict[str, Any]:
        """
        接收當前推論結果 DataFrame (包含 'symbol', 'close', 'pred_proba')，
        執行到期平倉檢驗、新進場訊號捕捉與淨值更新。
        """
        if inference_df.empty:
            logger.warning("⚠️ 傳入的推論結果為空，跳過模擬交易處理。")
            return self.state

        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # 價格對照表
        price_map = dict(zip(inference_df["symbol"], inference_df["close"]))
        prob_map = dict(zip(inference_df["symbol"], inference_df["pred_proba"]))

        # ----------------------------------------------------
        # 1. 檢驗並處理到期平倉 (H = 12 Hours)
        # ----------------------------------------------------
        remaining_positions = []
        for pos in self.state.get("open_positions", []):
            entry_time = datetime.strptime(pos["entry_time"], "%Y-%m-%d %H:%M:%S")
            exit_time_target = entry_time + timedelta(hours=self.holding_hours)
            
            # 若已達持倉時間上限，進行平倉
            if now >= exit_time_target:
                symbol = pos["symbol"]
                exit_price = price_map.get(symbol, pos["entry_price"])
                
                # 計算淨利 (扣除買入與賣出雙邊手續費)
                gross_pnl = (exit_price - pos["entry_price"]) * pos["qty"]
                total_fee = (pos["entry_price"] * pos["qty"] * self.fee_rate) + (exit_price * pos["qty"] * self.fee_rate)
                net_pnl = gross_pnl - total_fee
                pnl_pct = (net_pnl / pos["margin"]) if pos["margin"] > 0 else 0.0

                # 歸還本金與實現盈虧
                self.state["cash"] += (pos["margin"] + net_pnl)

                # 寫入歷史交易紀錄 CSV
                trade_record = {
                    "trade_id": pos["trade_id"],
                    "symbol": symbol,
                    "entry_time": pos["entry_time"],
                    "entry_price": pos["entry_price"],
                    "exit_time": now_str,
                    "exit_price": exit_price,
                    "qty": pos["qty"],
                    "margin_usd": round(pos["margin"], 4),
                    "pnl_usd": round(net_pnl, 4),
                    "pnl_pct": round(pnl_pct, 4),
                    "prob": pos["prob"],
                    "status": "CLOSED_H12"
                }
                
                df_trade = pd.DataFrame([trade_record])
                header_needed = not os.path.exists(self.trades_log_file)
                df_trade.to_csv(self.trades_log_file, mode="a", header=header_needed, index=False)

                msg = (
                    f"📉 *[PAPER TRADING] 到期平倉*\n"
                    f"• *標的*: `{symbol}`\n"
                    f"• *進場價*: `${pos['entry_price']:.4f}` | *平倉價*: `${exit_price:.4f}`\n"
                    f"• *實現盈虧*: `${net_pnl:+.2f} USD` ({pnl_pct*100:+.2f}%)\n"
                    f"• *當前可用現金*: `${self.state['cash']:.2f} USD`"
                )
                send_telegram_alert(msg)
                logger.info(f"✅ {symbol} 持倉滿 12 小時平倉完畢 | PnL: ${net_pnl:+.2f} USD")
            else:
                remaining_positions.append(pos)

        self.state["open_positions"] = remaining_positions

        # ----------------------------------------------------
        # 2. 掃描訊號並執行開倉 (One Position Per Symbol)
        # ----------------------------------------------------
        active_symbols = [p["symbol"] for p in self.state["open_positions"]]

        for _, row in inference_df.iterrows():
            symbol = row["symbol"]
            price = float(row["close"])
            prob = float(row["pred_proba"])

            # 觸發門檻且該幣種當前無持倉
            if prob >= self.prob_threshold and symbol not in active_symbols:
                trade_amount = self.state["total_equity"] * self.position_pct
                
                # 邊界保護：確保現金足夠且單筆大於幣安最小 $5 門檻
                if self.state["cash"] >= trade_amount and trade_amount >= 5.0:
                    qty = trade_amount / price
                    self.state["cash"] -= trade_amount

                    new_pos = {
                        "trade_id": f"T-{now.strftime('%Y%m%d%H%M')}-{symbol}",
                        "symbol": symbol,
                        "entry_time": now_str,
                        "entry_price": price,
                        "margin": trade_amount,
                        "qty": qty,
                        "prob": prob
                    }
                    
                    self.state["open_positions"].append(new_pos)
                    active_symbols.append(symbol)

                    msg = (
                        f"🚀 *[PAPER TRADING] 開倉觸發*\n"
                        f"• *標的*: `{symbol}`\n"
                        f"• *進場價*: `${price:.4f}`\n"
                        f"• *看漲機率*: `{prob:.4f}`\n"
                        f"• *開倉金額*: `${trade_amount:.2f} USD`"
                    )
                    send_telegram_alert(msg)
                    logger.info(f"🔥 {symbol} 觸發開倉訊號 (機率: {prob:.4f}) | 下注: ${trade_amount:.2f} USD")

        # ----------------------------------------------------
        # 3. 更新帳戶總權益 (現金 + 持倉未實現盈虧)
        # ----------------------------------------------------
        unrealized_pnl = 0.0
        for pos in self.state["open_positions"]:
            curr_price = price_map.get(pos["symbol"], pos["entry_price"])
            unrealized_pnl += (curr_price - pos["entry_price"]) * pos["qty"]

        used_margin = sum(p["margin"] for p in self.state["open_positions"])
        self.state["total_equity"] = self.state["cash"] + used_margin + unrealized_pnl
        self._save_account_state(self.state)

        # ----------------------------------------------------
        # 4. 寫入淨值紀錄 (Equity Log Snapshot)
        # ----------------------------------------------------
        equity_record = {
            "timestamp": now_str,
            "total_equity": round(self.state["total_equity"], 4),
            "cash": round(self.state["cash"], 4),
            "unrealized_pnl": round(unrealized_pnl, 4),
            "active_positions": len(self.state["open_positions"])
        }
        
        df_equity = pd.DataFrame([equity_record])
        header_needed = not os.path.exists(self.equity_log_file)
        df_equity.to_csv(self.equity_log_file, mode="a", header=header_needed, index=False)

        return self.state