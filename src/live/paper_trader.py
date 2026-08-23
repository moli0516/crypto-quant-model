import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

# 👈 匯入優化後的事件推播函式
from src.live.utils.notifier import notify_paper_trade_event

logger = logging.getLogger(__name__)

class PaperTrader:
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
        if not os.path.exists(self.account_state_file):
            initial_state = {
                "initial_balance": self.initial_balance,
                "cash": self.initial_balance,
                "total_equity": self.initial_balance,
                "open_positions": []
            }
            self._save_account_state(initial_state)
            return initial_state
        try:
            with open(self.account_state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 讀取帳戶狀態檔失敗: {e}")
            return {
                "initial_balance": self.initial_balance,
                "cash": self.initial_balance,
                "total_equity": self.initial_balance,
                "open_positions": []
            }

    def _save_account_state(self, state: Dict[str, Any]) -> None:
        try:
            with open(self.account_state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ 儲存帳戶狀態檔失敗: {e}")

    def update_and_process(self, inference_df: pd.DataFrame) -> Dict[str, Any]:
        if inference_df.empty:
            return self.state

        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        price_map = dict(zip(inference_df["symbol"], inference_df["close"]))

        # ----------------------------------------------------
        # 1. 處理到期平倉 (H = 12 Hours)
        # ----------------------------------------------------
        remaining_positions = []
        for pos in self.state.get("open_positions", []):
            entry_time = datetime.strptime(pos["entry_time"], "%Y-%m-%d %H:%M:%S")
            exit_time_target = entry_time + timedelta(hours=self.holding_hours)
            
            if now >= exit_time_target:
                symbol = pos["symbol"]
                exit_price = price_map.get(symbol, pos["entry_price"])
                
                gross_pnl = (exit_price - pos["entry_price"]) * pos["qty"]
                total_fee = (pos["entry_price"] * pos["qty"] * self.fee_rate) + (exit_price * pos["qty"] * self.fee_rate)
                net_pnl = gross_pnl - total_fee
                pnl_pct = (net_pnl / pos["margin"]) if pos["margin"] > 0 else 0.0

                self.state["cash"] += (pos["margin"] + net_pnl)

                # 寫入歷史交易 CSV
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

                # 👈 觸發優化後的平倉通知卡片
                notify_paper_trade_event("CLOSE", {
                    "symbol": symbol,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "pnl_usd": net_pnl,
                    "pnl_pct": pnl_pct,
                    "current_cash": self.state["cash"]
                })
            else:
                remaining_positions.append(pos)

        self.state["open_positions"] = remaining_positions

        # ----------------------------------------------------
        # 2. 掃描訊號與觸發開倉
        # ----------------------------------------------------
        active_symbols = [p["symbol"] for p in self.state["open_positions"]]

        for _, row in inference_df.iterrows():
            symbol = row["symbol"]
            price = float(row["close"])
            prob = float(row["pred_proba"])

            if prob >= self.prob_threshold and symbol not in active_symbols:
                trade_amount = self.state["total_equity"] * self.position_pct
                
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

                    # 👈 觸發優化後的開倉通知卡片
                    notify_paper_trade_event("OPEN", {
                        "symbol": symbol,
                        "entry_price": price,
                        "prob": prob,
                        "margin": trade_amount,
                        "entry_time": now_str
                    })

        # ----------------------------------------------------
        # 3. 更新權益與 Snapshot 紀錄
        # ----------------------------------------------------
        unrealized_pnl = 0.0
        for pos in self.state["open_positions"]:
            curr_price = price_map.get(pos["symbol"], pos["entry_price"])
            unrealized_pnl += (curr_price - pos["entry_price"]) * pos["qty"]

        used_margin = sum(p["margin"] for p in self.state["open_positions"])
        self.state["total_equity"] = self.state["cash"] + used_margin + unrealized_pnl
        self._save_account_state(self.state)

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