import os
import logging
import requests
import pandas as pd
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(message: str, parse_mode: str = "Markdown") -> bool:
    """
    發送 Telegram 訊息，內建輕量化重試機制。
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[NOTIFIER] 未設定 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID，跳過推播。")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }

    for attempt in range(1, 3):
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.info("[NOTIFIER] Telegram 訊息推播成功。")
                return True
            else:
                logger.error(f"[NOTIFIER] 發送失敗 (HTTP {response.status_code}): {response.text}")
        except Exception as e:
            logger.error(f"[NOTIFIER] 網路異常 (嘗試 {attempt}/2): {e}")

    return False

def notify_inference_signals(csv_path: str = "logs/inference_history.csv", prob_threshold: float = 0.53):
    """
    掃描最新推論結果，渲染高可讀性的訊號提醒卡片。
    """
    if not os.path.exists(csv_path):
        return

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return

        latest_ts = df["timestamp"].max()
        latest_df = df[df["timestamp"] == latest_ts]
        high_prob_signals = latest_df[latest_df["pred_proba"] >= prob_threshold]

        if not high_prob_signals.empty:
            msg = f"🚀 *[QUANT SIGNAL DETECTED]*\n"
            msg += f"⏰ *Time:* `{latest_ts}`\n"
            msg += f"🎯 *Threshold:* `{prob_threshold}`\n"
            msg += "-----------------------------------\n"
            
            for _, row in high_prob_signals.iterrows():
                prob_pct = row['pred_proba'] * 100
                msg += f"• *{row['symbol']}*\n"
                msg += f"  └ Price: `${row['close_price']:.4f}` | Prob: `{prob_pct:.2f}%` 🔥\n"
            
            send_telegram_alert(msg)
        else:
            logger.info(f"[NOTIFIER] {latest_ts} 無超過門檻訊號，維持觀望。")
    except Exception as e:
        logger.error(f"[NOTIFIER] 處理推論日誌時出錯: {e}")

def notify_paper_trade_event(event_type: str, data: Dict[str, Any]):
    """
    渲染模擬交易專用的卡片訊息 (OPEN / CLOSE / EQUITY_SNAPSHOT)
    """
    if event_type == "OPEN":
        msg = (
            f"📈 *[PAPER TRADE] 開倉通知*\n"
            f"-----------------------------------\n"
            f"• *標的*: `{data['symbol']}`\n"
            f"• *進場價格*: `${data['entry_price']:.4f}`\n"
            f"• *模型機率*: `{data['prob']*100:.2f}%`\n"
            f"• *投入金額*: `${data['margin']:.2f} USD`\n"
            f"• *進場時間*: `{data['entry_time']}`\n"
        )
    elif event_type == "CLOSE":
        pnl_symbol = "🟢" if data['pnl_usd'] >= 0 else "🔴"
        msg = (
            f"📉 *[PAPER TRADE] 到期平倉*\n"
            f"-----------------------------------\n"
            f"• *標的*: `{data['symbol']}`\n"
            f"• *進場價*: `${data['entry_price']:.4f}` ➔ *平倉價*: `${data['exit_price']:.4f}`\n"
            f"• *實現盈虧*: {pnl_symbol} `${data['pnl_usd']:+.2f} USD` (`{data['pnl_pct']*100:+.2f}%`)\n"
            f"• *可用現金*: `${data['current_cash']:.2f} USD`\n"
        )
    elif event_type == "SNAPSHOT":
        msg = (
            f"📊 *[PAPER TRADE] 帳戶狀態快照*\n"
            f"-----------------------------------\n"
            f"• *總資產 (Equity)*: `${data['total_equity']:.2f} USD`\n"
            f"• *可用現金 (Cash)*: `${data['cash']:.2f} USD`\n"
            f"• *未實現盈虧*: `${data['unrealized_pnl']:+.2f} USD`\n"
            f"• *當前持倉數*: `{data['active_positions']}`\n"
        )
    else:
        return

    send_telegram_alert(msg)

if __name__ == "__main__":
    # 測試本地推播
    send_telegram_alert("🔔 *Notifier 模組優化測試完成！*")