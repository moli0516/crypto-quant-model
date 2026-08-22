import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# Telegram Bot Configuration
# 請在此填入你的 Token 與 Chat ID
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(message: str):
    """Send notification via Telegram API."""
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN":
        print(f"[MOCK NOTIFIER] 偵測到訊號，但尚未設定 Token，終端機輸出:\n{message}")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print("[NOTIFIER] Telegram 提醒發送成功。")
        else:
            print(f"[NOTIFIER] Telegram 發送失敗: {response.text}")
    except Exception as e:
        print(f"[NOTIFIER] 傳送提醒時發生錯誤: {e}")

def check_and_notify(csv_path: str = "logs/inference_history.csv"):
    """Check the latest inference logs for LONG signals."""
    if not os.path.exists(csv_path):
        print(f"[NOTIFIER] 找不到檔案 {csv_path}。")
        return
    
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    
    # 取得最新一筆的時間戳記
    latest_ts = df["timestamp"].max()
    latest_df = df[df["timestamp"] == latest_ts]
    
    # 篩選出大於等於 0.55 的 LONG 訊號
    long_signals = latest_df
    
    if not long_signals.empty:
        msg = f"🚀 *QUANT SIGNAL DETECTED*\n*Time:* `{latest_ts}`\n\n"
        for _, row in long_signals.iterrows():
            msg += f"• *{row['symbol']}* | Price: `${row['close_price']:.4f}` | Prob: `{row['pred_proba']:.4f}`\n"
        send_telegram_alert(msg)
    else:
        print(f"[NOTIFIER] {latest_ts} 訊號檢查完畢: 目前維持 WAIT 觀望。")

if __name__ == "__main__":
    check_and_notify()
