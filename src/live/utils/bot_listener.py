import os
import json
import logging
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
STATE_FILE = "logs/paper_account_state.json"
TRADES_FILE = "logs/paper_trades.csv"

# ----------------------------------------------------
# 指令處理邏輯
# ----------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 *Quant Trading Bot 已連線*\n\n"
        "可用指令如下：\n"
        "• /status - 檢視帳戶總覽\n"
        "• /positions - 檢視當前持倉\n"
        "• /history - 查看歷史平倉紀錄\n"
        "• /help - 顯示此選單"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """讀取 paper_account_state.json 渲染帳戶總覽"""
    if not os.path.exists(STATE_FILE):
        await update.message.reply_text("❌ 找不到帳戶狀態檔，可能尚未執行任何週期。")
        return

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        pos_count = len(state.get("open_positions", []))
        init_bal = state.get("initial_balance", 200.0)
        total_eq = state.get("total_equity", init_bal)
        ret_pct = ((total_eq - init_bal) / init_bal) * 100

        msg = (
            f"📊 *[帳戶資產總覽]*\n"
            f"-----------------------------------\n"
            f"• *初始資金*: `${init_bal:.2f} USD`\n"
            f"• *總資產 (Equity)*: `${total_eq:.2f} USD`\n"
            f"• *可用現金 (Cash)*: `${state.get('cash', 0.0):.2f} USD`\n"
            f"• *累計報酬率*: `{ret_pct:+.2f}%`\n"
            f"• *當前持倉數*: `{pos_count}` 筆\n"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ 讀取數據失敗: {e}")

async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """讀取當前持倉細節"""
    if not os.path.exists(STATE_FILE):
        await update.message.reply_text("❌ 找不到持倉資料。")
        return

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        positions = state.get("open_positions", [])
        if not positions:
            await update.message.reply_text("🟢 當前無任何在場持倉 (All Cash)。")
            return

        msg = f"📦 *[當前持倉細節 ({len(positions)} 筆)]*\n-----------------------------------\n"
        for pos in positions:
            msg += (
                f"• *{pos['symbol']}*\n"
                f"  ├ 進場價: `${pos['entry_price']:.4f}`\n"
                f"  ├ 保證金: `${pos['margin']:.2f} USD` | 機率: `{pos['prob']*100:.2f}%`\n"
                f"  └ 時間: `{pos['entry_time']}`\n"
            )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ 讀取持倉失敗: {e}")

async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """讀取 paper_trades.csv 最近 5 筆紀錄"""
    if not os.path.exists(TRADES_FILE):
        await update.message.reply_text("❌ 尚無任何平倉紀錄。")
        return

    try:
        df = pd.read_csv(TRADES_FILE)
        if df.empty:
            await update.message.reply_text("❌ 尚無任何平倉紀錄。")
            return

        recent_df = df.tail(5)
        msg = "📜 *[最近 5 筆平倉歷史]*\n-----------------------------------\n"
        for _, row in recent_df.iterrows():
            pnl_emoji = "🟢" if row['pnl_usd'] >= 0 else "🔴"
            msg += (
                f"{pnl_emoji} *{row['symbol']}* | PnL: `${row['pnl_usd']:+.2f}` (`{row['pnl_pct']*100:+.2f}%`)\n"
                f"  └ 進: `${row['entry_price']:.4f}` ➔ 出: `${row['exit_price']:.4f}`\n"
            )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ 讀取歷史數據失敗: {e}")

# ----------------------------------------------------
# 機器人啟動入口
# ----------------------------------------------------
def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ 未設定 TELEGRAM_BOT_TOKEN！")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("positions", positions_command))
    app.add_handler(CommandHandler("history", history_command))

    logger.info("🤖 Telegram Bot 指令監聽器啟動中 (Long Polling)...")
    app.run_polling()

if __name__ == "__main__":
    main()