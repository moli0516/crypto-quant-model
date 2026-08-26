import os
import json
import logging
import urllib.request
import asyncio
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 無 GUI 環境（如 EC2）繪圖必備
import matplotlib.pyplot as plt
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from dotenv import load_dotenv
import sys

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
STATE_FILE = "logs/paper_account_state.json"
TRADES_FILE = "logs/paper_trades.csv"
EQUITY_FILE = "logs/paper_equity_daily.csv"
DIAGNOSTIC_IMG = "logs/diagnostic_report.png"

# ----------------------------------------------------
# 輔助函式
# ----------------------------------------------------
def get_binance_prices() -> dict:
    url = "https://api.binance.com/api/v3/ticker/price"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return {item['symbol']: float(item['price']) for item in data}
    except Exception as e:
        logger.error(f"抓取幣安現價失敗: {e}")
        return {}

def calculate_live_equity(state: dict, live_prices: dict) -> tuple[float, float]:
    """
    實時計算：Total Equity = Cash + 每一筆持倉的現值 (qty * live_price)
    回傳: (live_total_equity, total_unrealized_pnl)
    """
    cash = state.get("cash", 0.0)
    positions = state.get("open_positions", [])
    
    current_positions_value = 0.0
    total_unrealized_pnl = 0.0
    
    for pos in positions:
        symbol = pos['symbol']
        entry_price = pos['entry_price']
        qty = pos['qty']
        margin = pos['margin']
        
        current_price = live_prices.get(symbol, entry_price)
        current_value = qty * current_price
        pnl_usd = current_value - margin
        
        current_positions_value += current_value
        total_unrealized_pnl += pnl_usd
        
    live_total_equity = cash + current_positions_value
    return live_total_equity, total_unrealized_pnl

def build_history_page(df: pd.DataFrame, page: int = 0, page_size: int = 5):
    """生成 /history 的分頁訊息文字與按鈕矩陣"""
    total_trades = len(df)
    total_pages = (total_trades + page_size - 1) // page_size
    page = max(0, min(page, total_pages - 1))

    # 按時間倒序排列（最新的在最上面）
    df_reversed = df.iloc[::-1].reset_index(drop=True)
    start_idx = page * page_size
    end_idx = start_idx + page_size
    page_df = df_reversed.iloc[start_idx:end_idx]

    msg = f"📜 *[歷史平倉紀錄 (第 {page + 1}/{total_pages} 頁)]*\n-----------------------------------\n"
    for _, row in page_df.iterrows():
        pnl_emoji = "🟢" if row['pnl_usd'] >= 0 else "🔴"
        msg += (
            f"{pnl_emoji} *{row['symbol']}* | PnL: `${row['pnl_usd']:+.2f}` (`{row['pnl_pct']*100:+.2f}%`)\n"
            f"  └ 進: `${row['entry_price']:.4f}` ➔ 出: `${row['exit_price']:.4f}`\n"
        )

    buttons = []
    if page > 0:
        buttons.append(InlineKeyboardButton("⬅️ 上一頁", callback_data=f"hist_page_{page - 1}"))
    if page < total_pages - 1:
        buttons.append(InlineKeyboardButton("下一頁 ➡️", callback_data=f"hist_page_{page + 1}"))

    reply_markup = InlineKeyboardMarkup([buttons]) if buttons else None
    return msg, reply_markup

# ----------------------------------------------------
# 指令處理邏輯
# ----------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 *Quant Trading Bot 已連線*\n\n"
        "可用指令如下：\n"
        "• /status - 檢視帳戶總覽 (實時現價計算)\n"
        "• /positions - 檢視持倉與即時損益\n"
        "• /history - 查看歷史平倉紀錄 (支援分頁)\n"
        "• /report - 生成資產淨值走勢圖\n"
        "• /diag - 執行模型診斷並生成 Diagnostic 圖表\n"
        "• /perf - 計算策略勝率與期望值\n"
        "• /help - 顯示此選單"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(STATE_FILE):
        await update.message.reply_text("❌ 找不到帳戶狀態檔。")
        return

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        live_prices = get_binance_prices()
        live_equity, unrealized_pnl = calculate_live_equity(state, live_prices)

        pos_count = len(state.get("open_positions", []))
        init_bal = state.get("initial_balance", 200.0)
        ret_pct = ((live_equity - init_bal) / init_bal) * 100
        pnl_emoji = "🟢" if unrealized_pnl >= 0 else "🔴"

        msg = (
            f"📊 *[帳戶資產總覽 (實時更新)]*\n"
            f"-----------------------------------\n"
            f"• *初始資金*: `${init_bal:.2f} USD`\n"
            f"• *實時總資產 (Equity)*: `${live_equity:.2f} USD`\n"
            f"• *可用現金 (Cash)*: `${state.get('cash', 0.0):.2f} USD`\n"
            f"• *浮動未實現損益*: {pnl_emoji} `${unrealized_pnl:+.2f} USD`\n"
            f"• *實時累計報酬率*: `{ret_pct:+.2f}%`\n"
            f"• *當前持倉數*: `{pos_count}` 筆\n"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ 讀取數據失敗: {e}")

async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

        live_prices = get_binance_prices()
        total_unrealized_pnl = 0.0

        msg = f"📦 *[當前持倉細節 ({len(positions)} 筆)]*\n-----------------------------------\n"
        
        for pos in positions:
            symbol = pos['symbol']
            entry_price = pos['entry_price']
            qty = pos['qty']
            margin = pos['margin']
            
            current_price = live_prices.get(symbol, entry_price)
            current_value = qty * current_price
            pnl_usd = current_value - margin
            pnl_pct = (pnl_usd / margin) * 100 if margin > 0 else 0.0
            total_unrealized_pnl += pnl_usd

            pnl_emoji = "🟢" if pnl_usd >= 0 else "🔴"

            msg += (
                f"• *{symbol}*\n"
                f"  ├ 進場價: `${entry_price:.4f}` ➔ *現價*: `${current_price:.4f}`\n"
                f"  ├ 浮動損益: {pnl_emoji} `${pnl_usd:+.2f} USD` (`{pnl_pct:+.2f}%`)\n"
                f"  └ 保證金: `${margin:.2f} USD` | 機率: `{pos['prob']*100:.2f}%`\n"
            )

        total_emoji = "🟢" if total_unrealized_pnl >= 0 else "🔴"
        msg += f"-----------------------------------\n"
        msg += f"💰 *預估總未實現損益*: {total_emoji} `${total_unrealized_pnl:+.2f} USD`\n"

        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ 讀取持倉失敗: {e}")

async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(TRADES_FILE):
        await update.message.reply_text("❌ 尚無任何平倉紀錄。")
        return

    try:
        df = pd.read_csv(TRADES_FILE)
        if df.empty:
            await update.message.reply_text("❌ 尚無任何平倉紀錄。")
            return

        msg, reply_markup = build_history_page(df, page=0)
        await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=reply_markup)
    except Exception as e:
        await update.message.reply_text(f"❌ 讀取歷史數據失敗: {e}")

async def history_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理 /history 下方按鈕點擊事件 (Inline Keyboard Callback)"""
    query = update.callback_query
    await query.answer()

    if not os.path.exists(TRADES_FILE):
        return

    try:
        target_page = int(query.data.split("_")[-1])
        df = pd.read_csv(TRADES_FILE)
        if df.empty:
            return

        msg, reply_markup = build_history_page(df, page=target_page)
        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"切換分頁失敗: {e}")

async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(EQUITY_FILE):
        await update.message.reply_text("❌ 找不到資產歷史紀錄。")
        return

    try:
        df = pd.read_csv(EQUITY_FILE)
        if len(df) < 2:
            await update.message.reply_text("⚠️ 數據點不足，尚無法繪製走勢圖。")
            return

        plt.figure(figsize=(9, 4.5))
        plt.plot(df['timestamp'], df['total_equity'], marker='o', color='#2b5c8f', linewidth=2)
        plt.axhline(y=200, color='r', linestyle='--', alpha=0.6, label='Initial Balance ($200)')
        plt.title("Paper Account Equity Curve", fontsize=12)
        plt.xlabel("Timestamp", fontsize=9)
        plt.ylabel("Total Equity ($)", fontsize=9)
        plt.xticks(rotation=30, ha='right', fontsize=8)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        img_path = "logs/equity_chart.png"
        plt.savefig(img_path, dpi=150)
        plt.close()

        with open(img_path, "rb") as photo:
            await update.message.reply_photo(photo=photo, caption="📈 *[資產淨值走勢圖 (Equity Curve)]*", parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"❌ 繪製圖表失敗: {e}")

async def diag_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("🔬 正在執行模型診斷分析腳本 (scripts.analyst_log)...")
    
    try:
        python_executable = sys.executable

        process = await asyncio.create_subprocess_exec(
            python_executable, '-m', 'scripts.analyst_log',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"Analyst Log 腳本執行錯誤: {stderr.decode()}")
            await status_msg.edit_text(f"❌ 診斷腳本執行失敗:\n`{stderr.decode()[:200]}`", parse_mode="Markdown")
            return

        if not os.path.exists(DIAGNOSTIC_IMG):
            await status_msg.edit_text("❌ 腳本執行完畢，但未找到 `logs/diagnostic.png` 圖表檔。")
            return

        await status_msg.delete()
        with open(DIAGNOSTIC_IMG, "rb") as photo:
            await update.message.reply_photo(
                photo=photo,
                caption="🔬 *[模型診斷與交易分析圖表 (Diagnostic Report)]*",
                parse_mode="Markdown"
            )

    except Exception as e:
        await update.message.reply_text(f"❌ 執行診斷失敗: {e}")

async def perf_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(TRADES_FILE):
        await update.message.reply_text("❌ 尚無足夠平倉數據。")
        return

    try:
        df = pd.read_csv(TRADES_FILE)
        total_trades = len(df)
        if total_trades == 0:
            await update.message.reply_text("❌ 尚無平倉紀錄。")
            return

        wins = df[df['pnl_usd'] > 0]
        win_rate = (len(wins) / total_trades) * 100
        total_pnl = df['pnl_usd'].sum()
        avg_pnl = df['pnl_usd'].mean()

        msg = (
            f"📈 *[策略績效統計]*\n"
            f"-----------------------------------\n"
            f"• *總平倉筆數*: `{total_trades}` 筆\n"
            f"• *勝率 (Win Rate)*: `{win_rate:.1f}%`\n"
            f"• *累計實現損益*: `${total_pnl:+.2f} USD`\n"
            f"• *單筆平均損益*: `${avg_pnl:+.2f} USD`\n"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ 計算績效失敗: {e}")
        
async def sltp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """執行 1m 高頻 K 線 SL/TP 網格碰撞回測，並回傳診斷圖表 (sltp_diagnostic_report.png)"""
    status_msg = await update.message.reply_text("⚡ 正在執行 1m 高頻 K 線 SL/TP 網格碰撞回測與診斷分析 (scripts.sltp_backtest)...")
    
    try:
        python_executable = sys.executable

        # 非同步子進程執行 scripts.sltp_backtest
        process = await asyncio.create_subprocess_exec(
            python_executable, '-m', 'scripts.sltp_backtest',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"SLTP Backtest 執行錯誤: {stderr.decode()}")
            await status_msg.edit_text(f"❌ SL/TP 回測腳本執行失敗:\n`{stderr.decode()[:200]}`", parse_mode="Markdown")
            return

        report_img = "local-logs/sltp_diagnostic_report.png"
        if not os.path.exists(report_img):
            await status_msg.edit_text("❌ 腳本執行完畢，但未找到 `local-logs/sltp_diagnostic_report.png` 圖表檔。")
            return

        await status_msg.delete()
        with open(report_img, "rb") as photo:
            await update.message.reply_photo(
                photo=photo,
                caption="🏆 *[SL/TP 網格碰撞回測診斷報告 (Heatmap & Equity)]*",
                parse_mode="Markdown"
            )

    except Exception as e:
        await update.message.reply_text(f"❌ 執行 SL/TP 回測失敗: {e}")

# ----------------------------------------------------
# 設定快捷選單與啟動
# ----------------------------------------------------
async def post_init(application: Application):
    commands = [
        BotCommand("status", "檢視帳戶資產總覽 (實時價格)"),
        BotCommand("positions", "檢視持倉與即時損益"),
        BotCommand("history", "查看歷史平倉紀錄 (按鈕分頁)"),
        BotCommand("report", "生成資產淨值走勢圖"),
        BotCommand("diag", "生成 Diagnostic 分析診斷圖"),
        BotCommand("perf", "計算策略勝率與期望值"),
        BotCommand("help", "顯示指令說明"),
        BotCommand("sltp", "執行 1m K 線 SL/TP 網格碰撞回測"),
    ]
    await application.bot.set_my_commands(commands)

def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ 未設定 TELEGRAM_BOT_TOKEN！")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("positions", positions_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("report", report_command))
    app.add_handler(CommandHandler("diag", diag_command))
    app.add_handler(CommandHandler("perf", perf_command))
    app.add_handler(CommandHandler("sltp", sltp_command))

    # 註冊歷史紀錄分頁按鈕的回調處理器
    app.add_handler(CallbackQueryHandler(history_page_callback, pattern=r"^hist_page_"))

    logger.info("🤖 Telegram Bot 指令監聽器啟動中 (Long Polling)...")
    app.run_polling()

if __name__ == "__main__":
    print(TELEGRAM_BOT_TOKEN)
    main()