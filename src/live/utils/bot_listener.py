"""
src/live/utils/bot_listener.py
==============================================================================
Institutional-Grade Telegram Bot Command Listener & Diagnostic Hub
全面對齊 Phase 3 Ensemble + Spot OCO 架構，支援實時資產、按鈕分頁與暗黑版走勢圖推播。
==============================================================================
"""

import os
import json
import logging
import urllib.request
import asyncio
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 無 GUI 環境 (如 EC2) 繪圖必備
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from dotenv import load_dotenv
import sys

# 🎯 統一從集中式配置層導入參數路徑
from src.config import (
    TELEGRAM_BOT_TOKEN,
    STATE_FILE,
    TRADES_LOG_FILE,
    EQUITY_LOG_FILE,
    DIAGNOSTIC_IMG_FILE,
    REPORT_IMG_FILE,
    SLTP_REPORT_IMG_FILE,
    INITIAL_CAPITAL
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================================================================
# 全局 Institutional Dark Theme 配置 (暗黑專業風格)
# =========================================================================
plt.style.use("dark_background")
PLT_STYLE_CONFIG = {
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#2C303E",
    "axes.linewidth": 1.2,
    "grid.color": "#2C303E",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "figure.facecolor": "#12141D",
    "axes.facecolor": "#1A1D29",
    "text.color": "#E0E6ED",
    "axes.labelcolor": "#A0AEC0",
    "xtick.color": "#A0AEC0",
    "ytick.color": "#A0AEC0",
}
plt.rcParams.update(PLT_STYLE_CONFIG)


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
    實時計算：Total Equity = Cash + 每一筆現貨持倉的現值 (qty * live_price)
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
        margin = pos.get('margin', qty * entry_price)
        
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
        pnl_usd = row.get('pnl_usd', 0.0)
        pnl_pct = row.get('pnl_pct', 0.0)
        pnl_emoji = "🟢" if pnl_usd >= 0 else "🔴"
        msg += (
            f"{pnl_emoji} *{row['symbol']}* | PnL: `${pnl_usd:+.2f}` (`{pnl_pct*100:+.2f}%`)\n"
            f"  └ 進: `${row['entry_price']:.4f}` ➔ 出: `${row['exit_price']:.4f}`\n"
        )

    buttons = []
    if page > 0:
        buttons.append(InlineKeyboardButton("⬅️ 上一頁", callback_data=f"hist_page_{page - 1}"))
    if page < total_pages - 1:
        buttons.append(InlineKeyboardButton("下一頁 ➡️", callback_data=f"hist_page_{page + 1}"))

    reply_markup = InlineKeyboardMarkup([buttons]) if buttons else None
    return msg, reply_markup

def generate_smooth_equity_chart(initial_balance: float = INITIAL_CAPITAL) -> str:
    """
    [核心優化]: 繪製機構級平滑資產淨值走勢圖 (Equity Curve)
    優先讀取 paper_trades.csv (已實現損益)，若不足則自動降級讀取 paper_equity_daily.csv。
    """
    df_plot = pd.DataFrame()

    if os.path.exists(TRADES_LOG_FILE):
        try:
            df_trades = pd.read_csv(TRADES_LOG_FILE)
            if not df_trades.empty and 'pnl_usd' in df_trades.columns:
                time_col = 'close_time' if 'close_time' in df_trades.columns else ('exit_time' if 'exit_time' in df_trades.columns else 'timestamp')
                df_trades['dt'] = pd.to_datetime(df_trades[time_col])
                df_trades = df_trades.sort_values('dt').reset_index(drop=True)
                
                df_trades['cum_pnl'] = df_trades['pnl_usd'].cumsum()
                df_trades['equity'] = initial_balance + df_trades['cum_pnl']
                
                start_dt = df_trades['dt'].iloc[0] - pd.Timedelta(hours=1)
                df_plot = pd.concat([
                    pd.DataFrame([{'dt': start_dt, 'equity': initial_balance}]),
                    df_trades[['dt', 'equity']]
                ], ignore_index=True)
        except Exception as e:
            logger.error(f"解析 trades log 失敗: {e}")

    if df_plot.empty and os.path.exists(EQUITY_LOG_FILE):
        try:
            df_eq = pd.read_csv(EQUITY_LOG_FILE)
            if len(df_eq) >= 2:
                df_eq['dt'] = pd.to_datetime(df_eq['timestamp'])
                df_eq = df_eq.sort_values('dt').reset_index(drop=True)
                df_eq['equity'] = df_eq['total_equity']
                df_plot = df_eq[['dt', 'equity']]
        except Exception as e:
            logger.error(f"解析 equity log 失敗: {e}")

    if len(df_plot) < 2:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    ax.axhline(
        y=initial_balance, 
        color="#E53E3E", 
        linestyle="--", 
        linewidth=1.2, 
        alpha=0.7, 
        label=f"Initial Capital (${initial_balance:,.2f})"
    )

    ax.plot(
        df_plot["dt"], 
        df_plot["equity"], 
        color="#3182CE", 
        linewidth=2.2, 
        label="Account Equity ($)", 
        zorder=3
    )
    
    ax.fill_between(
        df_plot["dt"], df_plot["equity"], initial_balance, 
        where=(df_plot["equity"] >= initial_balance),
        interpolate=True, color="#38A169", alpha=0.18, zorder=2
    )
    ax.fill_between(
        df_plot["dt"], df_plot["equity"], initial_balance, 
        where=(df_plot["equity"] < initial_balance),
        interpolate=True, color="#E53E3E", alpha=0.18, zorder=2
    )

    ax.scatter(
        df_plot["dt"].iloc[1:], df_plot["equity"].iloc[1:], 
        color="#63B3ED", s=25, edgecolor="#1A1D29", linewidth=0.8, zorder=4
    )

    latest_eq = df_plot["equity"].iloc[-1]
    ax.annotate(
        f" Current: ${latest_eq:,.2f}",
        xy=(df_plot["dt"].iloc[-1], latest_eq),
        xytext=(8, 0), textcoords="offset points",
        color="#63B3ED", weight="bold", fontsize=9, va="center"
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=20, ha="right")

    eq_min, eq_max = df_plot["equity"].min(), df_plot["equity"].max()
    margin = max((eq_max - eq_min) * 0.15, 2.0)
    ax.set_ylim(eq_min - margin, eq_max + margin)

    ax.set_title("Spot Testnet Account Equity Curve (Realized Progress)", fontsize=13, pad=12, weight="bold", color="#F7FAFC")
    ax.set_xlabel("Timestamp", fontsize=9, labelpad=8)
    ax.set_ylabel("Total Equity ($)", fontsize=9, labelpad=8)
    ax.grid(True)
    ax.legend(loc="upper left", frameon=True, facecolor="#1A1D29", edgecolor="#2C303E")

    os.makedirs(os.path.dirname(REPORT_IMG_FILE), exist_ok=True)
    plt.tight_layout()
    plt.savefig(REPORT_IMG_FILE, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

    return REPORT_IMG_FILE


# ----------------------------------------------------
# Telegram 指令處理邏輯
# ----------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🤖 *Ensemble Spot Quant Bot 已連線*\n\n"
        "可用指令如下：\n"
        "• /status - 檢視現貨帳戶總覽 (即時價格計算)\n"
        "• /positions - 檢視現貨持倉與 OCO 掛單損益\n"
        "• /history - 查看歷史平倉紀錄 (支援按鈕分頁)\n"
        "• /report - 生成暗黑版資產淨值走勢圖 (Equity Curve)\n"
        "• /diag - 執行模型診斷並生成 2x2 報表\n"
        "• /perf - 計算策略勝率與期望值\n"
        "• /sltp - 執行 1m K 線 SL/TP 網格碰撞回測\n"
        "• /help - 顯示此選單"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(STATE_FILE):
        await update.message.reply_text("❌ 找不到帳戶狀態檔 (`paper_account_state.json`)。")
        return

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        live_prices = get_binance_prices()
        live_equity, unrealized_pnl = calculate_live_equity(state, live_prices)

        pos_count = len(state.get("open_positions", []))
        init_bal = INITIAL_CAPITAL
        ret_pct = ((live_equity - init_bal) / init_bal) * 100
        pnl_emoji = "🟢" if unrealized_pnl >= 0 else "🔴"

        msg = (
            f"📊 *[Binance Spot Testnet 總覽]*\n"
            f"-----------------------------------\n"
            f"• *初始資金*: `${init_bal:,.2f} USD`\n"
            f"• *實時總資產 (Equity)*: `${live_equity:,.2f} USD`\n"
            f"• *可用現金 (Cash)*: `${state.get('cash', 0.0):,.2f} USD`\n"
            f"• *浮動未實現損益*: {pnl_emoji} `${unrealized_pnl:+,.2f} USD`\n"
            f"• *實時累計報酬率*: `{ret_pct:+.2f}%`\n"
            f"• *當前持倉數*: `{pos_count}` 筆\n"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ 讀取帳戶狀態失敗: {e}")

async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(STATE_FILE):
        await update.message.reply_text("❌ 找不到持倉資料。")
        return

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        positions = state.get("open_positions", [])
        if not positions:
            await update.message.reply_text("🟢 當前無任何現貨持倉 (All Cash)。")
            return

        live_prices = get_binance_prices()
        total_unrealized_pnl = 0.0

        msg = f"📦 *[當前現貨持倉細節 ({len(positions)} 筆)]*\n-----------------------------------\n"
        
        for pos in positions:
            symbol = pos['symbol']
            entry_price = pos['entry_price']
            qty = pos['qty']
            margin = pos.get('margin', qty * entry_price)
            tp_price = pos.get('tp_price', 0.0)
            sl_price = pos.get('sl_price', 0.0)
            
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
                f"  ├ OCO 掛單 | TP: `${tp_price:.4f}` | SL: `${sl_price:.4f}`\n"
                f"  └ 投入金額: `${margin:.2f} USD` | 機率: `{pos.get('prob', 0)*100:.2f}%`\n\n"
            )

        total_emoji = "🟢" if total_unrealized_pnl >= 0 else "🔴"
        msg += f"-----------------------------------\n"
        msg += f"💰 *預估總未實現損益*: {total_emoji} `${total_unrealized_pnl:+.2f} USD`\n"

        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ 讀取持倉失敗: {e}")

async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(TRADES_LOG_FILE):
        await update.message.reply_text("❌ 尚無任何平倉紀錄。")
        return

    try:
        df = pd.read_csv(TRADES_LOG_FILE)
        if df.empty:
            await update.message.reply_text("❌ 尚無任何平倉紀錄。")
            return

        msg, reply_markup = build_history_page(df, page=0)
        await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=reply_markup)
    except Exception as e:
        await update.message.reply_text(f"❌ 讀取歷史數據失敗: {e}")

async def history_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理 /history 下方按鈕點擊事件"""
    query = update.callback_query
    await query.answer()

    if not os.path.exists(TRADES_LOG_FILE):
        return

    try:
        target_page = int(query.data.split("_")[-1])
        df = pd.read_csv(TRADES_LOG_FILE)
        if df.empty:
            return

        msg, reply_markup = build_history_page(df, page=target_page)
        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"切換分頁失敗: {e}")

async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("📈 正在生成平滑暗黑版資產淨值走勢圖 (Equity Curve)...")

    try:
        img_path = await asyncio.to_thread(generate_smooth_equity_chart, INITIAL_CAPITAL)

        if not img_path or not os.path.exists(img_path):
            await status_msg.edit_text("⚠️ 數據點不足（需至少 2 筆平倉紀錄），尚無法繪製走勢圖。")
            return

        await status_msg.delete()
        with open(img_path, "rb") as photo:
            await update.message.reply_photo(
                photo=photo, 
                caption="📈 *[資產淨值走勢圖 (Realized Equity Curve)]*", 
                parse_mode="Markdown"
            )
    except Exception as e:
        logger.error(f"繪製 report 圖表失敗: {e}")
        await status_msg.edit_text(f"❌ 繪製圖表失敗: {e}")

async def diag_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("🔬 正在執行模型診斷分析腳本 (scripts.analyst_log)...")
    
    try:
        python_executable = sys.executable
        process = await asyncio.create_subprocess_exec(
            python_executable, '-m', 'scripts.analyst_log',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"Analyst Log 腳本執行錯誤: {stderr.decode()}")
            await status_msg.edit_text(f"❌ 診斷腳本執行失敗:\n`{stderr.decode()[:200]}`", parse_mode="Markdown")
            return

        if not os.path.exists(DIAGNOSTIC_IMG_FILE):
            await status_msg.edit_text("❌ 腳本執行完畢，但未找到診斷圖表檔。")
            return

        await status_msg.delete()
        with open(DIAGNOSTIC_IMG_FILE, "rb") as photo:
            await update.message.reply_photo(
                photo=photo,
                caption="🔬 *[模型診斷與交易分析圖表 (Diagnostic Report)]*",
                parse_mode="Markdown"
            )
    except Exception as e:
        await update.message.reply_text(f"❌ 執行診斷失敗: {e}")

async def perf_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(TRADES_LOG_FILE):
        await update.message.reply_text("❌ 尚無足夠平倉數據。")
        return

    try:
        df = pd.read_csv(TRADES_LOG_FILE)
        total_trades = len(df)
        if total_trades == 0:
            await update.message.reply_text("❌ 尚無平倉紀錄。")
            return

        pnl_col = 'pnl_usd' if 'pnl_usd' in df.columns else 'pnl_amount'
        wins = df[df[pnl_col] > 0]
        win_rate = (len(wins) / total_trades) * 100
        total_pnl = df[pnl_col].sum()
        avg_pnl = df[pnl_col].mean()

        msg = (
            f"📈 *[策略績效統計]*\n"
            f"-----------------------------------\n"
            f"• *總平倉筆數*: `{total_trades}` 筆\n"
            f"• *勝率 (Win Rate)*: `{win_rate:.1f}%`\n"
            f"• *累計實現損益*: `${total_pnl:+,.2f} USD`\n"
            f"• *單筆平均損益*: `${avg_pnl:+,.2f} USD`\n"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ 計算績效失敗: {e}")
        
async def sltp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("⚡ 正在執行 1m 高頻 K 線 SL/TP 網格碰撞回測與診斷分析 (scripts.sltp_backtest)...")
    
    try:
        python_executable = sys.executable
        process = await asyncio.create_subprocess_exec(
            python_executable, '-m', 'scripts.sltp_backtest',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"SLTP Backtest 執行錯誤: {stderr.decode()}")
            await status_msg.edit_text(f"❌ SL/TP 回測腳本執行失敗:\n`{stderr.decode()[:200]}`", parse_mode="Markdown")
            return

        if not os.path.exists(SLTP_REPORT_IMG_FILE):
            await status_msg.edit_text("❌ 腳本執行完畢，但未找到 SLTP 回測圖表檔。")
            return

        await status_msg.delete()
        with open(SLTP_REPORT_IMG_FILE, "rb") as photo:
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
        BotCommand("status", "檢視現貨帳戶資產總覽 (實時價格)"),
        BotCommand("positions", "檢視現貨持倉與 OCO 掛單損益"),
        BotCommand("history", "查看歷史平倉紀錄 (按鈕分頁)"),
        BotCommand("report", "生成暗黑版資產淨值走勢圖"),
        BotCommand("diag", "生成 Diagnostic 分析診斷圖"),
        BotCommand("perf", "計算策略勝率與期望值"),
        BotCommand("sltp", "執行 1m K 線 SL/TP 網格碰撞回測"),
        BotCommand("help", "顯示指令說明"),
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

    app.add_handler(CallbackQueryHandler(history_page_callback, pattern=r"^hist_page_"))

    logger.info("🤖 Telegram Bot 指令監聽器啟動中 (Long Polling)...")
    app.run_polling()

if __name__ == "__main__":
    main()