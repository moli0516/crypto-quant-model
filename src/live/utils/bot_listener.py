"""Telegram monitoring commands for Binance Spot Testnet trading."""

import asyncio
import json
import logging
import os
import subprocess
import sys
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from src.config import (
    DIAGNOSTIC_IMG_FILE,
    EQUITY_LOG_FILE,
    INITIAL_CAPITAL,
    REPORT_IMG_FILE,
    SLTP_REPORT_IMG_FILE,
    STATE_FILE,
    TELEGRAM_BOT_TOKEN,
    TRADES_LOG_FILE,
)
from src.live.binance_spot_trader import BinanceSpotTrader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

plt.style.use("dark_background")
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#2C303E", "axes.linewidth": 1.2,
    "grid.color": "#2C303E", "grid.linestyle": "--", "grid.alpha": 0.5,
    "figure.facecolor": "#12141D", "axes.facecolor": "#1A1D29",
    "text.color": "#E0E6ED", "axes.labelcolor": "#A0AEC0",
    "xtick.color": "#A0AEC0", "ytick.color": "#A0AEC0",
})


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_orders() -> List[Dict[str, Any]]:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as state_file:
            data = json.load(state_file)
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to read order state: %s", exc)
        return []


def open_orders() -> List[Dict[str, Any]]:
    return [order for order in load_orders() if order.get("status") == "OPEN"]


def closed_orders() -> List[Dict[str, Any]]:
    return [order for order in load_orders() if order.get("status") == "CLOSED"]


def get_binance_prices() -> Dict[str, float]:
    request = urllib.request.Request(
        "https://api.binance.com/api/v3/ticker/price",
        headers={"User-Agent": "crypto-quant-telegram-listener"},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {
            str(item["symbol"]).upper(): _number(item["price"])
            for item in payload
            if isinstance(item, dict) and item.get("symbol") and item.get("price")
        } if isinstance(payload, list) else {}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Unable to fetch Binance prices: %s", exc)
        return {}


async def get_usdt_balance() -> Optional[Dict[str, float]]:
    trader = None
    try:
        trader = BinanceSpotTrader()
        return await trader.get_usdt_balance()
    except Exception as exc:
        logger.warning("Unable to fetch Testnet USDT balance: %s", exc)
        return None
    finally:
        if trader is not None:
            await trader.close()


async def get_account_snapshot(prices: Dict[str, float]) -> Optional[Dict[str, Any]]:
    trader = None
    try:
        trader = BinanceSpotTrader()
        return await trader.get_account_snapshot(prices)
    except Exception as exc:
        logger.exception("Unable to fetch Testnet account snapshot: %s", exc)
        return None
    finally:
        if trader is not None:
            await trader.close()


def order_pnl(order: Dict[str, Any], prices: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
    entry = _number(order.get("entry_price"))
    quantity = _number(order.get("qty"))
    margin = _number(order.get("margin_usd"), quantity * entry)
    if order.get("status") == "CLOSED":
        stored_pnl = _number(order.get("pnl_usd"), float("nan"))
        if stored_pnl == stored_pnl:
            pnl = stored_pnl
        else:
            exit_quantity = _number(order.get("exit_qty"), quantity)
            exit_price = _number(order.get("exit_price"), entry)
            fees = _number(order.get("fee_usd"))
            pnl = (exit_price - entry) * exit_quantity - fees
    else:
        symbol = str(order.get("symbol", "")).upper()
        current = (prices or {}).get(symbol, entry)
        pnl = quantity * current - margin
    return pnl, (pnl / margin * 100.0) if margin else 0.0


def _money(value: float) -> str:
    return f"${value:+,.2f}"


def _message(update: Update) -> Any:
    return update.effective_message


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    await _message(update).reply_text(
        "🤖 *Ensemble Spot Quant Bot*\n\n"
        "/status - Binance 餘額與即時持倉總覽\n"
        "/positions - OPEN OCO 持倉細節\n"
        "/history - CLOSED 歷史訂單\n"
        "/report - 已實現損益 Equity Curve\n"
        "/diag - 模型診斷圖\n"
        "/perf - 勝率與累計損益\n"
        "/sltp - SL/TP 回測圖\n"
        "/help - 顯示本選單",
        parse_mode="Markdown",
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = _message(update)
    orders = open_orders()
    prices = await asyncio.to_thread(get_binance_prices)
    account = await get_account_snapshot(prices)
    if account is None:
        await message.reply_text("❌ 無法取得 Binance Spot Testnet 帳戶餘額。")
        return
    unrealized = sum(order_pnl(order, prices)[0] for order in orders)
    position_value = account["holding_value"]
    equity = account["available_cash"] + position_value
    return_pct = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100 if INITIAL_CAPITAL else 0.0
    mark = "🟢" if unrealized >= 0 else "🔴"
    await message.reply_text(
        "📊 *Binance Spot Testnet 總覽*\n-----------------------------------\n"
        f"• 可用現金 (USDT): `${account['available_cash']:,.2f}`\n"
        f"• USDT 總額: `${account['usdt_total']:,.2f}`\n"
        f"• 實際持倉市值: `${position_value:,.2f}`\n"
        f"• 總權益 (USDT + 持倉): `${equity:,.2f}`\n"
        f"• 浮動未實現損益: {mark} `{_money(unrealized)}`\n"
        f"• 相對初始資金: `{return_pct:+.2f}%`\n"
        f"• OPEN 持倉數: `{len(orders)}`",
        parse_mode="Markdown",
    )


async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = _message(update)
    orders = open_orders()
    if not orders:
        await message.reply_text("🟢 目前沒有 status=OPEN 的持倉。")
        return
    prices = await asyncio.to_thread(get_binance_prices)
    lines = [f"📦 *OPEN OCO 持倉 ({len(orders)} 筆)*", "-----------------------------------"]
    total_pnl = 0.0
    for order in orders:
        symbol = str(order.get("symbol", "UNKNOWN")).upper()
        entry = _number(order.get("entry_price"))
        current = prices.get(symbol, entry)
        pnl, pnl_pct = order_pnl(order, prices)
        total_pnl += pnl
        mark = "🟢" if pnl >= 0 else "🔴"
        lines.append(
            f"• *{symbol}* | 數量 `{_number(order.get('qty')):.8f}`\n"
            f"  進場 `${entry:.8f}` / 現價 `${current:.8f}`\n"
            f"  浮動損益 {mark} `{_money(pnl)}` (`{pnl_pct:+.2f}%`)\n"
            f"  TP `${_number(order.get('tp_price')):.8f}` | SL `${_number(order.get('sl_price')):.8f}`\n"
            f"  保證金 `${_number(order.get('margin_usd')):,.2f}` | 機率 `{_number(order.get('prob')) * 100:.2f}%`"
        )
    lines.append(f"-----------------------------------\n合計浮動損益: `{_money(total_pnl)}`")
    await message.reply_text("\n".join(lines), parse_mode="Markdown")


def build_history_page(
    orders: List[Dict[str, Any]], page: int = 0, page_size: int = 5
) -> Tuple[str, Optional[InlineKeyboardMarkup]]:
    ordered = list(reversed(orders))
    total_pages = max(1, (len(ordered) + page_size - 1) // page_size)
    page = max(0, min(page, total_pages - 1))
    selected = ordered[page * page_size : (page + 1) * page_size]
    lines = [f"📜 *CLOSED 歷史訂單 ({page + 1}/{total_pages})*", "-----------------------------------"]
    for order in selected:
        pnl, pnl_pct = order_pnl(order)
        mark = "🟢" if pnl >= 0 else "🔴"
        lines.append(
            f"{mark} *{str(order.get('symbol', 'UNKNOWN')).upper()}* `{_money(pnl)}` (`{pnl_pct:+.2f}%`)\n"
            f"  進場 `${_number(order.get('entry_price')):.8f}` | 時間 `{order.get('close_time', '-')}`\n"
            f"  原因 `{order.get('close_reason', '-')}`"
        )
    buttons = []
    if page > 0:
        buttons.append(InlineKeyboardButton("⬅️ 上一頁", callback_data=f"hist_page_{page - 1}"))
    if page < total_pages - 1:
        buttons.append(InlineKeyboardButton("下一頁 ➡️", callback_data=f"hist_page_{page + 1}"))
    return "\n".join(lines), InlineKeyboardMarkup([buttons]) if buttons else None


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    orders = closed_orders()
    if not orders:
        await _message(update).reply_text("❌ 尚無 status=CLOSED 歷史訂單。")
        return
    text, markup = build_history_page(orders)
    await _message(update).reply_text(text, parse_mode="Markdown", reply_markup=markup)


async def history_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    query = update.callback_query
    await query.answer()
    try:
        page = int(str(query.data).rsplit("_", 1)[1])
        text, markup = build_history_page(closed_orders(), page)
        await query.edit_message_text(text, parse_mode="Markdown", reply_markup=markup)
    except (ValueError, TypeError) as exc:
        logger.warning("Invalid history page callback: %s", exc)


def _read_realized_records() -> pd.DataFrame:
    try:
        frame = pd.read_csv(TRADES_LOG_FILE)
        if "pnl_usd" not in frame.columns and "pnl_amount" in frame.columns:
            frame["pnl_usd"] = pd.to_numeric(frame["pnl_amount"], errors="coerce")
        time_col = next((col for col in ("close_time", "exit_time", "timestamp") if col in frame.columns), None)
        if time_col and "pnl_usd" in frame.columns:
            frame["dt"] = pd.to_datetime(frame[time_col], errors="coerce")
            realized = frame.dropna(subset=["dt"])
            if not realized.empty:
                return realized
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.warning("Unable to read realized trades: %s", exc)
    records = closed_orders()
    if records:
        frame = pd.DataFrame(records)
        if "pnl_usd" not in frame.columns:
            frame["pnl_usd"] = [order_pnl(order)[0] for order in records]
        frame["dt"] = pd.to_datetime(frame.get("close_time"), errors="coerce")
        return frame.dropna(subset=["dt"])
    return pd.DataFrame()


def generate_equity_chart(initial_balance: float = INITIAL_CAPITAL) -> str:
    realized = _read_realized_records()
    if not realized.empty:
        realized = realized.sort_values("dt").copy()
        realized["pnl_usd"] = pd.to_numeric(realized["pnl_usd"], errors="coerce").fillna(0.0)
        realized["equity"] = initial_balance + realized["pnl_usd"].cumsum()
        start = realized["dt"].iloc[0] - pd.Timedelta(hours=1)
        chart = pd.concat([
            pd.DataFrame({"dt": [start], "equity": [initial_balance]}),
            realized[["dt", "equity"]],
        ], ignore_index=True)
    else:
        try:
            chart = pd.read_csv(EQUITY_LOG_FILE)
            chart["dt"] = pd.to_datetime(chart.get("timestamp"), errors="coerce")
            chart["equity"] = pd.to_numeric(chart.get("total_equity"), errors="coerce")
            chart = chart.dropna(subset=["dt", "equity"])[["dt", "equity"]].sort_values("dt")
        except (OSError, ValueError, pd.errors.ParserError):
            chart = pd.DataFrame()
    if len(chart) < 2:
        return ""
    figure, axis = plt.subplots(figsize=(10, 5), dpi=220)
    axis.axhline(initial_balance, color="#E53E3E", linestyle="--", linewidth=1.2, label="Initial Capital")
    axis.plot(chart["dt"], chart["equity"], color="#3182CE", linewidth=2.2, label="Account Equity")
    axis.fill_between(chart["dt"], chart["equity"], initial_balance, where=chart["equity"] >= initial_balance, color="#38A169", alpha=0.18)
    axis.fill_between(chart["dt"], chart["equity"], initial_balance, where=chart["equity"] < initial_balance, color="#E53E3E", alpha=0.18)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axis.set_title("Spot Testnet Account Equity Curve (Realized Progress)", weight="bold")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Total Equity ($)")
    axis.grid(True)
    axis.legend(loc="upper left", facecolor="#1A1D29", edgecolor="#2C303E")
    figure.autofmt_xdate(rotation=20, ha="right")
    os.makedirs(os.path.dirname(REPORT_IMG_FILE) or ".", exist_ok=True)
    figure.tight_layout()
    figure.savefig(REPORT_IMG_FILE, facecolor=figure.get_facecolor())
    plt.close(figure)
    return REPORT_IMG_FILE


async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = _message(update)
    status = await message.reply_text("📈 正在生成 Equity Curve...")
    try:
        path = await asyncio.to_thread(generate_equity_chart)
        if not path or not os.path.isfile(path):
            await status.edit_text("⚠️ 沒有足夠的已實現損益資料。")
            return
        await status.delete()
        with open(path, "rb") as image:
            await message.reply_photo(image, caption="📈 *Realized Equity Curve*", parse_mode="Markdown")
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("Report generation failed")
        await status.edit_text(f"❌ 生成報表失敗: {exc}")


async def _run_script(command: str, image_path: str, update: Update, title: str) -> None:
    message = _message(update)
    status = await message.reply_text(f"⏳ 正在執行 {command}...")
    try:
        result = await asyncio.to_thread(
            subprocess.run, [sys.executable, "-m", command], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "未知錯誤").strip()[-800:]
            await status.edit_text(f"❌ 腳本執行失敗:\n`{detail}`", parse_mode="Markdown")
            return
        if not os.path.isfile(image_path):
            await status.edit_text("❌ 腳本完成，但找不到輸出圖表。")
            return
        await status.delete()
        with open(image_path, "rb") as image:
            await message.reply_photo(image, caption=title)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.exception("Script %s failed", command)
        await status.edit_text(f"❌ 執行失敗: {exc}")


async def diag_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    await _run_script("scripts.analyst_log", DIAGNOSTIC_IMG_FILE, update, "🔬 模型診斷報告")


async def sltp_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    await _run_script("scripts.sltp_backtest", SLTP_REPORT_IMG_FILE, update, "🏆 SL/TP 回測報告")


async def perf_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    message = _message(update)
    records = closed_orders()
    if not records:
        await message.reply_text("❌ 尚無 status=CLOSED 歷史訂單。")
        return
    pnls = [order_pnl(order)[0] for order in records]
    wins = sum(pnl > 0 for pnl in pnls)
    total = sum(pnls)
    await message.reply_text(
        "📈 *策略績效*\n-----------------------------------\n"
        f"• 平倉筆數: `{len(pnls)}`\n"
        f"• 勝率: `{wins / len(pnls) * 100:.1f}%`\n"
        f"• 累計實現損益: `{_money(total)}`\n"
        f"• 平均單筆損益: `{_money(total / len(pnls))}`",
        parse_mode="Markdown",
    )


async def post_init(application: Application) -> None:
    await application.bot.set_my_commands([
        BotCommand("start", "顯示指令選單"), BotCommand("help", "顯示指令選單"),
        BotCommand("status", "帳戶總覽"), BotCommand("positions", "OPEN 持倉"),
        BotCommand("history", "CLOSED 歷史"), BotCommand("report", "Equity Curve"),
        BotCommand("diag", "模型診斷"), BotCommand("perf", "策略績效"),
        BotCommand("sltp", "SL/TP 回測"),
    ])


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not configured")
        return
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    application.add_handler(CommandHandler(["start", "help"], start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("positions", positions_command))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(CommandHandler("report", report_command))
    application.add_handler(CommandHandler("diag", diag_command))
    application.add_handler(CommandHandler("perf", perf_command))
    application.add_handler(CommandHandler("sltp", sltp_command))
    application.add_handler(CallbackQueryHandler(history_page_callback, pattern=r"^hist_page_"))
    logger.info("Telegram bot listener started")
    application.run_polling()


if __name__ == "__main__":
    main()
