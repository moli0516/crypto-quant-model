"""
scripts/analyst_log.py
==============================================================================
Institutional-Grade 2x2 Dark Theme Model Diagnostic Report Generator
完全動態對接 src.config，支援動態 Top-K 機率追蹤與無障礙數據降級顯示。
==============================================================================
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")  # 無 GUI 環境 (如 EC2 或背景服務) 繪圖必備
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, PercentFormatter
import seaborn as sns

# 🎯 統一從集中式配置層讀取設定
from src.config import (
    INFERENCE_LOG_FILE,
    TRADES_LOG_FILE,
    DIAGNOSTIC_IMG_FILE,
    PROB_THRESHOLD
)

# =========================================================================
# 全局 Institutional Dark Theme 配置
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

logger = logging.getLogger(__name__)


def fetch_btc_minute_prices(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    """Fetch Binance 1-minute BTCUSDT closes for the diagnostic time range."""
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    rows = []
    cursor = start_ms

    try:
        while cursor <= end_ms:
            response = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                },
                timeout=10,
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + 60_000
            if next_cursor <= cursor:
                break
            cursor = next_cursor

        if not rows:
            return pd.DataFrame()

        minute_df = pd.DataFrame(rows, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_base_volume",
            "taker_quote_volume", "ignore",
        ])
        minute_df["dt"] = pd.to_datetime(minute_df["timestamp"], unit="ms", utc=True)
        minute_df["close"] = pd.to_numeric(minute_df["close"])
        return minute_df[["dt", "close"]].drop_duplicates("dt").sort_values("dt")
    except (requests.RequestException, ValueError, TypeError) as exc:
        logger.warning("Unable to fetch BTC 1-minute prices: %s", exc)
        return pd.DataFrame()


def generate_diagnostic_report(
    csv_path: str = INFERENCE_LOG_FILE,
    trades_log_path: str = TRADES_LOG_FILE,
    output_img: str = DIAGNOSTIC_IMG_FILE,
    prob_threshold: float = PROB_THRESHOLD
):
    """
    生成高度模組化、極致平滑且具備實戰決策價值的 2x2 暗黑主題模型診斷報告。
    """
    if not os.path.exists(csv_path):
        print(f"[ANALYZER] ❌ 找不到推論歷史檔: {csv_path}。請確認路徑。")
        return
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ANALYZER] ❌ 讀取推論歷史檔失敗: {e}")
        return

    if df.empty:
        print(f"[ANALYZER] ❌ 推論歷史檔為空: {csv_path}。")
        return

    # 時間戳記防禦轉型
    time_col = 'dt' if 'dt' in df.columns else 'timestamp'
    df['dt'] = pd.to_datetime(df[time_col], utc=True)
    
    # 防禦性讀取歷史平倉紀錄
    df_trades = pd.DataFrame()
    if os.path.exists(trades_log_path):
        try:
            df_trades = pd.read_csv(trades_log_path)
        except Exception as e:
            print(f"[ANALYZER] ⚠️ 讀取平倉紀錄 {trades_log_path} 失敗: {e}")

    # 建立高解析度 2x2 Canvas
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    fig.suptitle("Model Inference & Performance Diagnostic Report", fontsize=16, fontweight='bold', color='#F7FAFC', y=0.98)

    # =========================================================================
    # 1. 總體機率分佈圖 (Overall Probability Distribution)
    # =========================================================================
    ax1 = axes[0, 0]
    sns.histplot(
        df['pred_proba'], 
        bins=30, 
        kde=True, 
        color='#3182CE', 
        ax=ax1, 
        edgecolor='#1A1D29', 
        alpha=0.75,
        line_kws={'linewidth': 2, 'color': '#63B3ED'}
    )
    ax1.axvline(prob_threshold, color='#E53E3E', linestyle='--', linewidth=2, label=f'Threshold ({prob_threshold})')
    ax1.set_title('Overall Probability Distribution (pred_proba)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Probability', fontsize=10, labelpad=8)
    ax1.set_ylabel('Frequency', fontsize=10, labelpad=8)
    ax1.legend(loc='upper right', frameon=True, facecolor='#1A1D29', edgecolor='#2C303E')
    ax1.grid(True)

    # =========================================================================
    # 2. 各幣種開倉觸發次數與歷史勝率雙軸矩陣 (Frequency vs Win Rate)
    # =========================================================================
    ax2 = axes[0, 1]
    
    triggered_signals = df[df['pred_proba'] >= prob_threshold]
    symbol_counts = triggered_signals.groupby('symbol').size()
    
    all_symbols = df['symbol'].unique()
    signal_summary = pd.DataFrame({'symbol': all_symbols}).set_index('symbol')
    signal_summary['trigger_count'] = symbol_counts
    signal_summary['trigger_count'] = signal_summary['trigger_count'].fillna(0).astype(int)
    
    # 計算實質勝率 (防禦相容 pnl_usd, pnl_amount, pnl_pct)
    pnl_col = None
    for col in ['pnl_usd', 'pnl_amount', 'pnl_pct']:
        if not df_trades.empty and col in df_trades.columns:
            pnl_col = col
            break
    
    if not df_trades.empty and 'symbol' in df_trades.columns and pnl_col:
        trade_stats = df_trades.groupby('symbol').agg(
            total_trades=(pnl_col, 'count'),
            win_trades=(pnl_col, lambda x: (x > 0).sum())
        )
        trade_stats['win_rate'] = (trade_stats['win_trades'] / trade_stats['total_trades']) * 100.0
        signal_summary = signal_summary.join(trade_stats['win_rate']).fillna(0.0)
    else:
        signal_summary['win_rate'] = 0.0

    signal_summary = signal_summary.sort_values(by='trigger_count', ascending=True)

    color_bar = '#2B6CB0'   # Muted Blue
    color_line = '#E53E3E'  # Signal Red
    
    # 繪製左軸：開倉訊號觸發次數 (橫向條形圖)
    ax2.barh(signal_summary.index, signal_summary['trigger_count'], color=color_bar, alpha=0.85, height=0.6, label='Signal Triggers')
    ax2.set_xlabel(f'Trigger Count (pred_proba >= {prob_threshold})', color='#63B3ED', fontweight='bold', labelpad=8)
    ax2.tick_params(axis='x', labelcolor='#63B3ED')
    ax2.tick_params(axis='y', labelsize=9)
    
    # 繪製右軸：實質勝率 % (點線圖)
    ax2_twin = ax2.twiny()
    ax2_twin.plot(signal_summary['win_rate'], signal_summary.index, color=color_line, marker='o', markersize=6, linestyle='None', label='Win Rate (%)')
    ax2_twin.set_xlabel('Historical Win Rate (%)', color=color_line, fontweight='bold', labelpad=8)
    ax2_twin.tick_params(axis='x', labelcolor=color_line)
    ax2_twin.set_xlim(-5, 105)
    ax2_twin.axvline(50.0, color='#A0AEC0', linestyle=':', alpha=0.6, label='50% Benchmark')
    ax2_twin.xaxis.set_major_formatter(PercentFormatter())

    ax2.set_title('Symbol Signal Frequency vs Win Rate Matrix', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True)

    # =========================================================================
    # 3. 動態 Top-3 高信心標的機率走勢圖 (Dynamic Top Symbols Probability Trend)
    # =========================================================================
    ax3 = axes[1, 0]
    
    # 🎯 動態選取最新一期或最高平均機率的前 3 個熱門標的，擺脫硬編碼
    top_avg_symbols = df.groupby('symbol')['pred_proba'].mean().nlargest(3).index.tolist()
    palette = ['#4299E1', '#ED8936', '#48BB78']
    
    for idx, sym in enumerate(top_avg_symbols):
        sub = df[df['symbol'] == sym].sort_values('dt')
        ax3.plot(sub['dt'], sub['pred_proba'], marker='o', markersize=3, linewidth=1.5, color=palette[idx % len(palette)], label=sym, alpha=0.85)
            
    ax3.axhline(prob_threshold, color='#E53E3E', linestyle='--', linewidth=1.5, label=f'Threshold ({prob_threshold})')
    ax3.set_title(f'Top Avg Confidence Symbols Probability Trend', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Timestamp', fontsize=10, labelpad=8)
    ax3.set_ylabel('Probability', fontsize=10, labelpad=8)
    ax3.legend(loc='upper left', frameon=True, facecolor='#1A1D29', edgecolor='#2C303E')
    
    # 防禦 X 軸標籤重疊
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax3.get_xticklabels(), rotation=20, ha='right')
    ax3.grid(True)

    # =========================================================================
    # 4. BTC 價格與模型信心雙軸關聯圖 (BTC Price vs Model Confidence)
    # =========================================================================
    ax4 = axes[1, 1]
    btc_df = df[df['symbol'] == 'BTCUSDT'].sort_values('dt')
    
    if not btc_df.empty:
        color_price = '#3182CE'
        color_prob = '#DD6B20'
        price_col = 'close_price' if 'close_price' in btc_df.columns else ('close' if 'close' in btc_df.columns else None)
        
        if price_col:
            minute_df = fetch_btc_minute_prices(btc_df['dt'].min(), btc_df['dt'].max())
            price_df = minute_df if not minute_df.empty else btc_df[['dt', price_col]].rename(columns={price_col: 'close'})

            ax4.set_xlabel('Timestamp', fontsize=10, labelpad=8)
            ax4.set_ylabel('BTC Close Price ($)', color=color_price, fontweight='bold', labelpad=8)
            ax4.plot(price_df['dt'], price_df['close'], color=color_price, linewidth=1.0, label='BTC 1-minute Price', alpha=0.9)
            ax4.tick_params(axis='y', labelcolor=color_price)
            
            ax4_twin = ax4.twinx()
            ax4_twin.set_ylabel('Model Probability', color=color_prob, fontweight='bold', labelpad=8)
            ax4_twin.plot(btc_df['dt'], btc_df['pred_proba'], color=color_prob, linewidth=1.2, marker='o', markersize=3, linestyle='--', label='Probability', alpha=0.8)
            ax4_twin.tick_params(axis='y', labelcolor=color_prob)
            ax4_twin.axhline(prob_threshold, color='#E53E3E', linestyle=':', label=f'Threshold ({prob_threshold})')
            
            ax4.set_title('BTCUSDT: 1-minute Price vs Model Confidence', fontsize=12, fontweight='bold', pad=10)
            
            # 防禦 X 軸標籤重疊
            ax4.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax4.get_xticklabels(), rotation=20, ha='right')
            ax4.grid(True)

    # 確保輸出目錄存在並匯出高解析度圖表
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_img, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[ANALYZER] 🔬 高階診斷報告已成功生成並輸出至: {output_img}")


if __name__ == "__main__":
    generate_diagnostic_report()