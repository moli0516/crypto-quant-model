import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 無 GUI 環境 (如 EC2 或背景服務) 繪圖必備
import matplotlib.pyplot as plt
import seaborn as sns

def generate_diagnostic_report(
    csv_path: str = "logs/inference_history.csv",
    trades_log_path: str = "logs/paper_trades.csv",
    output_img: str = "logs/diagnostic_report.png",
    prob_threshold: float = 0.53
):
    """
    生成高度模組化且具備實戰決策價值的 2x2 模型診斷報告。
    [升級點]: 右上角替換為各幣種「開倉觸發頻率 (Trigger Count)」與「歷史平倉勝率 (Win Rate %)」雙軸圖。
    """
    if not os.path.exists(csv_path):
        print(f"[ANALYZER] ❌ 找不到推論歷史檔: {csv_path}。請確認路徑。")
        return
    
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[ANALYZER] ❌ 推論歷史檔為空: {csv_path}。")
        return

    df['dt'] = pd.to_datetime(df['timestamp'])
    
    # 防禦性讀取歷史平倉紀錄 (若存在，用於計算各幣種實質勝率)
    df_trades = pd.DataFrame()
    if os.path.exists(trades_log_path):
        try:
            df_trades = pd.read_csv(trades_log_path)
        except Exception as e:
            print(f"[ANALYZER] ⚠️ 讀取平倉紀錄 {trades_log_path} 失敗: {e}")

    # 設定圖表風格與高解析度 Canvas
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # =========================================================================
    # 1. 總體機率分佈圖 (Overall Probability Distribution)
    # =========================================================================
    ax1 = axes[0, 0]
    sns.histplot(df['pred_proba'], bins=30, kde=True, color='skyblue', ax=ax1)
    ax1.axvline(prob_threshold, color='red', linestyle='--', linewidth=2, label=f'LONG Threshold ({prob_threshold})')
    ax1.set_title('Overall Probability Distribution (pred_proba)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Probability')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # =========================================================================
    # 2. 🆕 各幣種開倉觸發次數與歷史勝率雙軸矩陣 (Frequency vs Win Rate)
    # =========================================================================
    ax2 = axes[0, 1]
    
    # 統計各幣種突破門檻的開倉觸發次數
    triggered_signals = df[df['pred_proba'] >= prob_threshold]
    symbol_counts = triggered_signals.groupby('symbol').size()
    
    all_symbols = df['symbol'].unique()
    signal_summary = pd.DataFrame({'symbol': all_symbols}).set_index('symbol')
    signal_summary['trigger_count'] = symbol_counts
    signal_summary['trigger_count'] = signal_summary['trigger_count'].fillna(0).astype(int)
    
    # 計算各幣種在 paper_trades.csv 中的勝率
    if not df_trades.empty and 'symbol' in df_trades.columns and 'pnl_usd' in df_trades.columns:
        trade_stats = df_trades.groupby('symbol').agg(
            total_trades=('pnl_usd', 'count'),
            win_trades=('pnl_usd', lambda x: (x > 0).sum())
        )
        trade_stats['win_rate'] = (trade_stats['win_trades'] / trade_stats['total_trades']) * 100.0
        signal_summary = signal_summary.join(trade_stats['win_rate']).fillna(0.0)
    else:
        signal_summary['win_rate'] = 0.0

    signal_summary = signal_summary.sort_values(by='trigger_count', ascending=True)

    color_bar = 'steelblue'
    color_line = 'crimson'
    
    # 繪製左軸：開倉訊號觸發次數 (橫向條形圖)
    ax2.barh(signal_summary.index, signal_summary['trigger_count'], color=color_bar, alpha=0.7, label='Signal Triggers')
    ax2.set_xlabel(f'Trigger Count (pred_proba >= {prob_threshold})', color=color_bar, fontweight='bold')
    ax2.tick_params(axis='x', labelcolor=color_bar)
    
    # 繪製右軸：實質勝率 % (點線圖)
    ax2_twin = ax2.twiny()
    ax2_twin.plot(signal_summary['win_rate'], signal_summary.index, color=color_line, marker='o', linestyle='None', label='Win Rate (%)')
    ax2_twin.set_xlabel('Historical Win Rate (%)', color=color_line, fontweight='bold')
    ax2_twin.tick_params(axis='x', labelcolor=color_line)
    ax2_twin.set_xlim(-5, 105)
    ax2_twin.axvline(50.0, color='gray', linestyle=':', alpha=0.6, label='50% Benchmark')

    ax2.set_title('Symbol Signal Frequency vs Win Rate Matrix', fontsize=12, fontweight='bold')

    # =========================================================================
    # 3. 重點標的機率走勢圖 (Top Symbols Probability Trend)
    # =========================================================================
    ax3 = axes[1, 0]
    target_symbols = ['BTCUSDT', 'ETHUSDT', 'NEARUSDT']
    for sym in target_symbols:
        if sym in df['symbol'].unique():
            sub = df[df['symbol'] == sym].sort_values('dt')
            ax3.plot(sub['dt'], sub['pred_proba'], marker='o', markersize=3, label=sym)
            
    ax3.axhline(prob_threshold, color='red', linestyle='--', label=f'Threshold ({prob_threshold})')
    ax3.set_title('Top Symbols Probability Trend Over Time', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Timestamp')
    ax3.set_ylabel('Probability')
    ax3.legend()
    plt.setp(ax3.get_xticklabels(), rotation=30)
    
    # =========================================================================
    # 4. BTC 價格與模型信心雙軸關聯圖 (BTC Price vs Model Confidence)
    # =========================================================================
    ax4 = axes[1, 1]
    btc_df = df[df['symbol'] == 'BTCUSDT'].sort_values('dt')
    if not btc_df.empty:
        color_price = 'tab:blue'
        color_prob = 'tab:orange'
        
        ax4.set_xlabel('Timestamp')
        ax4.set_ylabel('BTC Close Price ($)', color=color_price)
        ax4.plot(btc_df['dt'], btc_df['close_price'], color=color_price, marker='s', markersize=3, label='BTC Price')
        ax4.tick_params(axis='y', labelcolor=color_price)
        
        ax4_twin = ax4.twinx()
        ax4_twin.set_ylabel('Model Probability', color=color_prob)
        ax4_twin.plot(btc_df['dt'], btc_df['pred_proba'], color=color_prob, marker='o', markersize=3, linestyle='--', label='Probability')
        ax4_twin.tick_params(axis='y', labelcolor=color_prob)
        ax4_twin.axhline(prob_threshold, color='red', linestyle=':', label=f'Threshold ({prob_threshold})')
        ax4.set_title('BTCUSDT: Price vs Model Confidence Correlation', fontsize=12, fontweight='bold')
        plt.setp(ax4.get_xticklabels(), rotation=30)

    # 防禦性確保輸出的父目錄存在
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_img, dpi=200)
    plt.close()
    print(f"[ANALYZER] 🔬 高階診斷報告已成功生成並輸出至: {output_img}")

if __name__ == "__main__":
    generate_diagnostic_report()