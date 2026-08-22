import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_diagnostic_report(csv_path: str = "logs/inference_history.csv", output_img: str = "logs/diagnostic_report.png"):
    if not os.path.exists(csv_path):
        print(f"[ANALYZER] 找不到檔案 {csv_path}。請確認路徑正確。")
        return
    
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[ANALYZER] CSV 檔案為空。")
        return

    df['dt'] = pd.to_datetime(df['timestamp'])
    
    # 設定圖表風格
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 總體機率分佈圖
    ax1 = axes[0, 0]
    sns.histplot(df['pred_proba'], bins=30, kde=True, color='skyblue', ax=ax1)
    ax1.axvline(0.53, color='red', linestyle='--', linewidth=2, label='LONG Threshold (0.53)')
    ax1.set_title('Overall Probability Distribution (pred_proba)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Probability')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # 2. 各幣種歷史最高機率排名
    ax2 = axes[0, 1]
    symbol_max = df.groupby('symbol')['pred_proba'].max().sort_values(ascending=True)
    colors = ['crimson' if val >= 0.53 else 'navy' for val in symbol_max.values]
    symbol_max.plot(kind='barh', color=colors, ax=ax2)
    ax2.axvline(0.53, color='red', linestyle='--', linewidth=1.5)
    ax2.set_title('Max Probability Reached per Symbol', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Max Probability')
    
    # 3. 重點標的機率走勢圖
    ax3 = axes[1, 0]
    for sym in ['BTCUSDT', 'ETHUSDT', 'NEARUSDT']:
        if sym in df['symbol'].unique():
            sub = df[df['symbol'] == sym].sort_values('dt')
            ax3.plot(sub['dt'], sub['pred_proba'], marker='o', label=sym)
    ax3.axhline(0.53, color='red', linestyle='--', label='Threshold (0.53)')
    ax3.set_title('Top Symbols Probability Trend Over Time', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Timestamp')
    ax3.set_ylabel('Probability')
    ax3.legend()
    plt.setp(ax3.get_xticklabels(), rotation=30)
    
    # 4. BTC 價格與模型信心雙軸圖
    ax4 = axes[1, 1]
    btc_df = df[df['symbol'] == 'BTCUSDT'].sort_values('dt')
    if not btc_df.empty:
        color_price = 'tab:blue'
        color_prob = 'tab:orange'
        
        ax4.set_xlabel('Timestamp')
        ax4.set_ylabel('BTC Close Price ($)', color=color_price)
        ax4.plot(btc_df['dt'], btc_df['close_price'], color=color_price, marker='s', label='BTC Price')
        ax4.tick_params(axis='y', labelcolor=color_price)
        
        ax4_twin = ax4.twinx()
        ax4_twin.set_ylabel('Model Probability', color=color_prob)
        ax4_twin.plot(btc_df['dt'], btc_df['pred_proba'], color=color_prob, marker='o', linestyle='--', label='Probability')
        ax4_twin.tick_params(axis='y', labelcolor=color_prob)
        ax4_twin.axhline(0.53, color='red', linestyle=':', label='Threshold (0.53)')
        ax4.set_title('BTCUSDT: Price vs Model Confidence Correlation', fontsize=12, fontweight='bold')
        plt.setp(ax4.get_xticklabels(), rotation=30)

    # 確保輸出資料夾存在
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"[ANALYZER] 診斷圖表已成功輸出至 {output_img}")

if __name__ == "__main__":
    generate_diagnostic_report()