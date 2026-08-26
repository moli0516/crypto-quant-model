import os
import sys
import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# 繪圖套件 (無 GUI 環境預設 Agg)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# 確保可載入 src 模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.collectors.binance.rest_client import BinanceAsyncRESTClient
from src.cleaners.binance_cleaner import BinanceOHLCVCleaner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class HighFreqSLTPBacktester:
    """
    高頻 1 分鐘 K 線 SL/TP 碰撞回測、網格搜尋與診斷圖表生成引擎
    """
    def __init__(
        self, 
        initial_balance: float = 200.0, 
        position_pct: float = 0.10, 
        fee_rate: float = 0.0008, 
        holding_hours: int = 12,
        logs_dir: str = "local-logs"
    ):
        """
        :param initial_balance: 初始本金 (預設 $200 USD)
        :param position_pct: 下注比例 (預設 10% 總資產)
        :param fee_rate: 來回交易手續費 + 滑點預估 (預設 0.08% = 0.0008)
        :param holding_hours: 最長持倉時間，超過此時間以第 H 小時 Close 價平倉
        :param logs_dir: 日誌與報告輸出目錄
        """
        self.initial_balance = initial_balance
        self.position_pct = position_pct
        self.fee_rate = fee_rate
        self.holding_hours = holding_hours
        self.logs_dir = logs_dir
        self.cleaner = BinanceOHLCVCleaner(fill_method="ffill")
        os.makedirs(self.logs_dir, exist_ok=True)

    async def fetch_1m_klines_for_signal(
        self, 
        client: BinanceAsyncRESTClient, 
        symbol: str, 
        start_dt: pd.Timestamp
    ) -> pd.DataFrame:
        """
        根據進場時間，抓取未來 holding_hours 小時內的 1 分鐘 K 線
        """
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = start_ms + int((self.holding_hours * 3600 + 600) * 1000)
        limit = self.holding_hours * 60 + 10

        try:
            raw_df = await client.fetch_historical_ohlcv(
                symbol=symbol,
                timeframe="1m",
                start_time=start_ms,
                end_time=end_ms,
                limit=limit
            )
            if raw_df.empty:
                return pd.DataFrame()
            
            cleaned_df = self.cleaner(raw_df)
            return cleaned_df
        except Exception as e:
            logger.error(f"❌ 抓取 {symbol} ({start_dt}) 1m K 線失敗: {e}")
            return pd.DataFrame()

    def simulate_single_trade(
        self, 
        entry_price: float, 
        klines_1m: pd.DataFrame, 
        tp_pct: float, 
        sl_pct: float
    ) -> dict:
        """
        對單筆持倉進行 1m 級別的逐分碰撞模擬
        """
        if klines_1m.empty:
            return {"exit_type": "NO_DATA", "pnl_pct": 0.0, "duration_mins": 0}

        tp_price = entry_price * (1.0 + tp_pct)
        sl_price = entry_price * (1.0 - sl_pct)

        max_mins = self.holding_hours * 60
        klines_sub = klines_1m.iloc[:max_mins]

        for idx, (_, row) in enumerate(klines_sub.iterrows()):
            high_p = row["high"]
            low_p = row["low"]

            # 碰撞檢查：同一根 K 線同時碰到 SL/TP 時，採保守原則視為觸發 SL
            if low_p <= sl_price:
                net_pnl = -sl_pct - self.fee_rate
                return {"exit_type": "SL", "pnl_pct": net_pnl, "duration_mins": idx + 1}
            elif high_p >= tp_price:
                net_pnl = tp_pct - self.fee_rate
                return {"exit_type": "TP", "pnl_pct": net_pnl, "duration_mins": idx + 1}

        # 超過最長持倉時間未觸發 SL/TP，以第 12 小時收盤價平倉
        last_close = klines_sub["close"].iloc[-1]
        raw_pnl = (last_close - entry_price) / entry_price
        net_pnl = raw_pnl - self.fee_rate
        return {"exit_type": "TIMEOUT", "pnl_pct": net_pnl, "duration_mins": len(klines_sub)}

    def plot_diagnostics(
        self, 
        report_df: pd.DataFrame, 
        best_equity_history: list, 
        best_tp: float, 
        best_sl: float
    ):
        """
        生成 SL/TP 報酬率熱力圖與最佳參數資產曲線圖
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 1. 熱力圖 (Heatmap) - 總報酬率分布
        heatmap_data = report_df.pivot(index="SL (%)", columns="TP (%)", values="Total Return (%)")
        sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt=".2f", 
            cmap="YlGnBu", 
            ax=axes[0], 
            cbar_kws={'label': 'Total Return (%)'}
        )
        axes[0].set_title("SL/TP Return Heatmap (%)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Take Profit (%)")
        axes[0].set_ylabel("Stop Loss (%)")

        # 2. 資金資產曲線圖 (Equity Curve)
        axes[1].plot(
            best_equity_history, 
            marker='o', 
            color='#1f77b4', 
            linewidth=2, 
            label=f"Best Strategy (TP {best_tp}% / SL {best_sl}%)"
        )
        axes[1].axhline(
            y=self.initial_balance, 
            color='r', 
            linestyle='--', 
            alpha=0.6, 
            label=f"Initial Balance (${self.initial_balance})"
        )
        axes[1].set_title(
            f"Equity Curve - Best Params (TP {best_tp}% / SL {best_sl}%)", 
            fontsize=12, 
            fontweight="bold"
        )
        axes[1].set_xlabel("Executed Trade Index")
        axes[1].set_ylabel("Total Equity ($)")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend()

        plt.tight_layout()
        save_path = os.path.join(self.logs_dir, "sltp_diagnostic_report.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"🎨 診斷圖表已成功輸出至: {save_path}")

    async def run_grid_backtest(
        self, 
        signals_df: pd.DataFrame, 
        tp_grid: list[float], 
        sl_grid: list[float]
    ) -> pd.DataFrame:
        """
        執行多筆持倉與 SL/TP 組合的網格回測 (含本金、10%倉位與同幣種排他機制，並繪製診斷圖)
        """
        # 1. 篩選出需要 LONG 的持倉紀錄，並確保按時間排序
        if "status" in signals_df.columns:
            long_signals = signals_df[signals_df["status"] == "LONG"].copy()
        else:
            long_signals = signals_df.copy()

        if long_signals.empty:
            logger.warning("⚠️ 沒有可回測的 LONG 持倉紀錄。")
            return pd.DataFrame()

        long_signals["timestamp"] = pd.to_datetime(long_signals["timestamp"], utc=True)
        long_signals = long_signals.sort_values(by="timestamp").reset_index(drop=True)

        # 2. 預載所有訊號觸發後的 1m 高頻 K 線
        logger.info(f"🚀 開始為 {len(long_signals)} 筆持倉紀錄批次下載 1 分鐘 K 線...")
        klines_cache = {}

        async with BinanceAsyncRESTClient() as client:
            for idx, (_, row) in enumerate(long_signals.iterrows()):
                symbol = str(row["symbol"]).strip()
                timestamp = row["timestamp"]

                cache_key = f"{symbol}_{timestamp.strftime('%Y%m%d%H%M')}"
                if cache_key not in klines_cache:
                    logger.info(f"[{idx+1}/{len(long_signals)}] 下載 {symbol} @ {timestamp}")
                    klines_cache[cache_key] = await self.fetch_1m_klines_for_signal(
                        client=client, symbol=symbol, start_dt=timestamp
                    )
                    await asyncio.sleep(0.05) # Rate limit 緩衝

        # 3. 網格搜尋計算
        logger.info("\n⚙️ 開始執行 SL/TP 參數網格碰撞模擬...")
        grid_results = []
        equity_histories = {} # 記錄各組合的資金變化曲線

        for tp in tp_grid:
            for sl in sl_grid:
                current_equity = self.initial_balance
                equity_curve = [self.initial_balance]
                win_count = 0
                sl_count = 0
                tp_count = 0
                timeout_count = 0
                executed_trades = 0
                
                # 🔒 紀錄各幣種目前持倉鎖定解除的時間點 (同幣種排他機制)
                active_until = {}

                for _, row in long_signals.iterrows():
                    symbol = str(row["symbol"]).strip()
                    price = float(row["close_price"]) if "close_price" in row else float(row["price"])
                    timestamp = row["timestamp"]

                    # ✋ 檢查條件 3: 同一幣種是否還在持倉中
                    if symbol in active_until and timestamp < active_until[symbol]:
                        continue # 當前已有該幣種持倉，跳過此訊號

                    cache_key = f"{symbol}_{timestamp.strftime('%Y%m%d%H%M')}"
                    k_df = klines_cache.get(cache_key, pd.DataFrame())

                    # 模擬單筆碰撞
                    res = self.simulate_single_trade(
                        entry_price=price, klines_1m=k_df, tp_pct=tp, sl_pct=sl
                    )

                    if res["exit_type"] == "NO_DATA":
                        continue

                    # 💰 條件 1 & 2: 計算 10% 資產下注與資金盈虧
                    position_size = current_equity * self.position_pct
                    pnl_pct = res["pnl_pct"]
                    pnl_usd = position_size * pnl_pct
                    current_equity += pnl_usd  # 動態更新總資產
                    equity_curve.append(current_equity)

                    # 更新該幣種持倉解鎖時間
                    exit_mins = res["duration_mins"]
                    active_until[symbol] = timestamp + pd.Timedelta(minutes=exit_mins)

                    # 統計指標
                    executed_trades += 1
                    if pnl_pct > 0:
                        win_count += 1
                    if res["exit_type"] == "SL":
                        sl_count += 1
                    elif res["exit_type"] == "TP":
                        tp_count += 1
                    elif res["exit_type"] == "TIMEOUT":
                        timeout_count += 1

                win_rate = (win_count / executed_trades * 100.0) if executed_trades > 0 else 0.0
                total_return_pct = ((current_equity - self.initial_balance) / self.initial_balance) * 100.0

                param_key = f"TP_{round(tp*100, 2)}_SL_{round(sl*100, 2)}"
                equity_histories[param_key] = equity_curve

                grid_results.append({
                    "TP (%)": round(tp * 100, 2),
                    "SL (%)": round(sl * 100, 2),
                    "Win Rate (%)": round(win_rate, 2),
                    "Final Equity ($)": round(current_equity, 2),
                    "Total Return (%)": round(total_return_pct, 2),
                    "TP Count": tp_count,
                    "SL Count": sl_count,
                    "Timeout Count": timeout_count,
                    "Executed Trades": executed_trades
                })

        report_df = pd.DataFrame(grid_results).sort_values(by="Final Equity ($)", ascending=False)

        # 🎨 繪製第一名冠軍組合的診斷圖表
        best_row = report_df.iloc[0]
        best_tp, best_sl = best_row["TP (%)"], best_row["SL (%)"]
        best_key = f"TP_{best_tp}_SL_{best_sl}"
        self.plot_diagnostics(report_df, equity_histories[best_key], best_tp, best_sl)

        return report_df


async def main():
    csv_file = "local-logs/inference_history.csv" 
    
    if not os.path.exists(csv_file):
        logger.error(f"❌ 找不到訊號檔案: {csv_file}")
        return

    signals_df = pd.read_csv(csv_file)

    # 定義網格搜尋範圍
    tp_grid = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04]  # 1.0% ~ 4.0%
    sl_grid = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04]  # 1.0% ~ 4.0%

    # 傳入 initial_balance=200.0, position_pct=0.10, logs_dir="local-logs"
    backtester = HighFreqSLTPBacktester(
        initial_balance=200.0,
        position_pct=0.10,
        fee_rate=0.0008,
        holding_hours=12,
        logs_dir="local-logs"
    )
    report_df = await backtester.run_grid_backtest(signals_df, tp_grid, sl_grid)

    print("\n" + "=" * 85)
    print("🏆 SL/TP 網格回測排行榜 (已生成診斷圖表 local-logs/sltp_diagnostic_report.png):")
    print("=" * 85)
    print(report_df.to_string(index=False))
    print("=" * 85)

    output_path = "local-logs/sltp_grid_report.csv"
    report_df.to_csv(output_path, index=False)
    logger.info(f"📊 完整回測報告已儲存至: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())