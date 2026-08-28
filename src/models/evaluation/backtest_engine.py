import logging
import pandas as pd
import numpy as np
import numba as nb
from joblib import Parallel, delayed
from tqdm import tqdm  # 👈 導入 tqdm 進行視覺化進度追蹤

logger = logging.getLogger(__name__)

# =====================================================================
# ⚡ Numba C-Level 高速迴圈 Kernel
# =====================================================================
@nb.njit(fastmath=True)
def _fast_backtest_kernel(
    close_prices: np.ndarray,
    preds: np.ndarray,
    prob_threshold: float,
    max_loss_pct: float,
    base_leverage: float,
    fee_rate: float,
    horizon: int
):
    n = len(close_prices)
    strategy_returns = np.zeros(n, dtype=np.float64)
    actual_returns = np.zeros(n, dtype=np.float64)
    signals = np.zeros(n, dtype=np.float64)
    trades_count = 0
    win_count = 0

    i = 0
    while i < n - horizon:
        prob = preds[i]
        
        if prob >= prob_threshold:
            signals[i] = 1.0
            trades_count += 1
            entry_price = close_prices[i]
            exit_idx = min(i + horizon, n - 1)
            
            # 動態槓桿計算
            confidence_boost = min((prob - prob_threshold) / (1.0 - prob_threshold), 1.0)
            current_leverage = base_leverage * (1.0 + confidence_boost)
            
            # 檢查停損
            actual_exit_price = close_prices[exit_idx]
            for j in range(i + 1, exit_idx + 1):
                current_return = (close_prices[j] - entry_price) / entry_price
                if current_return <= -max_loss_pct:
                    actual_exit_price = entry_price * (1.0 - max_loss_pct)
                    exit_idx = j
                    break
            
            trade_actual_ret = (actual_exit_price - entry_price) / entry_price
            actual_returns[i] = trade_actual_ret
            
            # 策略報酬 = 實際報酬 * 槓桿 - 手續費
            trade_strat_ret = (trade_actual_ret * current_leverage) - (fee_rate * 2.0 * current_leverage)
            strategy_returns[i] = trade_strat_ret

            if trade_actual_ret > 0:
                win_count += 1

            i = exit_idx
        else:
            i += 1

    return strategy_returns, actual_returns, signals, trades_count, win_count


class SimpleStrategyBacktester:
    """
    支援單一資產或多幣種投資組合（Portfolio）的高效能動態槓桿與停損回測引擎。
    內部整合 Numba JIT 與 Joblib 多核心平行加速，並內建進度顯示。
    """

    def __init__(
        self, 
        prob_threshold: float = 0.55, 
        fee_rate: float = 0.0005, 
        max_loss_pct: float = 0.03,
        base_leverage: float = 2.0
    ):
        self.prob_threshold = prob_threshold
        self.fee_rate = fee_rate
        self.max_loss_pct = max_loss_pct
        self.base_leverage = base_leverage

    def _backtest_single_series(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """對單一幣種執行互斥波段與停損回測 (Numba 加速版)"""
        df = df.sort_index().copy()
        close_prices = df["close"].values.astype(np.float64)
        preds = df["pred_proba"].values.astype(np.float64)

        # 🚀 調用 Numba C-level 高速編譯 Kernel
        strat_ret, act_ret, sigs, t_count, w_count = _fast_backtest_kernel(
            close_prices=close_prices,
            preds=preds,
            prob_threshold=float(self.prob_threshold),
            max_loss_pct=float(self.max_loss_pct),
            base_leverage=float(self.base_leverage),
            fee_rate=float(self.fee_rate),
            horizon=int(horizon)
        )

        df["signal"] = sigs
        df["strategy_return"] = strat_ret
        df["trades_count"] = t_count
        df["wins"] = w_count
        return df

    def run_backtest(self, wf_predictions: pd.DataFrame, horizon: int = 12, show_progress: bool = True) -> dict:
        """
        接收預測 DataFrame 進行回測 (支援 CPU 多核心平行加速與進度條顯示)。
        
        :param wf_predictions: 包含預測與標價格的 DataFrame
        :param horizon: 預測持有區間 (預設 12H)
        :param show_progress: 是否開啟視覺化進度條 (預設 True)
        """
        if "pred_proba" not in wf_predictions.columns or "close" not in wf_predictions.columns:
            raise KeyError("❌ 預測 DataFrame 缺少 'pred_proba' 或 'close' 欄位！")

        # 判斷是否為多幣種資料
        if "symbol" in wf_predictions.columns:
            logger.info("🌍 偵測到多幣種資料，啟動 CPU 多核心平行加速回測...")
            
            groups = [group for _, group in wf_predictions.groupby("symbol")]
            total_symbols = len(groups)

            # 📊 建立平行的 tqdm 進度監控
            if show_progress:
                processed_dfs = Parallel(n_jobs=-1)(
                    delayed(self._backtest_single_series)(group, horizon) 
                    for group in tqdm(groups, desc="⚡ 回測運算進度", unit="symbol", ascii=True)
                )
            else:
                processed_dfs = Parallel(n_jobs=-1)(
                    delayed(self._backtest_single_series)(group, horizon) for group in groups
                )
            
            total_trades = 0
            total_wins = 0
            symbol_metrics = {}

            for res_df in processed_dfs:
                symbol = res_df["symbol"].iloc[0] if "symbol" in res_df.columns else "UNKNOWN"
                t_count = res_df["trades_count"].iloc[0]
                w_count = res_df["wins"].iloc[0]
                w_rate = float(w_count / t_count) if t_count > 0 else 0.0
                strat_ret = float(res_df["strategy_return"].sum())
                
                eq_curve = 1.0 + res_df["strategy_return"].cumsum()
                roll_max = eq_curve.cummax()
                dd = (eq_curve - roll_max) / (roll_max + 1e-10)
                max_dd = float(dd.min())
                
                ret = res_df["strategy_return"]
                sr = float((ret.mean() / (ret.std() + 1e-10)) * np.sqrt(8760)) if ret.std() > 0 else 0.0
                bnh = float((res_df["close"].iloc[-1] - res_df["close"].iloc[0]) / res_df["close"].iloc[0]) if len(res_df) > 0 else 0.0
                
                symbol_metrics[symbol] = {
                    "Trades": t_count,
                    "WinRate": w_rate,
                    "StratRet": strat_ret,
                    "B&H": bnh,
                    "Sharpe": sr,
                    "MaxDD": max_dd
                }

                total_trades += t_count
                total_wins += w_count

            # 終端機列印各幣種明細表 (依策略總報酬排序)
            print("\n" + "="*85)
            print("✨ 各幣種獨立回測績效明細:")
            print(f"{'Symbol':<10} | {'Trades':<8} | {'Win%':<8} | {'Strat Ret':<12} | {'B&H Ret':<10} | {'Sharpe':<8} | {'MaxDD':<10}")
            print("-" * 85)
            for sym, m in sorted(symbol_metrics.items(), key=lambda x: x[1]['StratRet'], reverse=True):
                print(f"{sym:<10} | {m['Trades']:<8} | {m['WinRate']*100:>6.2f}% | {m['StratRet']*100:>9.2f}%   | {m['B&H']*100:>7.2f}%   | {m['Sharpe']:>6.2f}   | {m['MaxDD']*100:>7.2f}%")
            print("="*85 + "\n")

            df = pd.concat(processed_dfs).sort_index()
            portfolio_returns = df.groupby(df.index)["strategy_return"].mean().fillna(0)
            
            df_port = pd.DataFrame({"strategy_return": portfolio_returns})
            df_port["close"] = df.groupby(df.index)["close"].mean()
            
            win_rate = float(total_wins / total_trades) if total_trades > 0 else 0.0
            total_return_pct = float(df_port["strategy_return"].sum())
            
            equity_curve = 1.0 + df_port["strategy_return"].cumsum()
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve - rolling_max) / (rolling_max + 1e-10)
            max_drawdown = float(drawdown.min())
            
            returns = df_port["strategy_return"]
            sharpe_ratio = float((returns.mean() / (returns.std() + 1e-10)) * np.sqrt(8760)) if returns.std() > 0 else 0.0
            
            if len(df_port) > 0:
                market_return_pct = float((df_port["close"].iloc[-1] - df_port["close"].iloc[0]) / df_port["close"].iloc[0])
            else:
                market_return_pct = 0.0
                
            trades_count = total_trades

        else:
            res_df = self._backtest_single_series(wf_predictions, horizon)
            total_trades = res_df["trades_count"].iloc[0]
            win_rate = float(res_df["wins"].iloc[0] / total_trades) if total_trades > 0 else 0.0
            total_return_pct = float(res_df["strategy_return"].sum())
            
            equity_curve = 1.0 + res_df["strategy_return"].cumsum()
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve - rolling_max) / (rolling_max + 1e-10)
            max_drawdown = float(drawdown.min())
            
            returns = res_df["strategy_return"]
            sharpe_ratio = float((returns.mean() / (returns.std() + 1e-10)) * np.sqrt(8760)) if returns.std() > 0 else 0.0
            
            if len(res_df) > 0:
                market_return_pct = float((res_df["close"].iloc[-1] - res_df["close"].iloc[0]) / res_df["close"].iloc[0])
            else:
                market_return_pct = 0.0
            
            trades_count = total_trades
            df = res_df

        metrics = {
            "total_trades": trades_count,
            "win_rate": win_rate,
            "strategy_total_return": total_return_pct,
            "buy_and_hold_return": market_return_pct,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }

        logger.info(f"💰 策略回測完畢！總報酬: {total_return_pct*100:.2f}% | 夏普比率: {sharpe_ratio:.2f}")
        return {
            "metrics": metrics,
            "dataframe": df
        }