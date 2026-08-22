import os
import sys
import shutil
import pandas as pd
from pathlib import Path

# 1. 自動加入專案根目錄至 sys.path，解決 ModuleNotFoundError
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.live.paper_trader import PaperTrader

def run_complete_paper_trader_test():
    test_dir = "logs_test"
    
    # 清理舊的測試日誌資料夾，確保每次測試都是乾淨的環境
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    print("🧪 開始執行 PaperTrader 完整生命週期測試...\n")

    # 2. 初始化測試用 PaperTrader Engine (設定持倉 12 小時)
    trader = PaperTrader(
        initial_balance=200.0,
        prob_threshold=0.53,
        position_pct=0.10,
        holding_hours=12,
        fee_rate=0.001,
        logs_dir=test_dir
    )

    # ----------------------------------------------------
    # 情境 A: 第 1 個小時推論 (ICP 機率 0.5372 -> 觸發開倉)
    # ----------------------------------------------------
    print("--- [Step 1] 測試開倉邏輯 ---")
    df_step1 = pd.DataFrame([
        {"symbol": "BTCUSDT", "close": 77300.0, "pred_proba": 0.4984},
        {"symbol": "ICPUSDT", "close": 2.3890, "pred_proba": 0.5372}
    ])
    state1 = trader.update_and_process(df_step1)
    
    print(f"• 可用現金: ${state1['cash']:.2f} USD (預期: $180.00)")
    print(f"• 當前持倉數: {len(state1['open_positions'])} (預期: 1)")
    print(f"• 總權益: ${state1['total_equity']:.2f} USD\n")

    # ----------------------------------------------------
    # 情境 B: 第 2 個小時推論 (ICP 機率 0.5400 -> 驗證防重複機制)
    # ----------------------------------------------------
    print("--- [Step 2] 測試單一持倉防護機制 (One Position Per Symbol) ---")
    df_step2 = pd.DataFrame([
        {"symbol": "BTCUSDT", "close": 77400.0, "pred_proba": 0.5010},
        {"symbol": "ICPUSDT", "close": 2.4100, "pred_proba": 0.5400}
    ])
    state2 = trader.update_and_process(df_step2)
    
    print(f"• 可用現金: ${state2['cash']:.2f} USD (預期維持: $180.00)")
    print(f"• 當前持倉數: {len(state2['open_positions'])} (預期維持: 1)\n")

    # ----------------------------------------------------
    # 情境 C: 模擬滿 12 小時到期平倉
    # ----------------------------------------------------
    print("--- [Step 3] 模擬滿 12 小時到期平倉機制 ---")
    # 將策略持倉限制改為 0 小時，模擬時間到達 12 小時後的下一期推論
    trader.holding_hours = 0
    
    # 假設 12 小時後 ICP 上漲至 2.5000 USD
    df_step3 = pd.DataFrame([
        {"symbol": "BTCUSDT", "close": 77500.0, "pred_proba": 0.4800},
        {"symbol": "ICPUSDT", "close": 2.5000, "pred_proba": 0.5100}
    ])
    state3 = trader.update_and_process(df_step3)
    
    print(f"• 可用現金: ${state3['cash']:.2f} USD (平倉歸還本金+獲利)")
    print(f"• 當前持倉數: {len(state3['open_positions'])} (預期: 0)")
    print(f"• 最終總權益: ${state3['total_equity']:.2f} USD\n")

    # ----------------------------------------------------
    # 驗證 CSV 紀錄檔生成情況
    # ----------------------------------------------------
    print("--- [Step 4] 檢查產出的檔案 ---")
    trades_csv = os.path.join(test_dir, "paper_trades.csv")
    equity_csv = os.path.join(test_dir, "paper_equity_daily.csv")
    
    if os.path.exists(trades_csv):
        df_trades = pd.read_csv(trades_csv)
        print(f"✅ `paper_trades.csv` 成功生成！已記錄交易筆數: {len(df_trades)}")
        print(df_trades[["symbol", "entry_price", "exit_price", "pnl_usd", "status"]].to_string(index=False))
    else:
        print("❌ `paper_trades.csv` 未生成！")

    if os.path.exists(equity_csv):
        df_equity = pd.read_csv(equity_csv)
        print(f"\n✅ `paper_equity_daily.csv` 成功生成！已記錄 Snapshot 筆數: {len(df_equity)}")

if __name__ == "__main__":
    run_complete_paper_trader_test()