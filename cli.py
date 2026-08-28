#!/usr/bin/env python3
"""
crypto-quant-model 統一 CLI 入口
"""

import argparse
import asyncio
import os
import sys
import pandas as pd
from pathlib import Path


async def collect_ohlcv(args: argparse.Namespace) -> None:
    """Fetch Binance OHLCV data and write it to stdout or a CSV file."""
    from src.collectors.binance.rest_client import BinanceAsyncRESTClient

    async with BinanceAsyncRESTClient(api_key=args.api_key) as client:
        data = await client.fetch_historical_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_time=args.start_time,
            end_time=args.end_time,
            limit=args.limit,
        )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        data.to_csv(args.output)
        print(f"已收集 {len(data)} 筆資料至 {args.output}")
    else:
        data.to_csv(sys.stdout)


async def batch_collect_ohlcv(args: argparse.Namespace) -> None:
    """非同步批量下載並清洗前 20 大熱門幣種歷史 K 線"""
    from src.collectors.batch_collector import run_batch_collection

    print(f"🌍 啟動 CLI: 批量收集前 20 大熱門幣種數據 (Timeframe: {args.timeframe}, Limit: {args.limit})...")
    await run_batch_collection(timeframe=args.timeframe, limit=args.limit)
    print("✨ 前 20 大幣種批量收集與清洗完畢！")


def clean_ohlcv(args: argparse.Namespace) -> None:
    """執行資料清洗命令"""
    from src.cleaners.binance_cleaner import BinanceOHLCVCleaner

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 錯誤: 找不到輸入檔案 {input_path}")
        sys.exit(1)

    print(f"📂 正在讀取原始資料: {input_path} ...")
    df = pd.read_csv(input_path, index_col="timestamp", parse_dates=True)

    cleaner = BinanceOHLCVCleaner(fill_method=args.fill_method)
    cleaned_df = cleaner(df)

    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)
    cleaned_df.to_csv(output_path)
    print(f"✨ 清洗完畢！已將 {len(cleaned_df)} 筆乾淨資料儲存至 {output_path}")


def generate_features(args: argparse.Namespace) -> None:
    """執行特徵工程命令"""
    from src.features.feature_pipeline import FeaturePipeline

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 錯誤: 找不到輸入檔案 {input_path}")
        sys.exit(1)

    print(f"📂 正在讀取清洗後的資料: {input_path} ...")
    df = pd.read_csv(input_path, index_col="timestamp", parse_dates=True)

    pipeline = FeaturePipeline()
    feature_df = pipeline.fit_transform(df)

    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)
    feature_df.to_csv(output_path)
    print(f"✨ 特徵工程完畢！已將維度為 {feature_df.shape} 的特徵矩陣儲存至 {output_path}")


def train_model(args: argparse.Namespace) -> None:
    """執行模型訓練命令"""
    from src.models.model_pipeline import ModelPipeline

    print("🚀 啟動模型訓練工作流...")
    pipeline = ModelPipeline()

    model, metrics = pipeline.run_train_pipeline(
        model_name=args.model_name,
        val_days=args.val_days,
        horizon=args.horizon,
        threshold=args.threshold
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    
    print(f"✨ 模型訓練完畢！")
    print(f"📊 驗證指標: {metrics}")
    print(f"💾 模型已儲存至: {output_path}")


def run_walk_forward(args: argparse.Namespace) -> None:
    """執行 Walk-Forward 滾動交叉驗證"""
    from src.models.data_loader import CryptoDataLoader
    from src.models.evaluation.walk_forward import WalkForwardEvaluator

    print("📈 啟動 Walk-Forward 滾動交叉驗證工作流...")
    
    loader = CryptoDataLoader()
    dataset, feature_cols = loader.load_dataset(horizon=args.horizon, threshold=args.threshold)

    evaluator = WalkForwardEvaluator(
        feature_cols=feature_cols,
        model_name=args.model_name,
        min_train_days=args.min_train_days,
        step_days=args.step_days,
        horizon=args.horizon,
        threshold=args.threshold
    )

    result = evaluator.evaluate(dataset)
    metrics = result["metrics"]

    print(f"\n✨ Walk-Forward 交叉驗證完成！")
    print(f"📊 外樣本外 (Out-of-Sample) 綜合績效:")
    for k, v in metrics.items():
        print(f"   • {k}: {v}")


def run_backtest_cli(args: argparse.Namespace) -> None:
    """執行策略回測與資金曲線模擬 (支援高速快取)"""
    from src.models.data_loader import CryptoDataLoader
    from src.models.evaluation.walk_forward import WalkForwardEvaluator
    from src.models.evaluation.backtest_engine import SimpleStrategyBacktester

    print("📈 正在執行 Walk-Forward 預測並進行策略回測...")
    
    # 💾 檢查有無先前的 Walk-Forward 預測快取檔
    cache_dir = Path("local-logs")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"wf_preds_cache_{args.model_name}_h{args.horizon}.parquet"

    if cache_file.exists():
        print(f"⚡ 偵測到快取預測檔 ({cache_file.name})，秒級載入預測結果...")
        preds = pd.read_parquet(cache_file)
    else:
        loader = CryptoDataLoader()
        dataset, feature_cols = loader.load_dataset(horizon=args.horizon, threshold=args.threshold)

        evaluator = WalkForwardEvaluator(
            feature_cols=feature_cols,
            model_name=args.model_name,
            min_train_days=args.min_train_days,
            step_days=args.step_days,
            horizon=args.horizon,
            threshold=args.threshold
        )

        # 1. 取得 WF 預測結果並寫入快取
        wf_result = evaluator.evaluate(dataset)
        preds = wf_result["predictions"]
        
        if not preds.empty:
            preds.to_parquet(cache_file)
            print(f"💾 預測結果已快取至: {cache_file}")

    if preds.empty:
        print("❌ 錯誤: WF 預測結果為空，無法回測。")
        return

    # 2. 執行超高速 Numba + Joblib 回測
    backtester = SimpleStrategyBacktester(
        prob_threshold=args.prob_threshold,
        fee_rate=args.fee_rate,
        base_leverage=args.base_leverage
    )
    bt_result = backtester.run_backtest(preds, horizon=args.horizon)
    metrics = bt_result["metrics"]

    print(f"\n✨ 策略回測報告 (扣除 {args.fee_rate*100}% 手續費):")
    print(f"   • 總進場交易次數: {metrics['total_trades']}")
    print(f"   • 進場勝率 (Win Rate): {metrics['win_rate']*100:.2f}%")
    print(f"   • 策略總報酬率 (Strategy Return): {metrics['strategy_total_return']*100:.2f}%")
    print(f"   • 標的買入持有報酬 (Buy & Hold): {metrics['buy_and_hold_return']*100:.2f}%")
    print(f"   • 年化夏普比率 (Sharpe Ratio): {metrics['sharpe_ratio']:.2f}")
    print(f"   • 最大回撤 (Max Drawdown): {metrics['max_drawdown']*100:.2f}%")

def run_batch_feature_cli(args: argparse.Namespace) -> None:
    """執行多幣種批次特徵工程與 Parquet 輸出"""
    from src.features.batch_feature_pipeline import run_batch_feature_engineering
    run_batch_feature_engineering(input_dir=args.input_dir, output_dir=args.output_dir)
    
# 在 cli.py 中新增此處理函式
def run_ensemble_backtest_cli(args: argparse.Namespace) -> None:
    """執行雙模型 (XGBoost + LightGBM) 集成策略回測"""
    from src.models.data_loader import CryptoDataLoader
    from src.models.evaluation.ensemble_evaluator import EnsembleWalkForwardEvaluator
    from src.models.evaluation.backtest_engine import SimpleStrategyBacktester

    print("📈 啟動雙模型集成 (Ensemble: XGB + LGB) Walk-Forward 與策略回測...")
    
    loader = CryptoDataLoader()
    dataset, feature_cols = loader.load_dataset(horizon=args.horizon, threshold=args.threshold)

    # 1. 執行雙模型評估與融合
    ensemble_eval = EnsembleWalkForwardEvaluator(
        feature_cols=feature_cols,
        model_a="xgb_classifier",
        model_b="lgb_classifier",
        weights=[0.5, 0.5],
        min_train_days=args.min_train_days,
        step_days=args.step_days,
        horizon=args.horizon,
        threshold=args.threshold
    )
    blended_preds = ensemble_eval.evaluate_and_blend(dataset)

    # 2. 傳入原有的 SimpleStrategyBacktester 進行高吞吐 Numba 回測
    backtester = SimpleStrategyBacktester(
        prob_threshold=args.prob_threshold,
        fee_rate=args.fee_rate,
        base_leverage=args.base_leverage
    )
    bt_result = backtester.run_backtest(blended_preds, horizon=args.horizon)
    metrics = bt_result["metrics"]

    print(f"\n✨ 雙模型集成 (XGB+LGB) 回測報告 (扣除 {args.fee_rate*100}% 手續費):")
    print(f"   • 總進場交易次數: {metrics['total_trades']}")
    print(f"   • 進場勝率 (Win Rate): {metrics['win_rate']*100:.2f}%")
    print(f"   • 策略總報酬率 (Strategy Return): {metrics['strategy_total_return']*100:.2f}%")
    print(f"   • 標的買入持有報酬 (Buy & Hold): {metrics['buy_and_hold_return']*100:.2f}%")
    print(f"   • 年化夏普比率 (Sharpe Ratio): {metrics['sharpe_ratio']:.2f}")
    print(f"   • 最大回撤 (Max Drawdown): {metrics['max_drawdown']*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(
        description="crypto-quant-model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 1. Collect 命令 (單一資產)
    collect_parser = subparsers.add_parser("collect", help="執行單一資產資料收集")
    collect_parser.add_argument("--symbol", type=str, default="BTCUSDT")
    collect_parser.add_argument("--timeframe", type=str, default="1h")
    collect_parser.add_argument("--limit", type=int, default=1000)
    collect_parser.add_argument("--start-time", type=int)
    collect_parser.add_argument("--end-time", type=int)
    collect_parser.add_argument("--output", type=str, default="data/raw/raw_record.csv", help="輸出的 CSV 檔案路徑")
    collect_parser.add_argument("--api-key", default=os.getenv("BINANCE_API_KEY"))

    # 1-1. 🆕 Batch-Collect 命令 (前 20 大熱門幣種批次收集)
    batch_parser = subparsers.add_parser("batch-collect", help="非同步批量下載並清洗前 20 大熱門幣種歷史 K 線")
    batch_parser.add_argument("--timeframe", type=str, default="1h", help="K 線時間週期 (預設 1h)")
    batch_parser.add_argument("--limit", type=int, default=15000, help="每個幣種最大抓取筆數 (預設 15000)")

    # 2. Clean 命令
    clean_parser = subparsers.add_parser("clean", help="執行資料清洗")
    clean_parser.add_argument("--input", type=str, default="data/raw/raw_record.csv", help="輸入的原始 CSV 檔案路徑")
    clean_parser.add_argument("--output", type=str, default="data/interim/clean_record.csv", help="清洗後的輸出 CSV 檔案路徑")
    clean_parser.add_argument("--fill-method", type=str, default="ffill", help="缺失值填補方法 (ffill/bfill/zero)")

    # 3. Feature 命令
    feature_parser = subparsers.add_parser("feature", help="執行特徵工程")
    feature_parser.add_argument("--input", type=str, default="data/interim/clean_record.csv", help="輸入的中繼清洗 CSV 檔案路徑")
    feature_parser.add_argument("--output", type=str, default="data/processed/feature_matrix.csv", help="特徵矩陣的輸出 CSV 檔案路徑")

    # 4. Train 命令
    train_parser = subparsers.add_parser("train", help="執行模型訓練")
    train_parser.add_argument("--model-name", type=str, default="xgb_classifier", help="模型註冊名稱")
    train_parser.add_argument("--val-days", type=int, default=30, help="驗證集天數")
    train_parser.add_argument("--horizon", type=int, default=1, help="預測未來幾小時的標籤")
    train_parser.add_argument("--threshold", type=float, default=0.0, help="正報酬判定門檻")
    train_parser.add_argument("--output", type=str, default="data/models/xgb_model.pkl", help="模型輸出的序列化檔案路徑")

    # 5. Walk-Forward Evaluation 命令
    wf_parser = subparsers.add_parser("wf-eval", help="執行 Walk-Forward 滾動交叉驗證")
    wf_parser.add_argument("--model-name", type=str, default="xgb_classifier", help="模型註冊名稱")
    wf_parser.add_argument("--min-train-days", type=int, default=365, help="最小訓練天數")
    wf_parser.add_argument("--step-days", type=int, default=30, help="每折測試步長天數")
    wf_parser.add_argument("--horizon", type=int, default=1, help="預測未來幾小時的標籤")
    wf_parser.add_argument("--threshold", type=float, default=0.0, help="正報酬判定門檻")

    # 6. Backtest 命令
    bt_parser = subparsers.add_parser("backtest", help="執行策略回測與資金曲線模擬")
    bt_parser.add_argument("--model-name", type=str, default="xgb_classifier", help="模型註冊名稱")
    bt_parser.add_argument("--min-train-days", type=int, default=365, help="最小訓練天數")
    bt_parser.add_argument("--step-days", type=int, default=30, help="每折測試步長天數")
    bt_parser.add_argument("--horizon", type=int, default=1, help="預測未來幾小時的標籤")
    bt_parser.add_argument("--threshold", type=float, default=0.0, help="正報酬判定門檻")
    bt_parser.add_argument("--prob-threshold", type=float, default=0.52, help="進場做多機率門檻")
    bt_parser.add_argument("--fee-rate", type=float, default=0.0005, help="每筆交易手續費率")
    bt_parser.add_argument("--base-leverage", type=float, default=2.0, help="基礎合約槓桿倍數")
    
    batch_feat_parser = subparsers.add_parser("batch-feature", help="批量對所有清洗後的幣種執行特徵工程並輸出為 Parquet")
    batch_feat_parser.add_argument("--input-dir", type=str, default="data/interim", help="輸入的中繼資料夾")
    batch_feat_parser.add_argument("--output-dir", type=str, default="data/processed", help="特徵矩陣輸出的 Parquet 資料夾")
    
    # 6-1. Ensemble Backtest 命令
    ens_parser = subparsers.add_parser("ensemble-backtest", help="執行雙模型 (XGB+LGB) 融合策略回測")
    ens_parser.add_argument("--min-train-days", type=int, default=180)
    ens_parser.add_argument("--step-days", type=int, default=30)
    ens_parser.add_argument("--horizon", type=int, default=12)
    ens_parser.add_argument("--threshold", type=float, default=0.0)
    ens_parser.add_argument("--prob-threshold", type=float, default=0.53)
    ens_parser.add_argument("--fee-rate", type=float, default=0.00075)
    ens_parser.add_argument("--base-leverage", type=float, default=1.0)

    # 7. Run 命令
    subparsers.add_parser("run", help="執行交易策略")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "collect":
        asyncio.run(collect_ohlcv(args))
    elif args.command == "batch-collect":
        asyncio.run(batch_collect_ohlcv(args))
    elif args.command == "clean":
        clean_ohlcv(args)
    elif args.command == "feature":
        generate_features(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "wf-eval":
        run_walk_forward(args)
    elif args.command == "backtest":
        run_backtest_cli(args)
    elif args.command == "batch-feature":
        run_batch_feature_cli(args)
    elif args.command == "ensemble-backtest":
        run_ensemble_backtest_cli(args)
    else:
        print(f"[CLI] 執行命令: {args.command}")


if __name__ == "__main__":
    main()