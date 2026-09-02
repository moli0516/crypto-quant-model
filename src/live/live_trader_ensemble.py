"""
src/live/live_trader_ensemble_spot.py
==============================================================================
Ensemble Live Trader for Binance Spot Testnet
- Ensemble (XGB + LGB) Soft Voting
- Top-K 最高機率幣種篩選
- 市價買入 + 現貨 OCO (TP Limit / SL Stop-Limit)
==============================================================================
"""

import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd

from src.config import (
    ACTIVE_SYMBOLS,
    SYMBOL_BLACKLIST,
    PROB_THRESHOLD,
    POSITION_SIZE_RATIO,
    MODEL_PATHS,
    ENSEMBLE_WEIGHTS,
    TOP_K_SIGNALS,
    USE_TOP_K_FILTER,
    DEFAULT_TAKE_PROFIT_PCT,
    DEFAULT_STOP_LOSS_PCT,
    INFERENCE_LOG_FILE,
)

from src.live.live_pipeline import LiveDataPipeline
from src.models.ensemble import EnsemblePredictor
from src.live.binance_spot_trader import BinanceSpotTrader
from src.live.utils.notifier import notify_inference_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EnsembleSpotLiveTrader:
    """
    Binance Spot Testnet 實盤版 Ensemble 交易器
    """

    def __init__(
        self,
        prob_threshold: float = PROB_THRESHOLD,
        top_k: int = TOP_K_SIGNALS,
        dry_run: bool = False,          # True = 只預測不下單
    ):
        self.symbols = ACTIVE_SYMBOLS
        self.prob_threshold = prob_threshold
        self.top_k = top_k
        self.use_top_k = USE_TOP_K_FILTER
        self.dry_run = dry_run

        self.pipeline = LiveDataPipeline(symbols=self.symbols)

        logger.info(f"🚀 初始化 Ensemble 預測器 (Weights: {ENSEMBLE_WEIGHTS})")
        self.predictor = EnsemblePredictor(
            model_paths=MODEL_PATHS,
            weights=ENSEMBLE_WEIGHTS,
        )

        # 只有非 dry_run 才初始化真實交易所
        self.trader = None
        if not self.dry_run:
            self.trader = BinanceSpotTrader()
            logger.info("✅ BinanceSpotTrader (Testnet) 已就緒")

        logger.info(
            f"✅ EnsembleSpotLiveTrader 初始化完成 | "
            f"標的數: {len(self.symbols)} | 門檻: {self.prob_threshold} | "
            f"Top-K: {self.top_k if self.use_top_k else 'OFF'} | "
            f"模式: {'DRY-RUN' if self.dry_run else 'LIVE TESTNET'}"
        )

    async def _execute_inference_cycle(self) -> None:
        logger.info("🔄 [整點] 開始 Ensemble 推論 + Top-K + OCO 下單週期...")

        # 1. 抓取最新特徵
        latest_features_df = await self.pipeline.fetch_and_process()
        if latest_features_df.empty:
            logger.error("❌ 無法取得特徵，跳過本次週期")
            return

        # 2. 黑名單二次過濾
        if "symbol" in latest_features_df.columns:
            latest_features_df = latest_features_df[
                ~latest_features_df["symbol"].isin(SYMBOL_BLACKLIST)
            ].copy()

        # 3. Ensemble Soft Voting
        try:
            preds_proba = self.predictor.predict_proba(latest_features_df)
            latest_features_df["pred_proba"] = preds_proba
        except Exception as e:
            logger.error(f"❌ Ensemble 推論失敗: {e}")
            return

        # 4. Top-K 剪枝（只保留最高機率的 K 個達標幣種）
        execution_df = latest_features_df.copy()

        if self.use_top_k:
            candidates = execution_df[
                execution_df["pred_proba"] >= self.prob_threshold
            ].sort_values("pred_proba", ascending=False)

            if len(candidates) > self.top_k:
                top_symbols = candidates.head(self.top_k)["symbol"].tolist()
                logger.info(
                    f"🔥 [Top-K] 達標 {len(candidates)} 個 → 精選 Top-{self.top_k}: {top_symbols}"
                )
                # 把非 Top-K 的機率壓成 0，避免被下單
                execution_df.loc[
                    ~execution_df["symbol"].isin(top_symbols), "pred_proba"
                ] = 0.0

        # 5. 寫入推論日誌（保留全市場原始機率，方便事後分析）
        try:
            self.pipeline.save_inference_log(
                latest_features_df, prob_threshold=self.prob_threshold
            )
        except Exception as e:
            logger.error(f"❌ 寫入推論日誌失敗: {e}")

        # 6. Telegram 訊號卡片
        try:
            notify_inference_signals(INFERENCE_LOG_FILE, prob_threshold=self.prob_threshold)
        except Exception as e:
            logger.error(f"❌ Telegram 推播失敗: {e}")

        # 7. 實際下單（或 dry-run）
        if self.dry_run:
            long_candidates = execution_df[
                execution_df["pred_proba"] >= self.prob_threshold
            ]
            if not long_candidates.empty:
                logger.info("📝 [DRY-RUN] 本輪達標標的（不會真實下單）:")
                for _, row in long_candidates.iterrows():
                    logger.info(
                        f"   • {row['symbol']} | 機率 {row['pred_proba']:.4f} | "
                        f"價格 ${row['close']:.4f}"
                    )
            else:
                logger.info("📝 [DRY-RUN] 本輪無達標標的")
            return

        # ===== LIVE 下單 =====
        for _, row in execution_df.iterrows():
            symbol = row["symbol"]
            price = float(row["close"])
            prob = float(row["pred_proba"])

            if prob < self.prob_threshold:
                continue

            # 檢查是否已有該幣種持倉（避免重複開倉）
            open_symbols = [
                p["symbol"] for p in self.trader.state.get("open_positions", [])
            ]
            if symbol in open_symbols:
                logger.info(f"⏭️ {symbol} 已有持倉，跳過")
                continue

            logger.info(
                f"🚀 準備開倉 {symbol} | 機率 {prob:.4f} | 價格 ${price:.4f} | "
                f"TP {DEFAULT_TAKE_PROFIT_PCT*100:.1f}% / SL {DEFAULT_STOP_LOSS_PCT*100:.1f}%"
            )

            success = await self.trader.execute_spot_buy_with_oco(
                symbol=symbol,
                current_price=price,
                prob=prob,
            )
            if success:
                logger.info(f"✅ {symbol} 市價買入 + OCO 掛單成功")
            else:
                logger.warning(f"⚠️ {symbol} 下單失敗")

    async def run_scheduler(self) -> None:
        """整點對齊排程（每小時 00 分 01 秒觸發）"""
        mode = "DRY-RUN" if self.dry_run else "LIVE TESTNET"
        logger.info(f"🚀 EnsembleSpotLiveTrader 已啟動 ({mode})，等待下一個整點...")

        while True:
            now = datetime.now()
            next_hour = (now + timedelta(hours=1)).replace(
                minute=0, second=1, microsecond=0
            )
            wait_seconds = (next_hour - now).total_seconds()
            logger.info(f"⏳ 休眠 {wait_seconds:.0f} 秒後觸發下一次推論...")
            await asyncio.sleep(wait_seconds)

            try:
                await self._execute_inference_cycle()
            except Exception as e:
                logger.error(f"❌ 推論週期發生未預期錯誤: {e}", exc_info=True)

    async def close(self):
        if self.trader is not None:
            await self.trader.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble Spot Testnet Trader")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只做預測與日誌，不實際下單（強烈建議先跑這個）",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只跑一次推論就結束（方便測試）",
    )
    args = parser.parse_args()

    trader = EnsembleSpotLiveTrader(dry_run=args.dry_run)

    async def main():
        try:
            if args.once:
                await trader._execute_inference_cycle()
            else:
                await trader.run_scheduler()
        finally:
            await trader.close()

    asyncio.run(main())