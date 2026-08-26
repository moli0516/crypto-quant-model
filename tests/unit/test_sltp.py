"""
HighFreqSLTPBacktester 1m 高頻 K 線碰撞引擎單元測試
"""
import pytest
import pandas as pd
from scripts.sltp_backtest import HighFreqSLTPBacktester


@pytest.fixture
def backtester():
    return HighFreqSLTPBacktester(
        initial_balance=200.0,
        position_pct=0.10,
        fee_rate=0.0008,
        holding_hours=12
    )


def test_simulate_trade_take_profit(backtester):
    """測試觸發 Take Profit (TP) 邏輯"""
    entry_price = 100.0
    tp_pct = 0.025  # TP 2.5% -> $102.5
    sl_pct = 0.040  # SL 4.0% -> $96.0

    # 模擬 1m K 線：第 2 根 High 觸及 $103.0 (TP)
    klines_1m = pd.DataFrame([
        {"high": 101.0, "low": 99.5, "close": 100.5},
        {"high": 103.0, "low": 100.0, "close": 102.8},
    ])

    res = backtester.simulate_single_trade(entry_price, klines_1m, tp_pct, sl_pct)

    assert res["exit_type"] == "TP"
    assert res["duration_mins"] == 2
    assert round(res["pnl_pct"], 6) == round(0.025 - 0.0008, 6)


def test_simulate_trade_stop_loss(backtester):
    """測試觸發 Stop Loss (SL) 邏輯"""
    entry_price = 100.0
    tp_pct = 0.025
    sl_pct = 0.040

    # 模擬 1m K 線：第 1 根 Low 降至 $95.0 (SL)
    klines_1m = pd.DataFrame([
        {"high": 100.5, "low": 95.0, "close": 95.5},
    ])

    res = backtester.simulate_single_trade(entry_price, klines_1m, tp_pct, sl_pct)

    assert res["exit_type"] == "SL"
    assert res["duration_mins"] == 1
    assert round(res["pnl_pct"], 6) == round(-0.040 - 0.0008, 6)


def test_simulate_trade_simultaneous_sl_tp_collision(backtester):
    """【關鍵測試】同 1 分鐘內同時觸及 SL 與 TP，採極端保守原則視為 SL 觸發"""
    entry_price = 100.0
    tp_pct = 0.025  # TP: $102.5
    sl_pct = 0.040  # SL: $96.0

    # 劇烈波動 K 線：High 105.0 (超過 TP) 且 Low 90.0 (跌破 SL)
    klines_1m = pd.DataFrame([
        {"high": 105.0, "low": 90.0, "close": 98.0},
    ])

    res = backtester.simulate_single_trade(entry_price, klines_1m, tp_pct, sl_pct)

    # 保守風控條款判定
    assert res["exit_type"] == "SL"
    assert round(res["pnl_pct"], 6) == round(-0.040 - 0.0008, 6)


def test_simulate_trade_timeout(backtester):
    """測試超過持倉上限 12H (720分鐘) 未觸發 SL/TP 之 Timeout 自然平倉邏輯"""
    entry_price = 100.0
    tp_pct = 0.05
    sl_pct = 0.05

    # 產生 720 根極度平穩的 1m K 線
    data = [{"high": 101.0, "low": 99.0, "close": 102.0} for _ in range(720)]
    klines_1m = pd.DataFrame(data)

    res = backtester.simulate_single_trade(entry_price, klines_1m, tp_pct, sl_pct)

    assert res["exit_type"] == "TIMEOUT"
    assert res["duration_mins"] == 720
    # 原始 PnL = (102 - 100) / 100 = 2.0%, 扣除手續費 0.08% = 1.92% (0.0192)
    assert round(res["pnl_pct"], 6) == round(0.02 - 0.0008, 6)