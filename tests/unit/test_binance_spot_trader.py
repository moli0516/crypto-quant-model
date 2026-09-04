from src.live.binance_spot_trader import BinanceSpotTrader


def test_find_exit_order_requires_filled_sell_after_entry():
    local_order = {
        "entry_time": "2026-09-04 13:00:09",
        "oco_order_ids": ["sell-1"],
    }
    closed_orders = [
        {"id": "sell-1", "side": "sell", "filled": 0.0, "timestamp": 1_000_000_000_000},
        {"id": "sell-1", "side": "sell", "filled": 2.0, "timestamp": 1_788_529_210_000},
    ]

    result = BinanceSpotTrader._find_exit_order(local_order, closed_orders)

    assert result is closed_orders[1]


def test_apply_close_data_calculates_net_realized_pnl():
    trader = BinanceSpotTrader.__new__(BinanceSpotTrader)
    local_order = {
        "entry_price": 100.0,
        "qty": 2.0,
        "margin_usd": 200.0,
        "entry_fee_usd": 0.20,
    }
    exit_order = {
        "symbol": "TEST/USDT",
        "filled": 2.0,
        "average": 105.0,
        "fees": [{"currency": "USDT", "cost": 0.21}],
    }

    trader._apply_close_data(local_order, exit_order)

    assert local_order["status"] == "CLOSED"
    assert local_order["exit_price"] == 105.0
    assert local_order["exit_qty"] == 2.0
    assert local_order["fee_usd"] == 0.41
    assert local_order["pnl_usd"] == 9.59
    assert local_order["pnl_pct"] == 0.04795