import asyncio
from datetime import datetime, timedelta

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


def test_close_expired_position_cancels_oco_and_market_sells(monkeypatch):
    class FakeExchange:
        def __init__(self):
            self.cancelled = []
            self.sold = None

        async def fetch_open_orders(self, symbol):
            return [{"id": "oco-child", "symbol": symbol}]

        async def cancel_order(self, order_id, symbol):
            self.cancelled.append((order_id, symbol))

        async def load_markets(self):
            return None

        def market(self, symbol):
            return {"base": "BTC"}

        async def fetch_balance(self):
            return {"BTC": {"free": 0.25}}

        def amount_to_precision(self, symbol, amount):
            return f"{amount:.8f}"

        async def create_market_sell_order(self, symbol, amount):
            self.sold = (symbol, float(amount))
            return {"filled": float(amount), "average": 105.0, "fees": []}

    trader = BinanceSpotTrader.__new__(BinanceSpotTrader)
    trader.exchange = FakeExchange()
    trader.local_orders = [{
        "symbol": "BTCUSDT",
        "status": "OPEN",
        "entry_time": (datetime.now() - timedelta(hours=13)).strftime("%Y-%m-%d %H:%M:%S"),
        "entry_price": 100.0,
        "qty": 0.25,
        "margin_usd": 25.0,
        "entry_fee_usd": 0.025,
        "oco_order_ids": ["oco-child"],
    }]
    monkeypatch.setattr(trader, "_save_local_orders", lambda: None)
    monkeypatch.setattr(trader, "_append_realized_trade", lambda order: None)
    monkeypatch.setattr(trader, "_notify_closed_order", lambda order: None)

    asyncio.run(trader._close_expired_positions())

    assert trader.exchange.cancelled == [("oco-child", "BTC/USDT")]
    assert trader.exchange.sold == ("BTC/USDT", 0.25)
    assert trader.local_orders[0]["status"] == "CLOSED"
    assert trader.local_orders[0]["close_reason"] == "H12_TIMEOUT"