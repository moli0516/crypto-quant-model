"""List all pending Binance Spot Demo/Testnet orders without changing account state."""

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List

import ccxt.async_support as ccxt
from dotenv import load_dotenv


def _clean(value: str) -> str:
    return value.strip("'\" \t\r\n")


def _order_summary(order: Dict[str, Any]) -> Dict[str, Any]:
    info = order.get("info") or {}
    return {
        "id": order.get("id"),
        "symbol": order.get("symbol"),
        "side": order.get("side"),
        "type": order.get("type"),
        "status": order.get("status"),
        "amount": order.get("amount"),
        "filled": order.get("filled"),
        "remaining": order.get("remaining"),
        "price": order.get("price"),
        "stop_price": order.get("stopPrice"),
        "cost": order.get("cost"),
        "timestamp": order.get("timestamp"),
        "datetime": order.get("datetime"),
        "order_list_id": info.get("orderListId"),
        "client_order_id": order.get("clientOrderId") or info.get("clientOrderId"),
    }


async def fetch_pending_orders(symbol: str | None = None) -> List[Dict[str, Any]]:
    load_dotenv(override=True)
    environment = os.getenv("BINANCE_TRADING_ENV", "demo").strip().lower()
    prefix = "BINANCE_SPOT_DEMO" if environment == "demo" else "BINANCE_SPOT_TESTNET"
    api_key = _clean(os.getenv(f"{prefix}_API_KEY", ""))
    api_secret = _clean(os.getenv(f"{prefix}_API_SECRET", ""))
    if not api_key or not api_secret:
        raise RuntimeError(
            f"{prefix}_API_KEY and {prefix}_API_SECRET must be configured"
        )

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
            "warnOnFetchOpenOrdersWithoutSymbol": False,
            "fetchOpenOrders": {"warnWithoutSymbol": False},
        },
    })
    if environment == "demo":
        exchange.enable_demo_trading(True)
    else:
        exchange.set_sandbox_mode(True)
    try:
        await exchange.load_markets()
        if symbol:
            normalized_symbol = symbol.upper().replace("/", "")
            formatted_symbol = normalized_symbol.replace("USDT", "/USDT")
        else:
            formatted_symbol = None
        orders = await exchange.fetch_open_orders(symbol=formatted_symbol)
        return [_order_summary(order) for order in orders]
    finally:
        await exchange.close()


def _print_table(orders: List[Dict[str, Any]]) -> None:
    if not orders:
        print("No pending Spot Testnet orders found.")
        return

    print(f"Found {len(orders)} pending Spot Testnet order(s):")
    for order in orders:
        print(
            f"- {order['symbol']} {order['side']} {order['type']} "
            f"id={order['id']} status={order['status']} "
            f"amount={order['amount']} filled={order['filled']} "
            f"price={order['price']} stop={order['stop_price']} "
            f"oco={order['order_list_id']}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch all pending Binance Spot Testnet orders."
    )
    parser.add_argument(
        "--symbol",
        help="Only fetch one symbol, for example BTCUSDT or BTC/USDT.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a summary table.",
    )
    args = parser.parse_args()

    try:
        orders = await fetch_pending_orders(args.symbol)
    except (ccxt.AuthenticationError, ccxt.NetworkError, ccxt.ExchangeError, RuntimeError) as exc:
        raise SystemExit(f"Unable to fetch pending orders: {exc}") from exc

    if args.json:
        print(json.dumps(orders, indent=2, ensure_ascii=True))
    else:
        _print_table(orders)


if __name__ == "__main__":
    asyncio.run(main())