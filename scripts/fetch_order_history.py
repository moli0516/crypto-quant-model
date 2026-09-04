"""Fetch historical Binance Spot Testnet orders for account reconciliation."""

import argparse
import asyncio
import csv
import json
import os
import sys
from typing import Any, Dict, List

import ccxt.async_support as ccxt
from dotenv import load_dotenv

from src.config import TARGET_SYMBOLS


def _clean(value: str) -> str:
    return value.strip("'\" \t\r\n")


def _summary(order: Dict[str, Any]) -> Dict[str, Any]:
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
        "average": order.get("average"),
        "cost": order.get("cost"),
        "timestamp": order.get("timestamp"),
        "datetime": order.get("datetime"),
        "order_list_id": info.get("orderListId"),
        "client_order_id": order.get("clientOrderId") or info.get("clientOrderId"),
    }


async def _fetch_symbol_history(
    exchange: Any, symbol: str, page_size: int, max_pages: int
) -> List[Dict[str, Any]]:
    history: Dict[str, Dict[str, Any]] = {}
    params: Dict[str, Any] = {}
    for _ in range(max_pages):
        orders = await exchange.fetch_orders(symbol, limit=page_size, params=params)
        if not orders:
            break
        for order in orders:
            history[str(order.get("id"))] = _summary(order)
        oldest = min(
            (order.get("timestamp") for order in orders if order.get("timestamp") is not None),
            default=None,
        )
        if len(orders) < page_size or oldest is None:
            break
        params = {"endTime": int(oldest) - 1}
    return list(history.values())


async def fetch_order_history(symbol: str | None, max_pages: int) -> List[Dict[str, Any]]:
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
        "options": {"defaultType": "spot", "adjustForTimeDifference": True},
    })
    if environment == "demo":
        exchange.enable_demo_trading(True)
    else:
        exchange.set_sandbox_mode(True)
    try:
        await exchange.load_markets()
        symbols = [symbol.upper().replace("/", "")] if symbol else TARGET_SYMBOLS
        all_orders: List[Dict[str, Any]] = []
        for raw_symbol in symbols:
            formatted_symbol = raw_symbol.replace("USDT", "/USDT")
            try:
                all_orders.extend(
                    await _fetch_symbol_history(exchange, formatted_symbol, 1000, max_pages)
                )
            except ccxt.ExchangeError as exc:
                print(f"Skipping {formatted_symbol}: {exc}", file=sys.stderr)
        return sorted(
            all_orders,
            key=lambda order: order.get("timestamp") or 0,
            reverse=True,
        )
    finally:
        await exchange.close()


def _write_csv(path: str, orders: List[Dict[str, Any]]) -> None:
    fields = list(orders[0]) if orders else ["id", "symbol", "status"]
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(orders)


def _print_table(orders: List[Dict[str, Any]]) -> None:
    print(f"Found {len(orders)} historical order(s).")
    for order in orders:
        print(
            f"{order['datetime'] or '-'} {order['symbol']} "
            f"{order['side']} {order['type']} status={order['status']} "
            f"id={order['id']} filled={order['filled']} "
            f"average={order['average']} oco={order['order_list_id']}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch historical Binance Spot Testnet orders."
    )
    parser.add_argument(
        "--symbol",
        help="Only fetch one symbol, for example ADAUSDT or ADA/USDT.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum 1000-order pages per symbol (default: 10).",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--csv", metavar="PATH", help="Write results to a CSV file.")
    args = parser.parse_args()
    if args.max_pages < 1:
        raise SystemExit("--max-pages must be at least 1")

    try:
        orders = await fetch_order_history(args.symbol, args.max_pages)
    except (ccxt.AuthenticationError, ccxt.NetworkError, RuntimeError) as exc:
        raise SystemExit(f"Unable to fetch order history: {exc}") from exc

    if args.csv:
        _write_csv(args.csv, orders)
        print(f"Wrote {len(orders)} order(s) to {args.csv}")
    elif args.json:
        print(json.dumps(orders, indent=2, ensure_ascii=True))
    else:
        _print_table(orders)


if __name__ == "__main__":
    asyncio.run(main())