"""Reconcile filled Binance Demo buys into the local order state."""

import argparse
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
import ccxt.async_support as ccxt

BASE_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = BASE_DIR / "logs"
TARGET_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "NEARUSDT", "FETUSDT", "ICPUSDT",
    "UNIUSDT", "SUIUSDT", "ARBUSDT", "OPUSDT", "BCHUSDT", "LTCUSDT",
    "AVAXUSDT", "LINKUSDT", "XRPUSDT", "ADAUSDT", "DOTUSDT", "MATICUSDT",
    "ATOMUSDT", "ETCUSDT",
]
TAKE_PROFIT_PCT = 0.04
STOP_LOSS_PCT = 0.01


def _clean(value: str) -> str:
    return value.strip("'\" \t\r\n")


def _state_file() -> Path:
    environment = os.getenv("BINANCE_TRADING_ENV", "demo").strip().lower()
    if environment not in {"demo", "testnet"}:
        raise RuntimeError("BINANCE_TRADING_ENV must be 'demo' or 'testnet'")
    return LOGS_DIR / f"paper_account_state_{environment}.json"


def _symbol(value: str) -> str:
    return value.replace("/", "").upper()


def _datetime(timestamp: Any) -> str:
    if timestamp is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return datetime.fromtimestamp(float(timestamp) / 1000.0).strftime("%Y-%m-%d %H:%M:%S")


def _order_list_id(order: Dict[str, Any]) -> str:
    return str((order.get("info") or {}).get("orderListId") or "")


def _filled_buys(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        order for order in orders
        if str(order.get("side", "")).lower() == "buy"
        and float(order.get("filled") or 0.0) > 0.0
    ]


def _build_record(
    buy: Dict[str, Any],
    related_orders: List[Dict[str, Any]],
) -> Dict[str, Any]:
    symbol = _symbol(str(buy.get("symbol", "")))
    quantity = float(buy.get("filled") or buy.get("amount") or 0.0)
    price = float(buy.get("average") or buy.get("price") or 0.0)
    cost = float(buy.get("cost") or quantity * price)
    oco_orders = [
        order for order in related_orders
        if str(order.get("side", "")).lower() == "sell" and _order_list_id(order)
    ]
    oco_ids = [str(order.get("id")) for order in oco_orders if order.get("id")]
    oco_list_ids = [_order_list_id(order) for order in oco_orders if _order_list_id(order)]
    oco_list_id = oco_list_ids[0] if oco_list_ids else None
    return {
        "trade_id": f"RECONCILED-{buy.get('id')}-{symbol}",
        "symbol": symbol,
        "status": "OPEN",
        "oco_status": "ACTIVE" if oco_list_id else "FAILED",
        "entry_time": _datetime(buy.get("timestamp")),
        "entry_price": price,
        "qty": quantity,
        "margin_usd": round(cost, 4),
        "entry_fee_usd": round(cost * 0.001, 8),
        "prob": None,
        "tp_price": round(price * (1.0 + TAKE_PROFIT_PCT), 8),
        "sl_price": round(price * (1.0 - STOP_LOSS_PCT), 8),
        "buy_order_id": str(buy.get("id")),
        "oco_order_list_id": oco_list_id,
        "oco_order_ids": oco_ids,
        "reconciled": True,
    }


def backfill_risk_levels(state: List[Dict[str, Any]]) -> int:
    updated = 0
    for record in state:
        entry_price = float(record.get("entry_price") or 0.0)
        if entry_price <= 0.0:
            continue
        changed = False
        if record.get("tp_price") is None:
            record["tp_price"] = round(entry_price * (1.0 + TAKE_PROFIT_PCT), 8)
            changed = True
        if record.get("sl_price") is None:
            record["sl_price"] = round(entry_price * (1.0 - STOP_LOSS_PCT), 8)
            changed = True
        if changed:
            updated += 1
    return updated


def _load_state(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as source:
        data = json.load(source)
    if isinstance(data, dict) and "open_positions" in data:
        data = data["open_positions"]
    return data if isinstance(data, list) else []


def _save_state(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(records, target, indent=4, ensure_ascii=False)
        target.write("\n")
    temporary.replace(path)


async def fetch_orders(max_pages: int, since_hours: int) -> List[Dict[str, Any]]:
    load_dotenv(override=True)
    environment = os.getenv("BINANCE_TRADING_ENV", "demo").strip().lower()
    prefix = "BINANCE_SPOT_DEMO" if environment == "demo" else "BINANCE_SPOT_TESTNET"
    api_key = _clean(os.getenv(f"{prefix}_API_KEY", ""))
    api_secret = _clean(os.getenv(f"{prefix}_API_SECRET", ""))
    if not api_key or not api_secret:
        raise RuntimeError(f"{prefix}_API_KEY and {prefix}_API_SECRET must be configured")

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

    since = int((datetime.now(timezone.utc) - timedelta(hours=since_hours)).timestamp() * 1000)
    try:
        await exchange.load_markets()
        collected: Dict[str, Dict[str, Any]] = {}
        for raw_symbol in TARGET_SYMBOLS:
            symbol = raw_symbol.replace("USDT", "/USDT")
            try:
                orders = await exchange.fetch_orders(symbol, since=since, limit=1000)
            except ccxt.ExchangeError:
                continue
            for order in orders:
                order_id = str(order.get("id") or "")
                if order_id:
                    collected[order_id] = order
        return list(collected.values())
    finally:
        await exchange.close()


def reconcile(orders: List[Dict[str, Any]], state: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing_ids = {str(item.get("buy_order_id")) for item in state}
    buys_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for buy in _filled_buys(orders):
        buys_by_symbol.setdefault(_symbol(str(buy.get("symbol", ""))), []).append(buy)
    for buys in buys_by_symbol.values():
        buys.sort(key=lambda item: item.get("timestamp") or 0)

    additions = []
    for buy in sorted(_filled_buys(orders), key=lambda item: item.get("timestamp") or 0):
        buy_id = str(buy.get("id") or "")
        if not buy_id or buy_id in existing_ids:
            continue
        symbol = _symbol(str(buy.get("symbol", "")))
        buy_timestamp = buy.get("timestamp") or 0
        next_buy_timestamp = next(
            (
                candidate.get("timestamp") or 0
                for candidate in buys_by_symbol[symbol]
                if (candidate.get("timestamp") or 0) > buy_timestamp
            ),
            None,
        )
        related = [
            order for order in orders
            if _symbol(str(order.get("symbol", ""))) == symbol
            and (order.get("timestamp") or 0) >= buy_timestamp
            and (next_buy_timestamp is None or (order.get("timestamp") or 0) < next_buy_timestamp)
        ]
        additions.append(_build_record(buy, related))
        existing_ids.add(buy_id)
    return additions


async def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile Binance Demo buys into local state.")
    parser.add_argument("--since-hours", type=int, default=48)
    parser.add_argument("--max-pages", type=int, default=1, help="Reserved for CLI compatibility.")
    parser.add_argument("--apply", action="store_true", help="Write imported records to local state.")
    args = parser.parse_args()
    if args.since_hours < 1 or args.max_pages < 1:
        raise SystemExit("since-hours and max-pages must be at least 1")

    path = _state_file()
    state = _load_state(path)
    updated = backfill_risk_levels(state)
    orders = await fetch_orders(args.max_pages, args.since_hours)
    additions = reconcile(orders, state)
    print(f"Online orders fetched: {len(orders)}")
    print(f"Missing filled buys: {len(additions)}")
    print(f"Risk levels backfilled: {updated}")
    for record in additions:
        print(
            f"{record['symbol']} buy={record['buy_order_id']} qty={record['qty']} "
            f"oco={record['oco_status']}"
        )
    if args.apply and (additions or updated):
        _save_state(path, state + additions)
        print(f"Applied {len(additions)} new record(s) and updated {updated} record(s) in {path}")
    elif args.apply:
        print(f"No changes needed in {path}")
    else:
        print("Dry run only. Re-run with --apply to update local state.")


if __name__ == "__main__":
    asyncio.run(main())
