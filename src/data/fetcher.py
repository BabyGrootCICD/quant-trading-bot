import time
import sys
import os
from datetime import datetime, timedelta, timezone

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import SYMBOLS, TIMEFRAME, HISTORY_YEARS, EXCHANGE_ID
from src.data.supabase_client import get_client


def create_exchange():
    exchange_class = getattr(ccxt, EXCHANGE_ID, None)
    if exchange_class is None:
        raise ValueError(f"Unknown exchange: {EXCHANGE_ID}")
    return exchange_class({"enableRateLimit": True})


def fetch_ohlcv_all(exchange, symbol: str, timeframe: str, since_ms: int) -> list[list]:
    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        since_ms = candles[-1][0] + 1
        if len(candles) < 1000:
            break
        time.sleep(0.5)
    return all_candles


def candles_to_rows(symbol: str, candles: list[list]) -> list[dict]:
    return [
        {
            "symbol": symbol,
            "timestamp": c[0],
            "open": c[1],
            "high": c[2],
            "low": c[3],
            "close": c[4],
            "volume": c[5],
        }
        for c in candles
    ]


def upsert_candles(client, rows: list[dict]) -> int:
    if not rows:
        return 0
    batch_size = 500
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        client.table("candles").upsert(batch, on_conflict="symbol,timestamp").execute()
        total += len(batch)
    return total


def fetch_and_store(exchange, client, symbol: str) -> int:
    since = exchange.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=365 * HISTORY_YEARS)).isoformat()
    )
    print(f"  Fetching {symbol} since {datetime.fromtimestamp(since / 1000, tz=timezone.utc).isoformat()}...")
    candles = fetch_ohlcv_all(exchange, symbol, TIMEFRAME, since)
    print(f"  Fetched {len(candles)} candles")
    rows = candles_to_rows(symbol, candles)
    upserted = upsert_candles(client, rows)
    print(f"  Upserted {upserted} rows")
    return upserted


def main():
    print("=" * 60)
    print("Quant Bot - Hourly Data Fetcher")
    print(f"Exchange: {EXCHANGE_ID}")
    print("=" * 60)
    exchange = create_exchange()
    exchange.load_markets()
    client = get_client()

    total = 0
    for symbol in SYMBOLS:
        try:
            count = fetch_and_store(exchange, client, symbol)
            total += count
        except Exception as e:
            print(f"  ERROR fetching {symbol}: {e}")

    print("=" * 60)
    print(f"Done. Total rows upserted: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
