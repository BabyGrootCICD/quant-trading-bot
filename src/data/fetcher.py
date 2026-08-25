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


def available_symbols(exchange, symbols: list[str]) -> list[str]:
    """The subset of `symbols` the exchange actually lists.

    The universe is deliberately wider than the venue is guaranteed to carry --
    trade frequency scales with it -- so a pair that is unlisted, delisted or
    named differently here must degrade quietly rather than take the run down.
    Anything dropped still flows through the rest of the pipeline as "no
    candles", which every downstream stage already tolerates.
    """
    try:
        markets = exchange.load_markets()
    except Exception as e:
        print(f"  Could not load markets ({e}); attempting all {len(symbols)} symbols")
        return list(symbols)

    listed = [s for s in symbols if s in markets]
    missing = [s for s in symbols if s not in markets]
    if missing:
        print(f"  Not listed on {EXCHANGE_ID}, skipping {len(missing)}: {', '.join(missing)}")
    return listed


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


def drop_incomplete_bars(candles: list[list], now_ms: int, bar_ms: int = 3_600_000) -> list[list]:
    """Keep only bars whose interval has fully elapsed.

    Exchanges return the bar currently forming as the last element, and it was
    being stored like any other. Everything downstream then inherited a partial
    bar as its newest row:

      * `engineer_features` computed that row's features from a fraction of an
        hour's data. Measured against the distribution of completed bars, five
        minutes in, `volume_ratio` and `volume_ratio_48` land at the **0th
        percentile** -- outside anything the model saw in training. The run that
        prompted this was 2.4 minutes in.
      * `model.predict()` masks on NaN features, not on the NaN *target*, so
        that row was scored, and `upsert_latest_prediction` takes the newest
        timestamp -- making the partial bar THE traded prediction.
      * the label is `sign(return of the NEXT bar)`, so a forecast made from
        the forming bar is a forecast for a bar that has not started yet, while
        `horizon_fraction()` discounts it as though it applied to the current
        one. The two disagreed by a whole bar.

    With only completed bars stored, the newest feature row is the last closed
    bar T and its forecast is for T+1 -- the bar now forming, which is exactly
    what the engine trades and exactly what `horizon_fraction()` measures.
    """
    if bar_ms <= 0:
        return candles
    return [c for c in candles if int(c[0]) + bar_ms <= now_ms]


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


def latest_stored_timestamp(client, symbol: str) -> int | None:
    """Newest candle timestamp already stored for `symbol`, or None."""
    resp = (
        client.table("candles")
        .select("timestamp")
        .eq("symbol", symbol)
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        return None
    return int(resp.data[0]["timestamp"])


def resolve_since(latest_ts: int | None, backfill_start_ms: int, overlap_bars: int = 2,
                  bar_ms: int = 3_600_000) -> int:
    """Where to resume fetching from.

    Full backfill only when the symbol has no candles. Otherwise resume a
    couple of bars before the newest stored one, so the most recent (possibly
    still-forming) candle gets corrected without re-downloading two years of
    history every single hour.
    """
    if latest_ts is None:
        return backfill_start_ms
    return max(backfill_start_ms, latest_ts - overlap_bars * bar_ms)


def fetch_and_store(exchange, client, symbol: str) -> int:
    backfill_start = exchange.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=365 * HISTORY_YEARS)).isoformat()
    )
    since = resolve_since(latest_stored_timestamp(client, symbol), backfill_start)
    print(f"  Fetching {symbol} since {datetime.fromtimestamp(since / 1000, tz=timezone.utc).isoformat()}...")
    candles = fetch_ohlcv_all(exchange, symbol, TIMEFRAME, since)
    fetched = len(candles)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    candles = drop_incomplete_bars(candles, now_ms)
    dropped = fetched - len(candles)
    print(f"  Fetched {fetched} candles"
          + (f" ({dropped} still forming, not stored)" if dropped else ""))
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
    client = get_client()

    symbols = available_symbols(exchange, SYMBOLS)
    print(f"  Universe: {len(symbols)}/{len(SYMBOLS)} symbols")

    total = 0
    for symbol in symbols:
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
