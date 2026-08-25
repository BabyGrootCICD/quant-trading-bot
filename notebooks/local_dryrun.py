"""End-to-end dry run: real code, real market data, in-memory database.

Verifies that the pipeline can actually open, hold and close paper trades --
without Supabase credentials and without touching the live books. Everything
except the database and the exchange is the production code path:
`engineer_features`, the real trainer, and `engine.main()`.

    python notebooks/local_dryrun.py            # taker cost (production default)
    EXECUTION_MODE=maker python notebooks/local_dryrun.py
    MIN_EDGE_MARGIN=0 python notebooks/local_dryrun.py
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import ccxt

SYMBOLS = os.getenv("DRYRUN_SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT,DOGE/USDT").split(",")
PAGES = int(os.getenv("DRYRUN_PAGES", "25"))          # 100 hourly bars per page
VENUE = os.getenv("DRYRUN_VENUE", "okx")
# Where in the hour to pretend the cycle is running. GitHub drops scheduled
# runs at effectively random points in the bar, and the horizon discount means
# that choice dominates whether anything is tradeable -- so pin it to compare
# like with like. Unset = real wall clock.
AT_MINUTE = os.getenv("DRYRUN_AT_MINUTE")


# --------------------------------------------------------------- fake database

class _Result:
    def __init__(self, data): self.data = data


class _Query:
    """Enough of the PostgREST surface for the pipeline's actual queries."""

    def __init__(self, store, table, op, payload=None, on_conflict=None):
        self.store, self.table, self.op = store, table, op
        self.payload, self.on_conflict = payload, on_conflict
        self.cols = None
        self.filters, self.excl = {}, {}
        self.order_by, self.desc = None, False
        self.lim, self.rng = None, None

    def select(self, cols="*"): self.cols = cols; return self
    def eq(self, c, v): self.filters[c] = v; return self
    def neq(self, c, v): self.excl[c] = v; return self
    def order(self, c, desc=False, **kw): self.order_by, self.desc = c, desc; return self
    def limit(self, n): self.lim = n; return self
    def range(self, a, b): self.rng = (a, b); return self

    def execute(self):
        rows = self.store.data.setdefault(self.table, [])
        if self.op == "insert":
            for r in ([self.payload] if isinstance(self.payload, dict) else self.payload):
                r = dict(r); r["id"] = self.store.next_id(self.table)
                r.setdefault("created_at", self.store.tick())
                rows.append(r)
            return _Result([])
        if self.op == "upsert":
            keys = [k.strip() for k in (self.on_conflict or "").split(",") if k.strip()]
            for r in ([self.payload] if isinstance(self.payload, dict) else self.payload):
                hit = None
                if keys:
                    hit = next((x for x in rows if all(x.get(k) == r.get(k) for k in keys)), None)
                if hit is not None:
                    hit.update(r)
                else:
                    r = dict(r); r["id"] = self.store.next_id(self.table)
                    r.setdefault("created_at", self.store.tick())
                    rows.append(r)
            return _Result([])
        if self.op == "update":
            for r in rows:
                if all(r.get(c) == v for c, v in self.filters.items()):
                    r.update(self.payload)
            return _Result([])

        out = [r for r in rows
               if all(r.get(c) == v for c, v in self.filters.items())
               and all(r.get(c) != v for c, v in self.excl.items())]
        if self.order_by:
            out = sorted(out, key=lambda r: (r.get(self.order_by) is None,
                                             r.get(self.order_by)), reverse=self.desc)
        if self.rng:
            a, b = self.rng; out = out[a:b + 1]
        if self.lim:
            out = out[:self.lim]
        return _Result([dict(r) for r in out])


class LocalStore:
    def __init__(self):
        self.data, self._ids, self._clock = {}, {}, 0

    def next_id(self, t):
        self._ids[t] = self._ids.get(t, 0) + 1
        return self._ids[t]

    def tick(self):
        self._clock += 1
        return self._clock

    def table(self, name):
        store = self

        class _T:
            def select(_s, cols="*"): return _Query(store, name, "select").select(cols)
            def insert(_s, p): return _Query(store, name, "insert", p)
            def update(_s, p): return _Query(store, name, "update", p)
            def upsert(_s, p, on_conflict=None): return _Query(store, name, "upsert", p, on_conflict)
        return _T()


# ------------------------------------------------------------------ market data

def seed_candles(store, exchange):
    """Bulk-load real hourly candles, then apply the production completeness rule."""
    from src.data.fetcher import drop_incomplete_bars, candles_to_rows

    now_ms = exchange.milliseconds()
    for sym in SYMBOLS:
        rows, end = [], now_ms
        for _ in range(PAGES):
            c = exchange.fetch_ohlcv(sym, "1h", limit=100, params={"after": end})
            if not c: break
            rows = c + rows; end = c[0][0]; time.sleep(0.05)
        kept = drop_incomplete_bars(rows, now_ms)
        store.data.setdefault("candles", []).extend(candles_to_rows(sym, kept))
        print(f"  {sym:10} {len(rows):>5} bars fetched, {len(rows)-len(kept)} still forming, "
              f"{len(kept)} stored")


def main():
    from src.strategy.economics import round_trip_cost_pct, EXECUTION_MODE
    print("=" * 68)
    print(f"LOCAL DRY RUN -- real code, real prices, in-memory database")
    print(f"venue {VENUE} | execution {EXECUTION_MODE} | round trip "
          f"{round_trip_cost_pct()*100:.3f}%")
    print("=" * 68)

    exchange = getattr(ccxt, VENUE)({"enableRateLimit": True})
    store = LocalStore()

    print("\n### seeding candles")
    seed_candles(store, exchange)

    # Point every module at the in-memory store and the reachable venue.
    from src.data import supabase_client
    supabase_client.get_client = lambda: store
    from src.features import engineer
    from src.models import trainer
    from src.paper_trading import engine
    for mod in (engineer, trainer, engine):
        mod.get_client = lambda: store
        if hasattr(mod, "SYMBOLS"): mod.SYMBOLS = SYMBOLS
    engine.create_exchange = lambda: exchange

    if AT_MINUTE is not None:
        import datetime as _dt
        m = float(AT_MINUTE)

        class _Clock:
            @staticmethod
            def now(tz=None):
                real = _dt.datetime.now(tz or _dt.timezone.utc)
                return real.replace(minute=int(m), second=int(m % 1 * 60), microsecond=0)

        engine.datetime = _Clock
        print(f"\n  [clock pinned to :{int(m):02d} past the hour]")

    print("\n### features")
    engineer.main()
    print(f"  feature rows: {len(store.data.get('features', []))}")

    print("\n### training")
    trainer.main()
    print(f"  predictions written: {len(store.data.get('predictions', []))}")

    print("\n### paper trading")
    engine.main()

    trades = store.data.get("paper_trades", [])
    print("\n" + "=" * 68)
    print(f"RESULT: {len(trades)} paper trade(s)")
    for t in trades:
        print(f"  #{t['id']} {t['side']:5} {t['symbol']:10} @ {t['entry_price']:.4f} "
              f"size ${t['size']:.2f} status={t['status']}")
    if store.data.get("portfolio"):
        p = store.data["portfolio"][-1]
        print(f"  equity ${p['equity']:.2f} | cash ${p['cash']:.2f} "
              f"| positions ${p['positions_value']:.2f}")
    print("=" * 68)
    return trades


if __name__ == "__main__":
    main()
