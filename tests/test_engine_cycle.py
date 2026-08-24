"""One full engine cycle, end to end, against fakes.

The unit tests pin each rule in isolation; nothing exercised `main()`. The
original outage was exactly that kind of gap -- every piece worked, the wiring
between them did not, and the step still exited 0. This runs the real
`main()` and asserts on what reaches the database.
"""

import pandas as pd
import pytest

from src.paper_trading import engine

NOW_MS = 1_800_000_000_000
HOUR = 3_600_000

PRICES = {"BTC/USDT": 100.0, "ETH/USDT": 50.0}


class _FrozenClock:
    """`datetime` stand-in so the cycle runs at a known instant.

    datetime.datetime is immutable, so its `now` cannot be monkeypatched
    directly; the engine looks the name up on the module, which can be.
    """

    @staticmethod
    def now(tz=None):
        import datetime as _dt

        return _dt.datetime.fromtimestamp(NOW_MS / 1000, tz=tz)


class FakeExchange:
    """Live prices plus candles for the unconditional volatility fallback."""

    def fetch_ticker(self, symbol):
        return {"last": PRICES[symbol]}

    def fetch_ohlcv(self, symbol, timeframe, limit=None, since=None):
        base = PRICES[symbol]
        # Alternating 0.2% bars -> unconditional E|move| of about 0.2%.
        return [[NOW_MS - (limit - i) * HOUR, base, base, base,
                 base * (1.002 if i % 2 else 0.998), 1.0] for i in range(limit)]


class FakeQuery:
    def __init__(self, store, table, op, payload=None):
        self.store, self.table, self.op, self.payload = store, table, op, payload
        self.filters = {}

    def select(self, *_):
        return self

    def eq(self, col, val):
        self.filters[col] = val
        return self

    def neq(self, col, val):
        self.filters["!" + col] = val
        return self

    def order(self, *_, **__):
        return self

    def limit(self, *_):
        return self

    def range(self, *_):
        return self

    def upsert(self, payload, on_conflict=None):
        self.op, self.payload = "upsert", payload
        return self

    def execute(self):
        if self.op in ("insert", "update", "upsert"):
            if self.store.fail_on == (self.table, self.op):
                raise RuntimeError("db unavailable")
            self.store.writes.append((self.table, self.op, self.payload, dict(self.filters)))
            self.store.apply(self.table, self.op, self.payload, self.filters)
            return type("Resp", (), {"data": []})()
        self.store.selects.append(self.table)
        return type("Resp", (), {"data": self.store.read(self.table, self.filters)})()


class FakeStore:
    def __init__(self, predictions, open_trades, history):
        self.data = {
            "predictions": predictions,
            "paper_trades": list(open_trades) + list(history),
            "portfolio": [],
        }
        self.writes = []
        self.selects = []
        self.fail_on = None
        self._next_id = 100

    def table(self, name):
        store = self

        class _T:
            def select(self, *_):
                return FakeQuery(store, name, "select")

            def insert(self, payload):
                return FakeQuery(store, name, "insert", payload)

            def update(self, payload):
                return FakeQuery(store, name, "update", payload)

            def upsert(self, payload, on_conflict=None):
                return FakeQuery(store, name, "upsert", payload)

        return _T()

    def read(self, table, filters):
        rows = self.data.get(table, [])
        for col, val in filters.items():
            if col.startswith("!"):
                rows = [r for r in rows if r.get(col[1:]) != val]
            else:
                rows = [r for r in rows if r.get(col) == val]
        return [dict(r) for r in rows]

    def apply(self, table, op, payload, filters):
        if table == "paper_trades" and op == "insert":
            row = dict(payload)
            row["id"] = self._next_id
            self._next_id += 1
            self.data["paper_trades"].append(row)
        elif table == "paper_trades" and op == "update":
            for row in self.data["paper_trades"]:
                if row.get("id") == filters.get("id"):
                    row.update(payload)
        elif table == "portfolio":
            self.data["portfolio"].append(dict(payload))

    # convenience
    def portfolio_row(self):
        return self.data["portfolio"][-1]

    def open_trades(self):
        return [r for r in self.data["paper_trades"] if r.get("status") == "open"]


def _prediction(symbol, prob_up, move, age_h=0.5):
    return {"symbol": symbol, "timestamp": NOW_MS - int(age_h * HOUR),
            "probability_up": prob_up, "confidence": abs(prob_up - 0.5) * 2,
            "model_name": "neural_v1", "expected_move_pct": move,
            "expected_return_pct": (2 * prob_up - 1) * move,
            "predicted_price": None}


@pytest.fixture
def patched(monkeypatch):
    def _run(predictions, open_trades=(), history=(), fail_on=None):
        store = FakeStore(predictions, open_trades, history)
        store.fail_on = fail_on
        monkeypatch.setattr(engine, "get_client", lambda: store)
        monkeypatch.setattr(engine, "create_exchange", lambda: FakeExchange())
        monkeypatch.setattr(engine, "SYMBOLS", ["BTC/USDT", "ETH/USDT"])
        monkeypatch.setattr(engine, "datetime", _FrozenClock)
        engine.main()
        return store

    return _run


# --- the cycle actually trades ---------------------------------------------

def test_a_tradeable_forecast_opens_a_sized_position(patched):
    """The whole point: a bar whose conditional move clears the round trip
    produces a position, where the unconditional gate produced nothing."""
    store = patched([_prediction("BTC/USDT", 0.75, 0.020),
                     _prediction("ETH/USDT", 0.50, 0.002)])

    opened = store.open_trades()
    assert [t["symbol"] for t in opened] == ["BTC/USDT"]
    assert opened[0]["side"] == "long"
    assert opened[0]["size"] > 0
    assert opened[0]["entry_time"] == NOW_MS


def test_position_size_respects_the_per_symbol_cap(patched):
    store = patched([_prediction("BTC/USDT", 0.95, 0.05)])
    size = store.open_trades()[0]["size"]
    # 10% of a $10,000 book, and nowhere near the old flat $100.
    assert size == pytest.approx(1_000.0, abs=1.0)


def test_a_bearish_forecast_goes_short(patched):
    store = patched([_prediction("BTC/USDT", 0.25, 0.020)])
    assert store.open_trades()[0]["side"] == "short"


def test_an_untradeable_bar_opens_nothing(patched):
    """A real BTC hourly move against a 0.30% round trip."""
    store = patched([_prediction("BTC/USDT", 0.60, 0.0022)])
    assert store.open_trades() == []


def test_a_stale_prediction_opens_nothing(patched):
    store = patched([_prediction("BTC/USDT", 0.75, 0.020, age_h=5)])
    assert store.open_trades() == []


# --- the cycle also exits ---------------------------------------------------

def _open_trade(symbol="BTC/USDT", side="long", entry=100.0, age_h=1.0, tid=1):
    return {"id": tid, "symbol": symbol, "side": side, "entry_price": entry,
            "size": 100.0, "entry_time": NOW_MS - int(age_h * HOUR), "status": "open"}


def test_a_stale_position_is_released(patched):
    """Rows 25-27 of the live table: held forever on a one-hour forecast."""
    store = patched([_prediction("BTC/USDT", 0.75, 0.020)],
                    open_trades=[_open_trade(age_h=24)])
    closed = [r for r in store.data["paper_trades"] if r.get("status") == "closed"]
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "max_holding"
    assert closed[0]["exit_time"] == NOW_MS


def test_a_losing_position_is_stopped_out(patched):
    store = patched([_prediction("BTC/USDT", 0.75, 0.020)],
                    open_trades=[_open_trade(entry=105.0)])
    closed = [r for r in store.data["paper_trades"] if r.get("status") == "closed"]
    assert closed[0]["exit_reason"] == "stop_loss"


def test_a_healthy_position_is_held_and_not_re_entered(patched):
    """Re-entering an identical position just pays the spread twice."""
    store = patched([_prediction("BTC/USDT", 0.75, 0.020)],
                    open_trades=[_open_trade(entry=100.2)])
    assert len(store.open_trades()) == 1
    assert store.open_trades()[0]["id"] == 1


# --- the books balance ------------------------------------------------------

def _closed(pnl, fees=0.3, tid=50):
    return {"id": tid, "symbol": "BTC/USDT", "side": "long", "entry_price": 100.0,
            "size": 100.0, "entry_time": NOW_MS - 10 * HOUR, "exit_time": NOW_MS - 9 * HOUR,
            "pnl": pnl, "actual_pnl_usd": pnl, "fees": fees, "status": "closed"}


def test_the_portfolio_row_reconciles(patched):
    store = patched([_prediction("BTC/USDT", 0.60, 0.0022)],
                    history=[_closed(-2.0, tid=50), _closed(3.0, tid=51)])
    row = store.portfolio_row()
    # 10000 + 1.0 of realized PnL, nothing open, fees already inside pnl.
    assert row["equity"] == pytest.approx(10_001.0)
    assert row["cash"] == pytest.approx(10_001.0)
    assert row["positions_value"] == pytest.approx(0.0)
    assert row["total_pnl"] == pytest.approx(1.0)
    assert row["total_trades"] == 2


def test_equity_equals_cash_plus_positions(patched):
    store = patched([_prediction("BTC/USDT", 0.75, 0.020)],
                    history=[_closed(-2.0)])
    row = store.portfolio_row()
    assert row["equity"] == pytest.approx(row["cash"] + row["positions_value"], abs=0.01)


def test_sharpe_is_unknown_not_zero_on_a_thin_record(patched):
    """0.0 is a claim; NULL is the truth when three trades have closed."""
    store = patched([_prediction("BTC/USDT", 0.60, 0.0022)],
                    history=[_closed(-2.0, tid=50), _closed(3.0, tid=51)])
    assert store.portfolio_row()["sharpe_ratio"] is None


def test_no_predictions_at_all_is_not_a_crash(patched):
    store = patched([])
    assert store.data["portfolio"] == []


# --- the cycle must not read back what it just wrote ------------------------

def test_paper_trades_is_read_exactly_twice_per_cycle(patched):
    """One open-positions read and one history read, both before any write.

    The old flow re-read `paper_trades` after closing and again after opening.
    Those re-reads pick up concurrent runs' rows, so the portfolio snapshot
    could describe a book this cycle did not create, and the allocator could
    size against cash a concurrent close had already changed. Everything the
    open phase needs is derived locally from confirmed writes instead.
    """
    store = patched([_prediction("BTC/USDT", 0.75, 0.020)],
                    open_trades=[_open_trade(age_h=24)])

    assert store.selects.count("paper_trades") == 2


def test_the_portfolio_row_is_written_even_when_a_close_fails(patched):
    """A missing equity record is worse than a recorded bad hour.

    The close `.update()` used to be bare, so one failure propagated out of
    main() and steps [5/6] and [6/6] never ran -- a hole in the equity curve
    rather than a wrong value in it.
    """
    with pytest.raises(SystemExit) as exc:
        patched([_prediction("BTC/USDT", 0.75, 0.020)],
                open_trades=[_open_trade(age_h=24)],
                fail_on=("paper_trades", "update"))

    assert exc.value.code == 1, "and the step must go red, not report green"


def test_the_portfolio_row_survives_the_failure(patched, monkeypatch):
    store = FakeStore([_prediction("BTC/USDT", 0.75, 0.020)], [_open_trade(age_h=24)], [])
    store.fail_on = ("paper_trades", "update")
    monkeypatch.setattr(engine, "get_client", lambda: store)
    monkeypatch.setattr(engine, "create_exchange", lambda: FakeExchange())
    monkeypatch.setattr(engine, "SYMBOLS", ["BTC/USDT", "ETH/USDT"])
    monkeypatch.setattr(engine, "datetime", _FrozenClock)

    with pytest.raises(SystemExit):
        engine.main()

    assert len(store.data["portfolio"]) == 1
    # The position stayed open in the database, so it must still be marked.
    assert store.portfolio_row()["positions_value"] > 0


def test_scored_trades_is_persisted(patched):
    """A win_rate of 0.5 over two trades and over two hundred look identical in
    the table without its denominator."""
    store = patched([_prediction("BTC/USDT", 0.60, 0.0022)],
                    history=[_closed(-2.0, tid=50), _closed(3.0, tid=51)])
    row = store.portfolio_row()
    assert row["scored_trades"] == 2
    assert row["win_rate"] == pytest.approx(0.5)


def test_a_position_with_no_live_price_is_marked_at_cost(patched):
    """Its size is already subtracted from cash as locked capital, so dropping
    it from positions_value made equity read the full notional low for as long
    as the price fetch kept failing. Unknown P&L is not vanished capital."""
    store = patched([_prediction("BTC/USDT", 0.60, 0.0022)],
                    open_trades=[_open_trade(symbol="SOL/USDT", entry=10.0, tid=9)])
    row = store.portfolio_row()

    assert row["positions_value"] == pytest.approx(100.0)
    assert row["equity"] == pytest.approx(10_000.0)
    assert row["equity"] == pytest.approx(row["cash"] + row["positions_value"])
