"""One reconciliation pass, decided before any database write.

The old shape was two unconditional phases -- close everything the signal no
longer wanted, re-read `paper_trades`, then open whatever survived -- so no
single place knew what the cycle intended. Three consequences, all pinned here:

  * `desired_sides()` was fed `active_signals` (rows with signal != 0), so a
    symbol the model called *flat* produced no dict key and was indistinguishable
    from a symbol the model never mentioned. Both closed the position, and both
    were recorded as the same `no_signal`.
  * `desired_sides_from_trades()` collapsed duplicate open rows with a dict
    comprehension over an unordered query, so which side survived was not
    deterministic -- and two opposite-side rows could let a third position open.
  * the close `.update()` had no error handling: one failure aborted the run
    before the portfolio row was written.
"""

import pandas as pd
import pytest

from src.paper_trading import engine
from src.paper_trading.engine import Action

NOW = 1_800_000_000_000
HOUR = 3_600_000
PRICES = {"BTC/USDT": 100.0, "ETH/USDT": 50.0}


# --- fakes ------------------------------------------------------------------

class _Q:
    def __init__(self, store, table, op, payload):
        self.store, self.table, self.op, self.payload = store, table, op, payload
        self.filters = {}

    def eq(self, col, val):
        self.filters[col] = val
        return self

    def execute(self):
        self.store.fail_if_configured(self.table, self.op, self.payload, self.filters)
        self.store.writes.append((self.table, self.op, self.payload, dict(self.filters)))
        return type("R", (), {"data": []})()


class FakeClient:
    """Records writes; can be told to raise on a given (table, op)."""

    def __init__(self, fail_on=None, fail_with=None):
        self.writes = []
        # (table, op) fails every such write; (table, op, key) fails only the
        # one whose payload symbol or `.eq()` filter matches -- a close is
        # addressed by id and carries no symbol, so both are checked.
        self.fail_on = fail_on
        self.fail_with = fail_with or RuntimeError("db unavailable")

    def fail_if_configured(self, table, op, payload, filters):
        if not self.fail_on:
            return
        if (table, op) != tuple(self.fail_on[:2]):
            return
        if len(self.fail_on) == 2:
            raise self.fail_with
        key = self.fail_on[2]
        if payload.get("symbol") == key or filters.get("id") == key:
            raise self.fail_with

    def table(self, name):
        store = self

        class _T:
            def update(self, payload):
                return _Q(store, name, "update", payload)

            def insert(self, payload):
                return _Q(store, name, "insert", payload)

        return _T()

    # convenience
    def updates(self):
        return [w for w in self.writes if w[1] == "update"]

    def inserts(self):
        return [w for w in self.writes if w[1] == "insert"]


def _trade(symbol="BTC/USDT", side="long", entry=100.0, size=100.0,
           age_h=1.0, tid=1):
    return {"id": tid, "symbol": symbol, "side": side, "entry_price": entry,
            "size": size, "entry_time": NOW - int(age_h * HOUR), "status": "open"}


def _open(*trades):
    return pd.DataFrame(list(trades))


def _signal(symbol="BTC/USDT", signal=1, prob=0.75, move=0.020):
    strength = prob if signal == 1 else (1 - prob if signal == -1 else 0.0)
    return {"symbol": symbol, "signal": signal, "probability_up": prob,
            "signal_strength": strength, "expected_move_pct": move}


def _signals(*rows):
    return pd.DataFrame(list(rows))


def _run(client, open_trades, signals, prices=None, signals_available=True):
    return engine.reconcile_positions(
        client, open_trades, signals, PRICES if prices is None else prices, NOW,
        model_name="neural_v1", equity_before=10_000.0, signals_available=signals_available,
    )


def _actions(decisions):
    return {d.symbol: d.action for d in decisions}


# ===========================================================================
# The six regressions requested in review
# ===========================================================================

def test_long_plus_long_keeps_the_same_trade_and_charges_no_fee():
    client = FakeClient()
    result = _run(client, _open(_trade(side="long", tid=7)), _signals(_signal(signal=1)))

    assert _actions(result.decisions)["BTC/USDT"] is Action.KEEP
    assert client.writes == [], "a held position must not be rewritten"
    assert result.closed == [] and result.opened == []
    # entry_price / entry_time / size untouched, because nothing was written.
    assert result.kept[0]["id"] == 7
    assert result.kept[0]["entry_price"] == 100.0


def test_short_plus_short_does_the_same():
    client = FakeClient()
    result = _run(client, _open(_trade(side="short", tid=8)),
                  _signals(_signal(signal=-1, prob=0.25)))

    assert _actions(result.decisions)["BTC/USDT"] is Action.KEEP
    assert client.writes == []
    assert result.kept[0]["id"] == 8


def test_long_plus_zero_closes_once_and_does_not_reopen():
    """A flat call is a decision, and it must be distinguishable from silence."""
    client = FakeClient()
    result = _run(client, _open(_trade(side="long")), _signals(_signal(signal=0, prob=0.5)))

    assert _actions(result.decisions)["BTC/USDT"] is Action.CLOSE
    assert len(client.updates()) == 1
    assert client.inserts() == []
    assert client.updates()[0][2]["exit_reason"] == "flat_signal"


def test_long_plus_short_closes_once_and_creates_exactly_one_short():
    client = FakeClient()
    result = _run(client, _open(_trade(side="long")),
                  _signals(_signal(signal=-1, prob=0.10)))

    assert _actions(result.decisions)["BTC/USDT"] is Action.REVERSE
    assert len(client.updates()) == 1
    inserts = client.inserts()
    assert len(inserts) == 1
    assert inserts[0][2]["side"] == "short"
    # Close strictly before open.
    assert client.writes[0][1] == "update" and client.writes[1][1] == "insert"


def test_missing_price_leaves_the_original_row_open():
    client = FakeClient()
    result = _run(client, _open(_trade(side="long")),
                  _signals(_signal(signal=-1, prob=0.10)), prices={"ETH/USDT": 50.0})

    assert _actions(result.decisions)["BTC/USDT"] is Action.SKIP
    assert client.writes == [], "no honest mark means no transition in either direction"
    assert result.kept[0]["id"] == 1


def test_running_the_same_cycle_twice_adds_no_trade_and_no_fee():
    """The steady state has to be free. Paying a round trip per hour on an
    unchanged signal is exactly what bled the account originally."""
    signals = _signals(_signal(signal=1))
    open_trades = _open(_trade(side="long"))

    first = FakeClient()
    _run(first, open_trades, signals)
    second = FakeClient()
    _run(second, open_trades, signals)

    assert first.writes == [] and second.writes == []


# ===========================================================================
# Cases the review did not cover
# ===========================================================================

def test_missing_signal_row_preserves_the_open_trade():
    """Silence is not a flat call. Flattening on a transient trainer failure
    pays a round trip on every open position for no change in conviction."""
    client = FakeClient()
    result = _run(client, _open(_trade(side="long")), _signals(_signal(symbol="ETH/USDT")))

    assert _actions(result.decisions)["BTC/USDT"] is Action.SKIP
    assert [d.reason for d in result.decisions if d.symbol == "BTC/USDT"] == ["no_decision"]
    btc_writes = [w for w in client.writes
                  if w[2].get("symbol") == "BTC/USDT" or w[3].get("id") == 1]
    assert btc_writes == []


def test_no_predictions_at_all_also_preserves():
    client = FakeClient()
    result = _run(client, _open(_trade(side="long")), pd.DataFrame(),
                  signals_available=False)

    assert _actions(result.decisions)["BTC/USDT"] is Action.SKIP
    assert client.writes == []


def test_a_preserved_position_is_still_bounded_by_risk():
    """Preserving on silence is only safe because the stop and the holding
    limit still fire -- otherwise a dead trainer orphans the book."""
    client = FakeClient()
    result = _run(client, _open(_trade(side="long", entry=105.0)), pd.DataFrame(),
                  signals_available=False)

    assert _actions(result.decisions)["BTC/USDT"] is Action.CLOSE
    assert client.updates()[0][2]["exit_reason"] == "stop_loss"


def test_stale_silence_still_hits_the_holding_limit():
    client = FakeClient()
    result = _run(client, _open(_trade(side="long", age_h=24)), pd.DataFrame(),
                  signals_available=False)
    assert client.updates()[0][2]["exit_reason"] == "max_holding"


def test_risk_exit_outranks_keep():
    """The model still liking a position is not a reason to sit through a stop."""
    client = FakeClient()
    result = _run(client, _open(_trade(side="long", entry=105.0)),
                  _signals(_signal(signal=1)))

    assert _actions(result.decisions)["BTC/USDT"] is Action.CLOSE
    assert client.updates()[0][2]["exit_reason"] == "stop_loss"
    assert client.inserts() == []


# --- duplicates -------------------------------------------------------------

def test_duplicate_open_rows_fail_closed():
    """Two opposite-side rows used to collapse to whichever the unordered query
    returned last, and the survivor could pass the `held` guard so a THIRD
    position opened."""
    client = FakeClient()
    result = _run(client,
                  _open(_trade(side="long", tid=1), _trade(side="short", tid=2)),
                  _signals(_signal(signal=1)))

    assert result.duplicates == ["BTC/USDT"]
    assert client.writes == [], "no close, no open, no guess"
    assert result.had_errors()


def test_duplicates_do_not_block_other_symbols():
    client = FakeClient()
    result = _run(client,
                  _open(_trade(side="long", tid=1), _trade(side="short", tid=2)),
                  _signals(_signal(signal=1), _signal(symbol="ETH/USDT", signal=1)))

    assert result.duplicates == ["BTC/USDT"]
    assert [i[2]["symbol"] for i in client.inserts()] == ["ETH/USDT"]


def test_open_by_symbol_separates_singles_from_duplicates():
    singles, dupes = engine.open_by_symbol(
        _open(_trade(symbol="BTC/USDT", tid=1), _trade(symbol="BTC/USDT", tid=2),
              _trade(symbol="ETH/USDT", tid=3)))
    assert sorted(singles) == ["ETH/USDT"]
    assert sorted(dupes) == ["BTC/USDT"]
    assert [r["id"] for r in dupes["BTC/USDT"]] == [1, 2]


def test_duplicate_capital_is_still_counted():
    """Refusing to trade a position is not a reason to drop it from equity --
    that would understate the book by its whole notional."""
    client = FakeClient()
    result = _run(client,
                  _open(_trade(side="long", tid=1), _trade(side="short", tid=2)),
                  _signals(_signal(signal=1)))

    assert result.exposure() == pytest.approx(200.0)
    assert client.writes == []


# --- write failures ---------------------------------------------------------

def test_reverse_does_not_open_when_the_close_fails():
    """Otherwise the book ends up both still long and freshly short."""
    client = FakeClient(fail_on=("paper_trades", "update"))
    result = _run(client, _open(_trade(side="long")),
                  _signals(_signal(signal=-1, prob=0.10)))

    assert client.inserts() == []
    assert result.close_failures == ["BTC/USDT"]
    # Still open in the database, so it must still count toward exposure.
    assert result.exposure() == pytest.approx(100.0)


def test_one_failing_close_does_not_stop_the_others():
    """The bare `.update()` used to abort the whole run on the first failure."""
    client = FakeClient(fail_on=("paper_trades", "update", 1))   # trade id 1 = BTC
    result = _run(client,
                  _open(_trade(symbol="BTC/USDT", side="long", tid=1),
                        _trade(symbol="ETH/USDT", side="long", entry=50.0, tid=2)),
                  _signals(_signal(symbol="BTC/USDT", signal=0, prob=0.5),
                           _signal(symbol="ETH/USDT", signal=0, prob=0.5)))

    assert result.close_failures == ["BTC/USDT"]
    assert [c.symbol for c in result.closed] == ["ETH/USDT"]


def test_a_lost_insert_race_is_quiet_but_a_real_error_is_reported():
    dup = Exception("duplicate key value violates unique constraint "
                    "\"paper_trades_one_open_per_symbol\"")
    assert engine.is_duplicate_open_error(dup)

    for real in (Exception("JWT expired"), Exception("connection reset by peer"),
                 Exception("PGRST204 column not found")):
        assert not engine.is_duplicate_open_error(real)


def test_a_real_insert_failure_lands_in_open_failures():
    client = FakeClient(fail_on=("paper_trades", "insert"),
                        fail_with=Exception("JWT expired"))
    result = _run(client, pd.DataFrame(), _signals(_signal(signal=1)))

    assert result.open_failures == ["BTC/USDT"]
    assert result.had_errors()


def test_a_duplicate_insert_is_not_an_error():
    client = FakeClient(
        fail_on=("paper_trades", "insert"),
        fail_with=Exception('duplicate key value violates unique constraint '
                            '"paper_trades_one_open_per_symbol"'))
    result = _run(client, pd.DataFrame(), _signals(_signal(signal=1)))

    assert result.open_failures == []
    assert not result.had_errors()


# --- classification still defers to the EV gate ------------------------------

def test_classified_open_still_has_to_clear_the_ev_gate():
    """Classification decides direction; `is_tradeable` decides whether it pays."""
    client = FakeClient()
    result = _run(client, pd.DataFrame(),
                  _signals(_signal(signal=1, prob=0.60, move=0.0022)))

    assert _actions(result.decisions)["BTC/USDT"] is Action.OPEN
    assert client.inserts() == [], "a 0.22% move cannot pay a 0.30% round trip"
    assert result.opened == []


def test_a_tradeable_bar_does_open():
    client = FakeClient()
    result = _run(client, pd.DataFrame(), _signals(_signal(signal=1, prob=0.75, move=0.020)))
    assert len(client.inserts()) == 1
    assert result.opened[0]["symbol"] == "BTC/USDT"


# --- exit hysteresis --------------------------------------------------------

def test_a_marginal_flip_does_not_churn_the_book():
    """With entry at sign(p - 0.5), a probability oscillating either side of
    0.50 would otherwise reverse every position every hour."""
    client = FakeClient()
    result = _run(client, _open(_trade(side="long")),
                  _signals(_signal(signal=-1, prob=0.495)))

    assert _actions(result.decisions)["BTC/USDT"] is Action.KEEP
    assert client.writes == []


def test_a_decisive_flip_still_reverses():
    client = FakeClient()
    result = _run(client, _open(_trade(side="long")),
                  _signals(_signal(signal=-1, prob=0.40)))
    assert _actions(result.decisions)["BTC/USDT"] is Action.REVERSE


def test_the_exit_band_is_symmetric():
    assert engine.signal_wants_exit("long", 0.47, band=0.02)
    assert not engine.signal_wants_exit("long", 0.49, band=0.02)
    assert engine.signal_wants_exit("short", 0.53, band=0.02)
    assert not engine.signal_wants_exit("short", 0.51, band=0.02)


# --- the flat/silent distinction, at the map level --------------------------

def test_flat_and_absent_are_different_in_the_desired_map():
    """This is the distinction the whole refactor exists for."""
    desired = engine.desired_side_by_symbol(
        _signals(_signal(symbol="BTC/USDT", signal=0, prob=0.5),
                 _signal(symbol="ETH/USDT", signal=1)))

    assert "BTC/USDT" in desired and desired["BTC/USDT"] is None  # said flat
    assert "SOL/USDT" not in desired                              # said nothing


def test_every_written_exit_reason_is_in_the_vocabulary():
    """A label longer than VARCHAR(24) truncates silently on the way in."""
    for reason in engine.EXIT_REASONS:
        assert len(reason) <= 24
    with pytest.raises(ValueError):
        engine.close_trade(FakeClient(), _trade(), 100.0, NOW, "not_a_real_reason")
