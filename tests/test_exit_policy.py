"""Exits. There were none, and that is why equity stopped moving.

Portfolio rows 25, 26 and 27 are identical: equity 9976.35, the same two open
positions worth 100.06, the same -11.97 of realized P&L, hours apart. A
position was only ever closed when its signal flipped, so a position whose
signal never flipped was held forever -- on a one-hour forecast.
"""

import pandas as pd
import pytest

from src.paper_trading import engine

NOW = 1_800_000_000_000
HOUR = 3_600_000


def _trade(side="long", entry=100.0, age_hours=1.0, symbol="BTC/USDT"):
    return pd.Series({
        "id": 1,
        "symbol": symbol,
        "side": side,
        "entry_price": entry,
        "size": 100.0,
        "entry_time": NOW - int(age_hours * HOUR),
    })


WANTED_LONG = {"BTC/USDT": "long"}


# --- unrealized return ------------------------------------------------------

def test_long_gains_when_price_rises():
    assert engine.unrealized_return("long", 100.0, 102.0) == pytest.approx(0.02)


def test_short_gains_when_price_falls():
    assert engine.unrealized_return("short", 100.0, 98.0) == pytest.approx(0.02)


def test_zero_entry_price_is_not_a_division_error():
    assert engine.unrealized_return("long", 0.0, 100.0) == 0.0


# --- the exit rules ---------------------------------------------------------

def test_a_quiet_winning_position_is_held():
    assert engine.exit_reason(_trade(), 100.5, NOW, wanted=WANTED_LONG) is None


def test_stop_loss_fires_even_though_the_signal_agrees():
    """The old code held any position the signal still liked, without limit."""
    assert engine.exit_reason(_trade(), 97.0, NOW, wanted=WANTED_LONG) == "stop_loss"


def test_take_profit_fires():
    assert engine.exit_reason(_trade(), 103.0, NOW, wanted=WANTED_LONG) == "take_profit"


def test_short_stop_loss_uses_the_other_direction():
    assert engine.exit_reason(_trade(side="short"), 103.0, NOW,
                              wanted={"BTC/USDT": "short"}) == "stop_loss"


def test_a_position_cannot_outlive_its_one_hour_forecast():
    """This is the row-25-to-27 freeze, expressed as a test."""
    stale = _trade(age_hours=20)
    assert engine.exit_reason(stale, 100.1, NOW, wanted=WANTED_LONG) == "max_holding"


def test_holding_limit_is_configurable_off():
    stale = _trade(age_hours=20)
    assert engine.exit_reason(stale, 100.1, NOW, wanted=WANTED_LONG,
                              max_holding_hours=0) is None


def test_signal_flip_still_closes():
    assert engine.exit_reason(_trade(), 100.1, NOW,
                              wanted={"BTC/USDT": "short"}) == "signal_flip"


def test_no_signal_at_all_closes():
    assert engine.exit_reason(_trade(), 100.1, NOW, wanted=None) == "no_signal"
    assert engine.exit_reason(_trade(), 100.1, NOW, wanted={}) == "no_signal"


def test_risk_exits_outrank_the_signal():
    """A stop must not be overridden by a signal that still agrees."""
    assert engine.exit_reason(_trade(), 90.0, NOW, wanted=WANTED_LONG) == "stop_loss"


def test_missing_entry_time_does_not_force_an_exit():
    """Legacy rows predate the column; they should not all close at once."""
    legacy = pd.Series({"id": 9, "symbol": "BTC/USDT", "side": "long",
                        "entry_price": 100.0, "size": 100.0})
    assert engine.exit_reason(legacy, 100.2, NOW, wanted=WANTED_LONG) is None


def test_position_age_handles_garbage():
    assert engine.position_age_hours(pd.Series({"entry_time": None}), NOW) is None
    assert engine.position_age_hours(pd.Series({"entry_time": "n/a"}), NOW) is None


# --- the reason is recorded -------------------------------------------------

class _Q:
    def __init__(self, rec, name, op, payload):
        self.rec, self.name, self.op, self.payload = rec, name, op, payload

    def eq(self, *_):
        return self

    def execute(self):
        self.rec.append((self.name, self.op, self.payload))
        return type("R", (), {"data": []})()


class _Client:
    def __init__(self):
        self.calls = []

    def table(self, name):
        return type("T", (), {
            "update": lambda _s, p: _Q(self.calls, name, "update", p),
            "insert": lambda _s, p: _Q(self.calls, name, "insert", p),
        })()


def test_exit_reason_is_written_to_the_trade_row():
    client = _Client()
    trades = pd.DataFrame([_trade(age_hours=20).to_dict()])
    engine.close_open_positions(client, trades, {"BTC/USDT": 100.1}, NOW,
                                wanted=WANTED_LONG)
    updates = [c for c in client.calls if c[1] == "update"]
    assert len(updates) == 1
    assert updates[0][2]["exit_reason"] == "max_holding"
    assert updates[0][2]["status"] == "closed"
