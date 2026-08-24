"""The risk pass: exits only, never entries.

A stop is about where price is now, so it should be checked far more often
than the model is retrained. Against a `0 * * * *` cron GitHub delivered gaps
of 47 to 136 minutes, which made "a 1.5% stop" mean "a 1.5% stop, consulted
some time in the next one to two hours".

The narrowness is the safety property: this module cannot open a position, so
it can never act on a stale prediction, and it never consults a signal, so it
cannot disagree with the hourly cycle about direction.
"""

import pandas as pd
import pytest

from src.paper_trading import risk_monitor

NOW = 1_800_000_000_000
HOUR = 3_600_000


def _trade(symbol="BTC/USDT", side="long", entry=100.0, size=100.0, age_h=1.0, tid=1):
    return {"id": tid, "symbol": symbol, "side": side, "entry_price": entry,
            "size": size, "entry_time": NOW - int(age_h * HOUR), "status": "open"}


def _open(*trades):
    return pd.DataFrame(list(trades))


def _reasons(exits):
    return {t["symbol"]: r for t, _p, r in exits}


# --- what it closes ---------------------------------------------------------

def test_a_healthy_position_is_left_alone():
    exits, dupes = risk_monitor.evaluate_risk(
        _open(_trade()), {"BTC/USDT": 100.5}, NOW)
    assert exits == [] and dupes == {}


def test_a_stop_fires():
    exits, _ = risk_monitor.evaluate_risk(_open(_trade()), {"BTC/USDT": 97.0}, NOW)
    assert _reasons(exits) == {"BTC/USDT": "stop_loss"}


def test_a_target_fires():
    exits, _ = risk_monitor.evaluate_risk(_open(_trade()), {"BTC/USDT": 103.0}, NOW)
    assert _reasons(exits) == {"BTC/USDT": "take_profit"}


def test_a_short_stops_on_the_other_side():
    exits, _ = risk_monitor.evaluate_risk(
        _open(_trade(side="short")), {"BTC/USDT": 103.0}, NOW)
    assert _reasons(exits) == {"BTC/USDT": "stop_loss"}


def test_the_holding_limit_fires_between_hourly_runs():
    """The case the 136-minute scheduling gap made unreachable."""
    exits, _ = risk_monitor.evaluate_risk(
        _open(_trade(age_h=24)), {"BTC/USDT": 100.1}, NOW)
    assert _reasons(exits) == {"BTC/USDT": "max_holding"}


def test_a_missing_price_closes_nothing():
    exits, _ = risk_monitor.evaluate_risk(_open(_trade()), {}, NOW)
    assert exits == []


def test_it_evaluates_every_symbol_independently():
    exits, _ = risk_monitor.evaluate_risk(
        _open(_trade(symbol="BTC/USDT", tid=1),
              _trade(symbol="ETH/USDT", entry=50.0, tid=2)),
        {"BTC/USDT": 97.0, "ETH/USDT": 50.1}, NOW)
    assert _reasons(exits) == {"BTC/USDT": "stop_loss"}


# --- what it refuses to do --------------------------------------------------

def test_duplicates_are_reported_not_guessed():
    exits, dupes = risk_monitor.evaluate_risk(
        _open(_trade(side="long", tid=1), _trade(side="short", tid=2)),
        {"BTC/USDT": 97.0}, NOW)
    assert exits == []
    assert sorted(dupes) == ["BTC/USDT"]


def test_it_only_ever_uses_risk_rules_never_a_signal():
    """No signal input means it cannot contradict the hourly cycle's direction
    or act on a prediction that has gone stale since."""
    import inspect

    src = inspect.getsource(risk_monitor)
    assert "generate_signals" not in src
    assert "build_candidates" not in src
    assert "open_new_positions" not in src
    assert "fetch_latest_predictions" not in src


def test_it_cannot_insert_a_trade():
    """Structural, not textual: find every `x.insert(...)` call in the module
    and allow only `sys.path.insert`. A grep for '.insert(' matches the import
    shim at the top and passes for the wrong reason."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(risk_monitor))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "insert":
            if ast.unparse(fn.value) != "sys.path":
                offenders.append(ast.unparse(fn))

    assert offenders == [], f"a risk pass must never open a position: {offenders}"


def test_the_risk_rules_are_the_engine_s_own():
    """Shared, not reimplemented -- two copies of a stop rule would drift."""
    from src.paper_trading import engine

    assert risk_monitor.risk_exit_reason is engine.risk_exit_reason
    assert risk_monitor.close_trade is engine.close_trade
    assert risk_monitor.update_portfolio is engine.update_portfolio


# --- accounting -------------------------------------------------------------

def test_the_history_is_read_before_the_closes_are_written():
    """Reading it afterwards means the closes are already in it, and
    `augment_history` then adds them again -- a stop-out booked at twice its
    loss. The order is the whole guard, so pin it structurally."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(risk_monitor))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "main")

    calls = [ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)]
    order = [c for c in calls if c in ("fetch_trade_history", "close_trade")]
    assert order.index("fetch_trade_history") < order.index("close_trade")


def test_the_still_open_frame_is_not_re_read_after_writing():
    """A read-after-write would pick up a concurrent hourly run's inserts, so
    the equity row could describe a book this pass did not create."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(risk_monitor))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    reads = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and ast.unparse(n.func) == "fetch_open_trades"]
    assert len(reads) == 1, "open positions are read once, at the start"
