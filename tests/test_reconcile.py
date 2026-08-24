"""Regression tests for position reconciliation (issue #3).

These cover the pure classification in `plan_reconciliation`, which decides
KEEP / CLOSE / OPEN / REVERSE / ERROR per symbol without touching the DB.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.paper_trading.reconcile import (  # noqa: E402
    plan_reconciliation,
    KEEP,
    CLOSE,
    OPEN,
    REVERSE,
    ERROR,
)


def _signals(rows):
    """rows: list of (symbol, signal). Fills the columns the planner/openers use."""
    return pd.DataFrame([
        {
            "symbol": sym,
            "signal": sig,
            "signal_strength": 0.8,
            "probability_up": 0.8 if sig == 1 else (0.2 if sig == -1 else 0.5),
            "estimated_change_pct": 0.01,
            "timestamp": 1,
        }
        for sym, sig in rows
    ])


def _open(rows):
    """rows: list of (symbol, side)."""
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([
        {"id": i + 1, "symbol": sym, "side": side, "entry_price": 100.0, "size": 50.0, "entry_time": 0}
        for i, (sym, side) in enumerate(rows)
    ])


def _by_symbol(plan):
    return {item["symbol"]: item for item in plan}


# 1. long + long keeps the same trade (no fee, unchanged).
def test_long_plus_long_keeps():
    plan = plan_reconciliation(_signals([("BTC/USDT", 1)]), _open([("BTC/USDT", "long")]), {"BTC/USDT": 100.0})
    item = _by_symbol(plan)["BTC/USDT"]
    assert item["action"] == KEEP
    assert not any(i["action"] in (CLOSE, OPEN, REVERSE) for i in plan)


# 2. short + short does the same.
def test_short_plus_short_keeps():
    plan = plan_reconciliation(_signals([("ETH/USDT", -1)]), _open([("ETH/USDT", "short")]), {"ETH/USDT": 100.0})
    assert _by_symbol(plan)["ETH/USDT"]["action"] == KEEP


# 3. long + zero closes once and does not reopen.
def test_long_plus_zero_closes_only():
    plan = plan_reconciliation(_signals([("BTC/USDT", 0)]), _open([("BTC/USDT", "long")]), {"BTC/USDT": 100.0})
    item = _by_symbol(plan)["BTC/USDT"]
    assert item["action"] == CLOSE
    assert not any(i["action"] in (OPEN, REVERSE) for i in plan)


# 4. long + short closes once and creates exactly one short.
def test_long_plus_short_reverses_once():
    plan = plan_reconciliation(_signals([("BTC/USDT", -1)]), _open([("BTC/USDT", "long")]), {"BTC/USDT": 100.0})
    reverses = [i for i in plan if i["action"] == REVERSE]
    assert len(reverses) == 1
    assert reverses[0]["side"] == "short"


# 5. missing price leaves the original row open (no transition).
def test_missing_price_keeps_open():
    # want to reverse long -> short, but no price for the symbol.
    plan = plan_reconciliation(_signals([("BTC/USDT", -1)]), _open([("BTC/USDT", "long")]), {})
    item = _by_symbol(plan)["BTC/USDT"]
    assert item["action"] == KEEP
    assert item["reason"] == "no price"


# 6. running the same cycle twice produces no additional trade (idempotent).
def test_idempotent_when_open_matches_desired():
    # After a long has been opened, the same long signal must not open again.
    plan = plan_reconciliation(_signals([("BTC/USDT", 1)]), _open([("BTC/USDT", "long")]), {"BTC/USDT": 100.0})
    assert all(i["action"] == KEEP for i in plan)


# --- extra coverage for the contract ---

# Fresh open when nothing is held.
def test_open_when_flat():
    plan = plan_reconciliation(_signals([("BTC/USDT", 1)]), _open([]), {"BTC/USDT": 100.0})
    item = _by_symbol(plan)["BTC/USDT"]
    assert item["action"] == OPEN and item["side"] == "long"


# No signal row for a held symbol -> KEEP ("no decision"), never closed.
def test_missing_signal_preserves_position():
    plan = plan_reconciliation(_signals([]), _open([("BTC/USDT", "long")]), {"BTC/USDT": 100.0})
    item = _by_symbol(plan)["BTC/USDT"]
    assert item["action"] == KEEP
    assert item["reason"] == "no signal"


# More than one open row for a symbol -> ERROR (fail closed, do not guess).
def test_duplicate_open_rows_error():
    plan = plan_reconciliation(
        _signals([("BTC/USDT", 1)]),
        _open([("BTC/USDT", "long"), ("BTC/USDT", "short")]),
        {"BTC/USDT": 100.0},
    )
    assert _by_symbol(plan)["BTC/USDT"]["action"] == ERROR
