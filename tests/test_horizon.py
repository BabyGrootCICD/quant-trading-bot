"""A forecast for the next bar is only collectable over what is left of it.

The EV gate priced every candidate at its full-bar expected move, but the
pipeline does not enter at the top of the hour. Against a `0 * * * *` cron,
GitHub actually started runs at 14:48, 13:00, 11:34 and 10:41 -- routinely 40
to 80 percent of the way through the bar the prediction was made for. The
round trip is paid in full whatever the entry time; the move is not.

So a setup clearing the gate by +0.100% at the top of the hour was worth
-0.137% entered fifty minutes in, and the gate approved it anyway.
"""

import numpy as np
import pandas as pd
import pytest

from src.paper_trading import engine
from src.strategy.economics import (
    collectable_move, expected_value, horizon_fraction, round_trip_cost_pct,
)

HOUR_MS = 3_600_000
BAR_START = 1_800_000_000_000          # exactly on an hour boundary
PRICES = {"BTC/USDT": 100.0}


def at(minutes_into_bar):
    return BAR_START + int(minutes_into_bar * 60_000)


def _signals(prob=0.60, move=0.020):
    return pd.DataFrame([{
        "symbol": "BTC/USDT", "signal": 1, "probability_up": prob,
        "signal_strength": prob, "expected_move_pct": move,
    }])


# --- the horizon term itself ------------------------------------------------

def test_a_full_bar_remains_at_the_open():
    assert horizon_fraction(at(0)) == pytest.approx(1.0)


def test_the_bar_drains_linearly():
    assert horizon_fraction(at(15)) == pytest.approx(0.75)
    assert horizon_fraction(at(30)) == pytest.approx(0.50)
    assert horizon_fraction(at(45)) == pytest.approx(0.25)


def test_the_fraction_never_leaves_the_unit_interval():
    for m in (0, 1, 30, 59, 59.99):
        assert 0.0 <= horizon_fraction(at(m)) <= 1.0


def test_move_scales_with_the_square_root_of_time_not_linearly():
    """Half a bar left is about 71% of the move, not 50%. Treating it as
    linear would under-price entries rather than over-price them, but it
    would still be wrong."""
    assert collectable_move(0.02, 0.5) == pytest.approx(0.02 * np.sqrt(0.5))
    assert collectable_move(0.02, 0.5) > 0.01


def test_a_full_horizon_leaves_the_forecast_alone():
    assert collectable_move(0.02, 1.0) == pytest.approx(0.02)


def test_collectable_move_clamps_a_nonsense_fraction():
    assert collectable_move(0.02, 1.5) == pytest.approx(0.02)
    assert collectable_move(0.02, -0.5) == pytest.approx(0.0)


# --- the gate acts on it ----------------------------------------------------

def _candidates(now_ms, **kw):
    return engine.build_candidates(_signals(**kw), PRICES, now_ms=now_ms)


def test_the_same_setup_is_taken_at_the_open_and_refused_late():
    """This is the defect, stated as one test."""
    assert len(_candidates(at(0))) == 1
    assert _candidates(at(50)) == []


def test_the_boundary_sits_where_the_arithmetic_says_it_does():
    """EV = (2p-1) * move * sqrt(f) - cost, so a +0.100% edge at the open
    crosses zero once sqrt(f) drops below cost/(edge+cost)."""
    cost = round_trip_cost_pct()
    edge_full = (2 * 0.60 - 1) * 0.020
    f_break = (cost / edge_full) ** 2
    minutes_break = (1 - f_break) * 60

    assert len(_candidates(at(minutes_break - 3))) == 1
    assert _candidates(at(minutes_break + 3)) == []


def test_nothing_opens_once_too_little_of_the_bar_is_left():
    """Below the floor the remaining-move estimate is microstructure, not
    model, so the gate stops pricing and just declines."""
    assert _candidates(at(58), move=0.10) == [], "even a huge forecast"


def test_the_floor_does_not_short_circuit_a_whole_quarter_of_the_hour():
    """It shipped at 0.25, which discarded every cycle landing in the last 15
    minutes before EV was ever consulted -- and GitHub drops runs at
    effectively random points in the bar. `collectable_move()` already prices
    a late entry via sqrt(fraction); the floor is only meant to stop that model
    being extrapolated into the final minutes."""
    from src.strategy.economics import MIN_HORIZON_FRACTION

    assert MIN_HORIZON_FRACTION < 0.25
    # 15 minutes left is priced, not vetoed.
    assert engine.build_candidates(_signals(prob=0.90, move=0.10), PRICES,
                                   now_ms=at(45)) != []


def test_the_floor_is_configurable():
    sig = _signals(move=0.10)
    assert engine.build_candidates(sig, PRICES, now_ms=at(55), min_horizon=0.0)


def test_omitting_the_clock_keeps_the_old_full_bar_behaviour():
    """`now_ms=None` is the no-discount path the older callers and tests use."""
    assert len(engine.build_candidates(_signals(), PRICES)) == 1


# --- sizing must see the same number ---------------------------------------

def test_the_allocator_sizes_on_the_discounted_move():
    """Kelly reads both the edge and the variance off expected_abs_move, so
    handing it the undiscounted figure would size a part-bar bet as a full one."""
    early = _candidates(at(0), prob=0.75, move=0.020)[0]
    late = _candidates(at(30), prob=0.75, move=0.020)[0]

    assert late.expected_abs_move < early.expected_abs_move
    assert late.expected_abs_move == pytest.approx(0.020 * np.sqrt(0.5))
    assert late.ev < early.ev


def test_shortening_the_horizon_raises_raw_kelly_not_lowers_it():
    """Worth pinning because it is counter-intuitive and easy to "fix" wrongly.

    Edge falls as sqrt(f) while variance falls as f, so edge/variance scales as
    1/sqrt(f): a shorter horizon is a *less variable* bet, and Kelly asks for
    more of it, right up until the cost term drives EV to zero. So the horizon
    discount must not be relied on to shrink positions -- the protection is the
    EV gate refusing the trade, not the sizer trimming it.
    """
    from src.strategy.allocation import kelly_fraction

    early = _candidates(at(0), prob=0.75, move=0.020)[0]
    late = _candidates(at(40), prob=0.75, move=0.020)[0]

    f_early = kelly_fraction(early.ev, early.expected_abs_move, cap=float("inf"))
    f_late = kelly_fraction(late.ev, late.expected_abs_move, cap=float("inf"))
    assert f_late > f_early


def test_the_per_symbol_cap_holds_late_entries_in_check():
    """Both are far above the cap, so the book never actually leverages up on
    a late entry -- but only because the cap is there."""
    from src.strategy.allocation import allocate

    early = allocate(_candidates(at(0), prob=0.75, move=0.020),
                     equity=10_000, available_cash=10_000)
    late = allocate(_candidates(at(40), prob=0.75, move=0.020),
                    equity=10_000, available_cash=10_000)
    assert early["BTC/USDT"] == late["BTC/USDT"] == pytest.approx(1_000.0)


# --- end to end -------------------------------------------------------------

def test_a_late_cycle_opens_nothing_and_says_why(capsys):
    client = type("C", (), {"table": lambda self, n: type("T", (), {
        "insert": lambda _s, p: type("Q", (), {"execute": lambda _: None})()})()})()

    opened = engine.open_new_positions(
        client, _signals(prob=0.75, move=0.030), PRICES, at(58),
        "neural_v1", 10_000.0, held={})

    assert opened == []
    out = capsys.readouterr().out
    assert "of the forecast bar" in out
