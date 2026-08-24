"""Trade economics: the reason the strategy loses regardless of model quality.

EV per trade = (2p - 1) * E|move| - cost. With a 0.30% round trip and BTC's
0.220% mean hourly move, break-even accuracy is 118% -- unreachable.
"""

import pytest

from src.strategy import economics as ec

# Measured over 1000 live binanceus hourly bars, 2026-08-24.
MEASURED_MOVES = {
    "BTC/USDT": 0.002203,
    "ETH/USDT": 0.003065,
    "ADA/USDT": 0.005722,
    "DOGE/USDT": 0.003636,
    "TRX/USDT": 0.002248,
}
MEASURED_ACCURACY = {
    "BTC/USDT": 0.5155, "ETH/USDT": 0.5236, "ADA/USDT": 0.5156,
    "DOGE/USDT": 0.5208, "TRX/USDT": 0.6502,
}


def test_round_trip_charges_two_legs():
    assert ec.round_trip_cost_pct() == pytest.approx(0.003)


def test_btc_breakeven_accuracy_is_unreachable():
    """The headline finding: no classifier can make 1h BTC profitable here."""
    assert ec.breakeven_accuracy(MEASURED_MOVES["BTC/USDT"]) > 1.0


def test_trx_breakeven_accuracy_is_unreachable():
    assert ec.breakeven_accuracy(MEASURED_MOVES["TRX/USDT"]) > 1.0


def test_every_measured_symbol_has_negative_ev_at_its_real_accuracy():
    for sym, move in MEASURED_MOVES.items():
        ev = ec.expected_value(MEASURED_ACCURACY[sym], move)
        assert ev < 0, f"{sym} should be unprofitable at 1h, got EV {ev}"


def test_no_measured_symbol_is_tradeable():
    for sym, move in MEASURED_MOVES.items():
        assert not ec.is_tradeable(MEASURED_ACCURACY[sym], move), sym


def test_a_genuine_edge_is_tradeable():
    """A 65% model on a 2% move clears the cost comfortably."""
    assert ec.is_tradeable(0.65, 0.02)
    assert ec.expected_value(0.65, 0.02) > 0


def test_coin_flip_has_zero_edge():
    assert ec.expected_edge(0.5, 0.05) == pytest.approx(0.0)
    assert ec.expected_value(0.5, 0.05) == pytest.approx(-0.003)


def test_margin_demands_a_buffer():
    strength, move = 0.60, 0.0160  # edge = 0.32%, just over the 0.30% cost
    assert ec.is_tradeable(strength, move, margin=1.0)
    assert not ec.is_tradeable(strength, move, margin=2.0)


def test_breakeven_on_a_zero_move_is_infinite():
    assert ec.breakeven_accuracy(0.0) == float("inf")


def test_longer_horizon_lowers_the_bar():
    """Moves scale ~sqrt(t) while cost is fixed -- the case for a longer hold."""
    hourly = ec.breakeven_accuracy(0.002203)
    daily = ec.breakeven_accuracy(0.002203 * (24 ** 0.5))
    assert daily < hourly
    assert daily < 1.0
