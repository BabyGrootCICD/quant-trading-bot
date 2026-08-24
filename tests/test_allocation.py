"""Capital allocation: more money where the edge per unit of risk is larger.

The replaced sizing rule was `risk_budget / |estimated_change|` capped at $100.
It sized inversely to conviction, ignored every other candidate, and ignored
cash. These tests pin the properties the EV allocator has to hold instead.
"""

import pytest

from src.strategy.allocation import Candidate, allocate, kelly_fraction


def _c(symbol, ev, move, side="long", p=0.6):
    return Candidate(symbol=symbol, side=side, ev=ev, expected_abs_move=move,
                     probability_up=p)


# --- Kelly ------------------------------------------------------------------

def test_no_edge_gets_no_capital():
    assert kelly_fraction(0.0, 0.01) == 0.0
    assert kelly_fraction(-0.002, 0.01) == 0.0


def test_more_edge_gets_more_capital():
    small = kelly_fraction(0.0001, 0.02, cap=1.0)
    large = kelly_fraction(0.0004, 0.02, cap=1.0)
    assert large > small


def test_more_risk_gets_less_capital_for_the_same_edge():
    """The old rule did the opposite: it divided by the expected move."""
    calm = kelly_fraction(0.0002, 0.02, cap=1.0)
    wild = kelly_fraction(0.0002, 0.08, cap=1.0)
    assert calm > wild


def test_position_cap_is_respected():
    # A huge edge on a tiny move implies a Kelly fraction well over 1.
    assert kelly_fraction(0.05, 0.005, cap=0.10) == pytest.approx(0.10)


def test_zero_volatility_is_not_infinite_leverage():
    assert kelly_fraction(0.01, 0.0) == 0.0


# --- allocation -------------------------------------------------------------

def test_capital_goes_to_the_better_edge():
    sizes = allocate([_c("A", 0.004, 0.01), _c("B", 0.001, 0.01)],
                     equity=10_000, available_cash=1_500)
    assert sizes["A"] > sizes["B"]


def test_two_saturated_candidates_come_out_equal_at_the_cap():
    """Not a ranking failure: neither may exceed 10% of equity, so both sit there.

    Kelly at a one-hour horizon asks for multiples of the account on almost any
    positive edge, so with a handful of candidates the per-symbol cap, not the
    edge, is what sets size.
    """
    sizes = allocate([_c("A", 0.004, 0.01), _c("B", 0.001, 0.01)],
                     equity=10_000, available_cash=10_000, max_position_frac=0.10)
    assert sizes == {"A": 1_000.0, "B": 1_000.0}


def test_capacity_freed_by_the_cap_is_redistributed():
    sizes = allocate([_c("A", 0.02, 0.01), _c("B", 0.0005, 0.01)],
                     equity=10_000, available_cash=1_200, max_position_frac=0.10)
    assert sizes["A"] == pytest.approx(1_000.0)
    assert sum(sizes.values()) == pytest.approx(1_200.0, abs=1.0)


def test_gross_exposure_is_capped():
    cands = [_c(f"S{i}", 0.01, 0.01) for i in range(8)]
    sizes = allocate(cands, equity=10_000, available_cash=10_000,
                     max_gross_exposure=0.60)
    assert sum(sizes.values()) == pytest.approx(6_000, abs=1.0)


def test_existing_exposure_shares_the_same_budget():
    """Open positions must not be topped up to a second full book."""
    cands = [_c(f"S{i}", 0.01, 0.01) for i in range(8)]
    sizes = allocate(cands, equity=10_000, available_cash=10_000,
                     existing_exposure=5_000, max_gross_exposure=0.60)
    assert sum(sizes.values()) == pytest.approx(1_000, abs=1.0)


def test_allocation_never_exceeds_available_cash():
    cands = [_c(f"S{i}", 0.01, 0.01) for i in range(8)]
    sizes = allocate(cands, equity=10_000, available_cash=250)
    assert sum(sizes.values()) <= 250 + 1e-6


def test_no_cash_means_no_trades():
    assert allocate([_c("A", 0.01, 0.01)], equity=10_000, available_cash=0.0) == {}


def test_scaling_down_preserves_the_ranking():
    """Squeezing the book must not arbitrarily fund whoever iterated first."""
    cands = [_c("A", 0.006, 0.01), _c("B", 0.002, 0.01)]
    roomy = allocate(cands, equity=10_000, available_cash=500)
    tight = allocate(cands, equity=10_000, available_cash=200)
    assert tight["A"] > tight["B"]
    assert roomy["A"] / roomy["B"] == pytest.approx(tight["A"] / tight["B"], rel=1e-6)


def test_dust_positions_are_dropped():
    sizes = allocate([_c("A", 0.01, 0.01)], equity=100, available_cash=5,
                     min_position_usd=10)
    assert sizes == {}


def test_negative_ev_candidates_get_nothing():
    sizes = allocate([_c("A", 0.004, 0.01), _c("B", -0.002, 0.01)],
                     equity=10_000, available_cash=10_000)
    assert "B" not in sizes


def test_empty_input_is_not_a_crash():
    assert allocate([], equity=10_000, available_cash=10_000) == {}
    assert allocate([_c("A", 0.01, 0.01)], equity=0, available_cash=100) == {}
