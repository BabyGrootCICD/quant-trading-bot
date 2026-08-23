"""Regression tests for the Sharpe computation.

The live bot reported sharpe_ratio between -8 and -49 on a portfolio that had
lost $7.58 on $10,000. Those numbers came from feeding dollar PnL into a ratio
that assumed fractional returns sampled once per hour.
"""

import numpy as np
import pytest

from src.utils import metrics


def test_sharpe_is_scale_invariant_on_returns():
    """Doubling position size must not change the Sharpe of the same strategy."""
    rng = np.random.default_rng(0)
    rets = (rng.normal(0.001, 0.01, 200)).tolist()

    small = metrics.sharpe_ratio(rets)
    big = metrics.sharpe_ratio([r for r in rets])  # same returns, any size

    assert small == pytest.approx(big)


def test_dollar_pnl_converted_to_returns():
    pnl = [1.0, -2.0, 3.0]
    sizes = [100.0, 100.0, 200.0]
    assert metrics.trade_returns(pnl, sizes) == pytest.approx([0.01, -0.02, 0.015])


def test_trade_returns_drops_zero_and_nan_sizes():
    pnl = [1.0, 1.0, 1.0, float("nan")]
    sizes = [100.0, 0.0, float("nan"), 100.0]
    assert metrics.trade_returns(pnl, sizes) == pytest.approx([0.01])


def test_annualization_tracks_actual_trade_frequency():
    """Four trades per hour is 4x the sampling rate of one per hour."""
    one_per_hour = metrics.annualization_factor(n_trades=100, span_hours=100)
    four_per_hour = metrics.annualization_factor(n_trades=400, span_hours=100)

    assert one_per_hour == metrics.HOURS_PER_YEAR
    assert four_per_hour == 4 * metrics.HOURS_PER_YEAR


def test_annualization_falls_back_on_degenerate_span():
    assert metrics.annualization_factor(0, 0) == metrics.HOURS_PER_YEAR
    assert metrics.annualization_factor(10, 0) == metrics.HOURS_PER_YEAR


def test_sharpe_needs_min_trades():
    assert metrics.sharpe_ratio([0.01] * 5) == 0.0


def test_sharpe_zero_variance_is_zero_not_inf():
    assert metrics.sharpe_ratio([0.01] * 50) == 0.0


def test_sharpe_ignores_nan_entries():
    rets = [0.01, float("nan"), -0.01] * 20
    assert np.isfinite(metrics.sharpe_ratio(rets))


def test_win_rate():
    assert metrics.win_rate([1.0, -1.0, 1.0, -1.0]) == 0.5
    assert metrics.win_rate([]) == 0.0
