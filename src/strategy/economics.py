"""Trade economics: does a signal clear its own transaction cost?

Measured over 1000 live hourly bars per symbol, the answer at a 1h horizon is
almost always no:

    EV per trade = (2p - 1) * E|move| - cost

With a 0.30% round trip (2 legs x (0.10% taker + 5bps slippage)) and BTC's
0.220% mean absolute hourly move, break-even accuracy is 118% -- unreachable
by any classifier. The models measured here carry ~1pp of edge over their
majority-class baseline. Trading every bar is therefore a guaranteed loss, and
the loss is roughly the cost, which is exactly what the portfolio recorded.

These helpers let the engine refuse those trades instead of taking them.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import env_float, env_str

# Fees are per leg. `EXECUTION_MODE=maker` models resting limit orders instead
# of crossing the spread; on most venues that roughly halves the cost term,
# which is the single biggest lever on whether a 1h horizon can ever pay (see
# .claude/STRATEGY_PLAN.md research direction 3). It is a modelling switch, not
# a discount -- a maker order is not guaranteed to fill.
EXECUTION_MODE = env_str("EXECUTION_MODE", "taker").lower()

TAKER_FEE = env_float("TAKER_FEE", 0.001)
MAKER_FEE = env_float("MAKER_FEE", 0.0004)
SLIPPAGE_BPS = env_float("SLIPPAGE_BPS", 5)
MAKER_SLIPPAGE_BPS = env_float("MAKER_SLIPPAGE_BPS", 0)


def per_leg_cost_pct(mode: str | None = None) -> float:
    """Fractional cost of one leg under the configured execution model."""
    mode = (mode or EXECUTION_MODE).lower()
    if mode == "maker":
        return MAKER_FEE + MAKER_SLIPPAGE_BPS / 10000
    return TAKER_FEE + SLIPPAGE_BPS / 10000


def round_trip_cost_pct(taker_fee: float | None = None, slippage_bps: float | None = None,
                        mode: str | None = None) -> float:
    """Fractional cost of opening and closing one position.

    Explicit `taker_fee` / `slippage_bps` still override, so the tests that pin
    the historical 0.30% number keep working regardless of environment.
    """
    if taker_fee is not None or slippage_bps is not None:
        fee = TAKER_FEE if taker_fee is None else taker_fee
        slip = SLIPPAGE_BPS if slippage_bps is None else slippage_bps
        return 2 * (fee + slip / 10000)
    return 2 * per_leg_cost_pct(mode)


def expected_edge(signal_strength: float, expected_abs_move: float) -> float:
    """Expected fractional return before costs.

    `signal_strength` is the model's probability of being right (>= 0.5), so
    (2*strength - 1) is the edge over a coin flip. `expected_abs_move` is the
    typical size of the move being bet on.
    """
    return (2 * signal_strength - 1) * expected_abs_move


def expected_value(signal_strength: float, expected_abs_move: float,
                   cost: float | None = None) -> float:
    """Expected fractional return after costs. Negative means do not trade."""
    cost = round_trip_cost_pct() if cost is None else cost
    return expected_edge(signal_strength, expected_abs_move) - cost


def breakeven_accuracy(expected_abs_move: float, cost: float | None = None) -> float:
    """Accuracy needed for EV to reach zero.

    Returns a value above 1.0 when the move is too small to ever pay for the
    round trip -- the situation for BTC (118%) and TRX (117%) at 1h.
    """
    cost = round_trip_cost_pct() if cost is None else cost
    if expected_abs_move <= 0:
        return float("inf")
    return (cost / expected_abs_move + 1) / 2


def is_tradeable(signal_strength: float, expected_abs_move: float,
                 margin: float = 1.0, cost: float | None = None) -> bool:
    """True when expected edge clears `margin` times the round-trip cost.

    margin=1.0 requires the edge to merely pay for the round trip; higher
    values demand a buffer against the estimate being wrong.
    """
    cost = round_trip_cost_pct() if cost is None else cost
    return expected_edge(signal_strength, expected_abs_move) > margin * cost


def expected_return(prob_up: float, expected_abs_move: float) -> float:
    """Signed expected fractional return over the horizon, before costs.

    `prob_up` is the calibrated probability that the next bar closes up, and
    `expected_abs_move` is the *conditional* typical size of that bar's move.
    A coin flip returns 0 no matter how large the move.
    """
    return (2 * prob_up - 1) * expected_abs_move


def net_expected_return(prob_up: float, expected_abs_move: float,
                        cost: float | None = None) -> float:
    """Signed expected return after paying the round trip in whichever
    direction the trade is taken. Costs never help, so they are subtracted
    from the magnitude, not from the signed value."""
    cost = round_trip_cost_pct() if cost is None else cost
    raw = expected_return(prob_up, expected_abs_move)
    net_mag = abs(raw) - cost
    if net_mag <= 0:
        return 0.0
    return math.copysign(net_mag, raw)


def predicted_price(price: float, prob_up: float, expected_abs_move: float) -> float:
    """Point forecast for the next bar's close.

    `expected_abs_move` is a log-return magnitude, so the forecast compounds
    rather than adding a percentage.
    """
    return float(price) * math.exp(expected_return(prob_up, expected_abs_move))


def move_volatility(expected_abs_move: float) -> float:
    """Standard deviation implied by a mean absolute move.

    For a zero-mean normal, E|X| = sigma * sqrt(2/pi), so sigma = E|X| *
    sqrt(pi/2). Kelly sizing needs a variance, and this is the only variance
    estimate available from the magnitude head.
    """
    return max(float(expected_abs_move), 0.0) * math.sqrt(math.pi / 2)
