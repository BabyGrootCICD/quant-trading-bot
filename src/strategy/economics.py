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

TAKER_FEE = 0.001
SLIPPAGE_BPS = 5


def round_trip_cost_pct(taker_fee: float = TAKER_FEE, slippage_bps: float = SLIPPAGE_BPS) -> float:
    """Fractional cost of opening and closing one position."""
    per_leg = taker_fee + slippage_bps / 10000
    return 2 * per_leg


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
