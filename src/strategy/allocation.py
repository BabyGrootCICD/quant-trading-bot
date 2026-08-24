"""Expected-value capital allocation across simultaneous candidates.

The old path was `percentage_based_size()`: `risk_budget / |estimated_change|`,
capped at $100. That has three problems.

  * It sizes *inversely* to the predicted move, so the least convincing setups
    got the most capital.
  * It ignores every other candidate, so eight symbols firing at once asked for
    eight independent positions with no shared budget.
  * It ignores available cash entirely -- the engine could open more exposure
    than the portfolio holds.

What the strategy actually wants is: given each candidate's expected return and
the volatility of that return, put more money where the edge per unit of risk
is larger, and stop when the budget runs out. That is Kelly, scaled down.

    f_i = KELLY_SCALE * EV_i / sigma_i^2        (fractional Kelly)
    size_i proportional to f_i, scaled to fit the budget, capped per symbol

Fractional Kelly (default 0.25) because EV_i is an estimate from a model with
about a percentage point of measured skill; full Kelly on a mis-estimated edge
is how accounts die. At a one-hour horizon even the fractional number usually
exceeds the whole account, so in practice the budget and the per-symbol cap are
what set position size -- see `allocate()` for which one binds when.
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import env_float
from src.strategy.economics import move_volatility

# Fraction of full Kelly to actually bet. Full Kelly assumes the edge estimate
# is exact; ours is not.
KELLY_SCALE = env_float("KELLY_SCALE", 0.25)

# No single symbol may take more than this fraction of equity, whatever Kelly
# says. Caps the damage from one badly calibrated probability.
MAX_POSITION_FRAC = env_float("MAX_POSITION_FRAC", 0.10)

# Total notional across all open positions, as a fraction of equity. Below 1.0
# there is no leverage and cash is always left over.
MAX_GROSS_EXPOSURE = env_float("MAX_GROSS_EXPOSURE", 0.60)

# Positions smaller than this are not worth the two fills.
MIN_POSITION_USD = env_float("MIN_POSITION_USD", 10)


@dataclass
class Candidate:
    """One tradeable signal, priced in expected-value terms."""
    symbol: str
    side: str            # "long" | "short"
    ev: float            # expected fractional return, net of the round trip
    expected_abs_move: float
    probability_up: float


def kelly_fraction(ev: float, expected_abs_move: float,
                   scale: float = KELLY_SCALE,
                   cap: float = MAX_POSITION_FRAC) -> float:
    """Fraction of equity to commit to a single candidate.

    Returns 0 for a non-positive edge: the EV gate should already have
    filtered those, and betting on one is strictly worse than holding cash.
    """
    if ev <= 0:
        return 0.0
    sigma = move_volatility(expected_abs_move)
    if sigma <= 0:
        return 0.0
    f = ev / (sigma * sigma)
    return max(0.0, min(cap, scale * f))


def allocate(candidates: list[Candidate], equity: float, available_cash: float,
             existing_exposure: float = 0.0,
             scale: float = KELLY_SCALE,
             max_position_frac: float = MAX_POSITION_FRAC,
             max_gross_exposure: float = MAX_GROSS_EXPOSURE,
             min_position_usd: float = MIN_POSITION_USD) -> dict[str, float]:
    """Dollar size per symbol, respecting cash and gross-exposure limits.

    `existing_exposure` is the notional already committed to open positions;
    new allocations share the same gross budget rather than stacking on top of
    it.

    Two constraints bind here, and which one bites changes the answer:

      * **Budget.** Kelly at a one-hour horizon is enormous -- a 0.3% edge
        against a 1% move implies many times the account -- so the raw
        fractions almost always exceed what the portfolio can fund. When they
        do, the whole vector is scaled down by one common factor, which
        preserves the ranking by edge instead of funding whichever symbol
        happened to be iterated first.
      * **Per-symbol cap.** No name may exceed `max_position_frac` of equity
        whatever Kelly says. Capacity freed by a capped name is redistributed
        to the names still below their cap, so the cap does not silently
        shrink the book.

    With few candidates the cap binds and they come out equal-sized; that is
    the correct answer, not a ranking failure -- you cannot express "twice the
    conviction" once both positions are already at the maximum a single symbol
    is allowed to hold.
    """
    if equity <= 0 or not candidates:
        return {}

    weights = {}
    for c in candidates:
        f = kelly_fraction(c.ev, c.expected_abs_move, scale=scale, cap=float("inf"))
        if f > 0:
            weights[c.symbol] = f

    if not weights:
        return {}

    gross_budget = max(0.0, equity * max_gross_exposure - existing_exposure)
    budget = min(gross_budget, max(0.0, available_cash))
    if budget <= 0:
        return {}

    cap = max_position_frac * equity
    total_weight = sum(weights.values())
    target = min(budget, total_weight * equity)

    sizes = {sym: target * (w / total_weight) for sym, w in weights.items()}

    # Redistribute whatever the per-symbol cap takes off the table, until
    # either nothing is left over or every name is capped.
    for _ in range(len(sizes)):
        over = {sym: v for sym, v in sizes.items() if v > cap}
        if not over:
            break
        spare = sum(v - cap for v in over.values())
        for sym in over:
            sizes[sym] = cap
        under = {sym: v for sym, v in sizes.items() if v < cap}
        if not under or spare <= 0:
            break
        under_total = sum(under.values())
        if under_total <= 0:
            break
        for sym, v in under.items():
            sizes[sym] = v + spare * (v / under_total)

    return {sym: round(v, 2) for sym, v in sizes.items() if v >= min_position_usd}
