import numpy as np

HOURS_PER_YEAR = 8760

# Below this many observations a Sharpe ratio is noise. Callers should report
# it as unknown rather than as 0.0 -- 0.0 is a real value meaning "no
# risk-adjusted return", which is a different claim from "not enough data".
MIN_SHARPE_TRADES = 30


def expected_value(pnl_series: list[float]) -> float:
    if not pnl_series:
        return 0.0
    return float(np.mean(pnl_series))


def sharpe_ratio(returns: list[float], periods_per_year: int = HOURS_PER_YEAR,
                 min_trades: int = MIN_SHARPE_TRADES) -> float:
    """Annualized Sharpe.

    `returns` must be *fractional returns per period*, not dollar PnL --
    feeding dollars in makes the ratio scale-dependent and meaningless. Use
    trade_returns() to convert closed trades.

    `periods_per_year` must match the actual sampling frequency of `returns`.
    Use annualization_factor() when trades do not arrive once per hour.
    """
    if len(returns) < min_trades:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < min_trades:
        return 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))

    # `std == 0` is too strict: a constant series of e.g. -0.0015 leaves a
    # float residue around 1e-19, and mean/std then explodes to ~1e17. That is
    # exactly the shape of this bot's PnL when a fixed fee is the only thing
    # moving, so guard on a relative tolerance instead of exact zero.
    if not np.isfinite(std) or std <= max(1e-12, abs(mean) * 1e-9):
        return 0.0

    return float((mean / std) * np.sqrt(periods_per_year))


def trade_returns(pnl_series: list[float], size_series: list[float]) -> list[float]:
    """Convert per-trade dollar PnL into fractional returns on capital at risk."""
    out = []
    for pnl, size in zip(pnl_series, size_series):
        if size is None or not np.isfinite(size) or size <= 0:
            continue
        if pnl is None or not np.isfinite(pnl):
            continue
        out.append(float(pnl) / float(size))
    return out


def annualization_factor(n_trades: int, span_hours: float) -> int:
    """Periods per year implied by observing `n_trades` over `span_hours`.

    The bot trades several symbols per hour, so the flat sqrt(8760) that used
    to be applied understated the sampling rate by the number of concurrent
    positions.
    """
    if n_trades <= 0 or span_hours <= 0:
        return HOURS_PER_YEAR
    trades_per_hour = n_trades / span_hours
    return max(1, int(round(trades_per_hour * HOURS_PER_YEAR)))


def win_rate(pnl_series: list[float]) -> float:
    if not pnl_series:
        return 0.0
    wins = sum(1 for p in pnl_series if p > 0)
    return wins / len(pnl_series)
