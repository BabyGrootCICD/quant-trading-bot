import numpy as np


def expected_value(pnl_series: list[float]) -> float:
    if not pnl_series:
        return 0.0
    return float(np.mean(pnl_series))


def sharpe_ratio(returns: list[float], periods_per_year: int = 8760, min_trades: int = 30) -> float:
    if len(returns) < min_trades:
        return 0.0
    arr = np.array(returns)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    return float((mean / std) * np.sqrt(periods_per_year))


def win_rate(pnl_series: list[float]) -> float:
    if not pnl_series:
        return 0.0
    wins = sum(1 for p in pnl_series if p > 0)
    return wins / len(pnl_series)