import numpy as np


DEFAULT_MAX_POSITION_USD = 100.0
DEFAULT_RISK_PER_TRADE = 0.02


def tanh_size(confidence: float, max_position: float = DEFAULT_MAX_POSITION_USD) -> float:
    return max_position * np.tanh(confidence)


def percentage_based_size(
    estimated_change_pct: float,
    total_asset_usd: float,
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
    max_position: float = DEFAULT_MAX_POSITION_USD,
) -> float:
    abs_change = abs(estimated_change_pct)
    if abs_change < 0.001:
        return 0.0
    size = min(total_asset_usd * risk_per_trade / abs_change, max_position)
    return max(0.0, size)


def calculate_position_size(
    signal_strength: float,
    method: str = "tanh",
    max_position: float = DEFAULT_MAX_POSITION_USD,
    estimated_change_pct: float = 0.0,
    total_asset_usd: float = 10000.0,
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
) -> float:
    if method == "tanh":
        return tanh_size(signal_strength, max_position)
    elif method == "percentage":
        return percentage_based_size(
            estimated_change_pct, total_asset_usd, risk_per_trade, max_position
        )
    elif method == "constant":
        return max_position
    else:
        raise ValueError(f"Unknown sizing method: {method}")
