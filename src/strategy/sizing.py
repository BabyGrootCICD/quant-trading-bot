import numpy as np


DEFAULT_MAX_POSITION_USD = 100.0


def tanh_size(confidence: float, max_position: float = DEFAULT_MAX_POSITION_USD) -> float:
    return max_position * np.tanh(confidence)


def hard_tanh_size(confidence: float, max_position: float = DEFAULT_MAX_POSITION_USD) -> float:
    clipped = max(-1.0, min(1.0, confidence))
    return max_position * clipped


def constant_size(max_position: float = DEFAULT_MAX_POSITION_USD) -> float:
    return max_position


def calculate_position_size(
    signal_strength: float,
    method: str = "tanh",
    max_position: float = DEFAULT_MAX_POSITION_USD,
) -> float:
    if method == "tanh":
        return tanh_size(signal_strength, max_position)
    elif method == "hard_tanh":
        return hard_tanh_size(signal_strength, max_position)
    elif method == "constant":
        return constant_size(max_position)
    else:
        raise ValueError(f"Unknown sizing method: {method}")
