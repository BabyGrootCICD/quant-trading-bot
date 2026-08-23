import numpy as np
import pandas as pd


DEFAULT_LONG_THRESHOLD = 0.55
DEFAULT_SHORT_THRESHOLD = 0.45
DEFAULT_MAX_CHANGE_PCT = 0.02


def generate_signals(
    predictions: pd.DataFrame,
    long_threshold: float = DEFAULT_LONG_THRESHOLD,
    short_threshold: float = DEFAULT_SHORT_THRESHOLD,
) -> pd.DataFrame:
    signals = predictions.copy()
    signals["signal"] = 0

    long_mask = signals["probability_up"] > long_threshold
    short_mask = signals["probability_up"] < short_threshold

    signals.loc[long_mask, "signal"] = 1
    signals.loc[short_mask, "signal"] = -1

    signals["estimated_change_pct"] = (signals["probability_up"] - 0.5) * 2 * DEFAULT_MAX_CHANGE_PCT

    signals["signal_strength"] = np.where(
        signals["signal"] == 1,
        signals["probability_up"],
        np.where(signals["signal"] == -1, 1 - signals["probability_up"], 0.0),
    )

    return signals


def filter_signals(signals: pd.DataFrame, min_strength: float = 0.0) -> pd.DataFrame:
    mask = (signals["signal"] != 0) & (signals["signal_strength"] >= min_strength)
    return signals[mask].copy()
