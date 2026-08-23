import numpy as np
import pandas as pd


LONG_THRESHOLD = 0.60
SHORT_THRESHOLD = 0.40


def generate_signals(predictions: pd.DataFrame) -> pd.DataFrame:
    signals = predictions.copy()
    signals["signal"] = 0

    long_mask = signals["probability_up"] >= LONG_THRESHOLD
    short_mask = signals["probability_up"] <= SHORT_THRESHOLD

    signals.loc[long_mask, "signal"] = 1
    signals.loc[short_mask, "signal"] = -1

    signals["signal_strength"] = np.where(
        signals["signal"] == 1,
        signals["probability_up"],
        np.where(signals["signal"] == -1, 1 - signals["probability_up"], 0.0),
    )

    return signals


def filter_signals(signals: pd.DataFrame, min_strength: float = 0.0) -> pd.DataFrame:
    mask = (signals["signal"] != 0) & (signals["signal_strength"] >= min_strength)
    return signals[mask].copy()
