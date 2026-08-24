import numpy as np
import pandas as pd


# Direction only. The thresholds used to be 0.55/0.45, chosen when the models
# emitted uncalibrated, overconfident probabilities. Calibration maps scores
# onto observed frequencies, and since the models carry about 1.5pp of real
# edge, `probability_up` now sits in a narrow band around 0.50 -- so 0.55/0.45
# fired on 0 to 1 of 8 symbols per hour and the EV gate downstream never got to
# see a candidate.
#
# A fixed probability threshold is also the wrong instrument: it judges `p`
# alone and cannot see the forecast move size, so it blocks p=0.53 on a 2% bar
# while passing p=0.60 on a 0.3% one. `is_tradeable()` weighs both. Let it
# decide, and keep churn in check with SIGNAL_EXIT_BAND on the way out rather
# than with a wider band on the way in.
DEFAULT_LONG_THRESHOLD = 0.50
DEFAULT_SHORT_THRESHOLD = 0.50


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

    # `estimated_change_pct = (p - 0.5) * 2 * 0.02` used to be produced here: a
    # fabricated linear map with a hardcoded 2% ceiling, unrelated to any
    # symbol's volatility, which the old sizer then *divided* by. The expected
    # move now comes from the magnitude head on the prediction row (see
    # src/models/magnitude.py); leaving the fake column in place invited
    # something downstream to pick it up again.

    signals["signal_strength"] = np.where(
        signals["signal"] == 1,
        signals["probability_up"],
        np.where(signals["signal"] == -1, 1 - signals["probability_up"], 0.0),
    )

    return signals


def filter_signals(signals: pd.DataFrame, min_strength: float = 0.0) -> pd.DataFrame:
    mask = (signals["signal"] != 0) & (signals["signal_strength"] >= min_strength)
    return signals[mask].copy()
