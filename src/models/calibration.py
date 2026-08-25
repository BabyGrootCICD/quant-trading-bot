"""Probability calibration for the directional head.

The whole allocation stack is built on `EV = (2p - 1) * E|move| - cost`. That
formula is only meaningful if `p` is a *calibrated* probability: if the bars
the model scores 0.60 actually close up about 60% of the time.

Nothing guaranteed that before. `LogisticModel` is fitted with
`class_weight="balanced"`, which deliberately distorts the decision threshold
on an imbalanced label and pushes probabilities away from their empirical
frequencies; XGBoost's raw sigmoid outputs are over-confident for the same
structural reasons. Feeding those numbers into an EV gate produces confident
nonsense in both directions -- refusing real edges and funding imaginary ones.

The fix is to fit a monotone map from raw score to observed frequency on the
walk-forward *out-of-sample* predictions, which are the only honest sample
available, and apply it at predict time.
"""

import numpy as np

# Below this many out-of-sample observations, any calibration is just
# memorising noise; fall through to the identity map instead.
MIN_CALIBRATION_ROWS = 200

# Default method. Sigmoid (Platt) is two parameters fitted over the whole
# score range, so it cannot manufacture confidence out of a handful of points.
#
# Isotonic was the original choice and it is the wrong tool here. It is fitted
# as a step function over pooled-adjacent blocks, and at the extremes those
# blocks are tiny: on live data the calibrated outputs included exactly 0.333
# and 0.667 -- one-third and two-thirds, i.e. blocks of three observations --
# alongside the 0.01/0.99 clips. The EV gate then consumed "p = 0.667" as a
# fact and opened a position on it.
#
# That is the same tail-overfitting the magnitude head had (see MoveCalibrator),
# and it is worse here because selection points the same way: the gate only
# ever fires on the most extreme probabilities, which are exactly the blocks
# with the least support. Backtested out of sample over 4068 bars, the
# isotonic-calibrated gate fired on 0.39% of bars and lost 0.34% per trade at
# a 37.5% win rate -- it was selecting its own artifacts.
DEFAULT_METHOD = "sigmoid"


def _logit(p, eps: float = 1e-6):
    """Log-odds, with the input clipped away from 0 and 1."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class ProbabilityCalibrator:
    """Isotonic map from raw model probability to empirical frequency.

    Falls back to the identity transform when there is not enough data or the
    fit fails, so a calibration problem can never take the pipeline down --
    it degrades to the previous (uncalibrated) behaviour.
    """

    def __init__(self, method: str = DEFAULT_METHOD):
        self.method = method
        self._model = None
        self.n_rows = 0

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, raw_probs, y_true) -> "ProbabilityCalibrator":
        raw = np.asarray(raw_probs, dtype=float)
        y = np.asarray(y_true, dtype=float)
        mask = np.isfinite(raw) & np.isfinite(y)
        raw, y = raw[mask], y[mask]
        self.n_rows = int(len(raw))

        if len(raw) < MIN_CALIBRATION_ROWS or len(np.unique(y)) < 2:
            self._model = None
            return self

        try:
            if self.method == "isotonic":
                from sklearn.isotonic import IsotonicRegression

                model = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
                model.fit(raw, y)
            else:
                from sklearn.linear_model import LogisticRegression

                # Platt scaling on the logit of the raw score. Two parameters
                # over the whole range, so no block of three observations can
                # become a 0.667 the EV gate then bets on.
                model = LogisticRegression(C=1.0, solver="lbfgs")
                model.fit(_logit(raw).reshape(-1, 1), y.astype(int))
            self._model = model
        except Exception:
            self._model = None
        return self

    def transform(self, raw_probs):
        raw = np.asarray(raw_probs, dtype=float)
        if self._model is None:
            return raw
        out = np.full_like(raw, np.nan, dtype=float)
        finite = np.isfinite(raw)
        if not finite.any():
            return out
        if self.method == "isotonic":
            out[finite] = self._model.predict(raw[finite])
        else:
            out[finite] = self._model.predict_proba(
                _logit(raw[finite]).reshape(-1, 1))[:, 1]
        return out


def calibration_error(raw_probs, y_true, n_bins: int = 10) -> float:
    """Expected calibration error: mean |predicted - observed| over bins.

    Logged per symbol so a drift in calibration is visible before it shows up
    as a run of EV-gated trades that all lose.
    """
    raw = np.asarray(raw_probs, dtype=float)
    y = np.asarray(y_true, dtype=float)
    mask = np.isfinite(raw) & np.isfinite(y)
    raw, y = raw[mask], y[mask]
    if len(raw) == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (raw >= lo) & (raw < hi if hi < 1.0 else raw <= hi)
        if not in_bin.any():
            continue
        total += in_bin.sum() * abs(raw[in_bin].mean() - y[in_bin].mean())
    return float(total / len(raw))
