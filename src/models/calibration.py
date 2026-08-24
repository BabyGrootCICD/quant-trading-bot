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

# Below this many out-of-sample observations, an isotonic fit is just
# memorising noise; fall through to the identity map instead.
MIN_CALIBRATION_ROWS = 200


class ProbabilityCalibrator:
    """Isotonic map from raw model probability to empirical frequency.

    Falls back to the identity transform when there is not enough data or the
    fit fails, so a calibration problem can never take the pipeline down --
    it degrades to the previous (uncalibrated) behaviour.
    """

    def __init__(self):
        self._iso = None
        self.n_rows = 0

    @property
    def is_fitted(self) -> bool:
        return self._iso is not None

    def fit(self, raw_probs, y_true) -> "ProbabilityCalibrator":
        raw = np.asarray(raw_probs, dtype=float)
        y = np.asarray(y_true, dtype=float)
        mask = np.isfinite(raw) & np.isfinite(y)
        raw, y = raw[mask], y[mask]
        self.n_rows = int(len(raw))

        if len(raw) < MIN_CALIBRATION_ROWS or len(np.unique(y)) < 2:
            self._iso = None
            return self

        try:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            iso.fit(raw, y)
            self._iso = iso
        except Exception as e:
            print(f"  probability calibration (isotonic) failed, using raw probs: {e}")
            self._iso = None
        return self

    def transform(self, raw_probs):
        raw = np.asarray(raw_probs, dtype=float)
        if self._iso is None:
            return raw
        out = np.full_like(raw, np.nan, dtype=float)
        finite = np.isfinite(raw)
        if finite.any():
            out[finite] = self._iso.predict(raw[finite])
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
