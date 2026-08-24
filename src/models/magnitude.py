"""How big is the next hour's move, given the current bar?

This is the missing half of the strategy. The engine had a direction model and
no magnitude model, so `estimate_expected_moves()` fell back to each symbol's
*unconditional* mean absolute hourly return -- one number per symbol, the same
every hour. Plug a constant into

    EV = (2p - 1) * E|move| - cost

and the gate stops being a filter: EV is then a monotone function of `p` alone,
the threshold is the same in a dead Sunday range as in a volatility spike, and
because the unconditional move is small on every pair in the universe, the
answer was "no" for every symbol on every bar. The bot has opened nothing since
the gate went in. That is safe, but it is not a strategy.

Volatility, unlike direction, is genuinely forecastable -- it clusters. A model
that says "this bar's move will be 0.9%, not the usual 0.22%" moves break-even
accuracy from an impossible 118% down to about 67%, and *that* is a bar worth
having an opinion about. The gate then does what it was designed to do: select
the minority of bars where a real edge can pay for its own execution.

Predicts E|log return| of the next bar. `MLPRegressor` -- a layered network --
with a ridge fallback so a missing/failed neural fit degrades rather than
taking training down, and an isotonic recalibration on top, because the raw
head is dramatically over-optimistic in exactly the top of its range that the
EV gate selects on (see `MoveCalibrator`).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.models.features import FEATURE_COLS

MODEL_DIR = Path(__file__).parent / "checkpoints"

TARGET_COL = "target_move_1h"

# Absolute returns are heavily right-skewed, so the network is trained on
# log1p(move) and predictions are mapped back with expm1. Fitting the raw
# target lets a handful of spike bars dominate the loss.
LOG_TARGET = True

# Floor on the predicted move. A prediction at or below zero would make the EV
# gate divide-by-zero its way to an infinite break-even accuracy; 1bp is small
# enough to be refused by the gate on its own merits.
MIN_PREDICTED_MOVE = 1e-4

# Below this many out-of-sample pairs the magnitude calibration is noise.
MIN_MOVE_CALIBRATION_ROWS = 200


class MoveCalibrator:
    """Maps a raw predicted move onto the move that actually follows it.

    This is not cosmetic. Measured on live hourly bars, the raw head is close
    to unbiased in the middle of its range and wildly optimistic at the top:

        symbol     top-5% predicted   realised   ratio
        BTC/USDT        0.958%          0.310%    0.32
        ETH/USDT        1.847%          0.553%    0.30
        SOL/USDT        0.776%          0.548%    0.71
        DOGE/USDT       5.360%          0.771%    0.14

    The EV gate only ever fires on the top of that range, so *every* trade the
    system takes would be priced off the head's most inflated estimate -- the
    selection and the bias point the same way. Left uncorrected, the gate does
    not admit genuinely volatile bars, it admits bars the model is most wrong
    about, and the EV it reports is fiction.

    An isotonic fit on the walk-forward out-of-sample pairs removes the
    monotone part of that error. It falls back to the identity map when there
    is too little data, which is the uncalibrated behaviour -- degraded, not
    broken.
    """

    def __init__(self):
        self._iso = None
        self.n_rows = 0

    @property
    def is_fitted(self) -> bool:
        return self._iso is not None

    def fit(self, predicted, realised) -> "MoveCalibrator":
        pred = np.asarray(predicted, dtype=float)
        real = np.asarray(realised, dtype=float)
        mask = np.isfinite(pred) & np.isfinite(real)
        pred, real = pred[mask], real[mask]
        self.n_rows = int(len(pred))

        if len(pred) < MIN_MOVE_CALIBRATION_ROWS or len(np.unique(pred)) < 10:
            self._iso = None
            return self
        try:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(y_min=MIN_PREDICTED_MOVE, increasing=True,
                                     out_of_bounds="clip")
            iso.fit(pred, real)
            self._iso = iso
        except Exception as e:
            print(f"  magnitude isotonic calibration failed, using raw predictions: {e}")
            self._iso = None
        return self

    def transform(self, predicted):
        pred = np.asarray(predicted, dtype=float)
        if self._iso is None:
            return np.maximum(pred, MIN_PREDICTED_MOVE)
        out = np.full_like(pred, np.nan, dtype=float)
        finite = np.isfinite(pred)
        if finite.any():
            out[finite] = self._iso.predict(pred[finite])
        return np.maximum(out, MIN_PREDICTED_MOVE)


def tail_ratio(predicted, realised, quantile: float = 0.9) -> float:
    """Realised over predicted move on the top-decile forecasts.

    1.0 means the forecasts the EV gate actually acts on are honest. Well
    below 1.0 means the gate is being fed inflated numbers exactly where it
    makes its decisions.
    """
    pred = np.asarray(predicted, dtype=float)
    real = np.asarray(realised, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(real)
    pred, real = pred[mask], real[mask]
    if len(pred) < 10:
        return 1.0
    top = pred >= np.quantile(pred, quantile)
    if not top.any() or pred[top].mean() <= 0:
        return 1.0
    return float(real[top].mean() / pred[top].mean())


class MagnitudeModel:
    """Conditional E|move| for the next bar."""

    name = "magnitude_mlp_v1"

    def __init__(self, hidden_layer_sizes: tuple[int, ...] = (64, 32),
                 max_iter: int = 400, random_state: int = 42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        # Hourly absolute returns live around 0.002. A network trained on a
        # target that small barely moves off its bias term -- adam's default
        # step size is larger than the whole range of the label, so it
        # converges to "predict the mean", which is exactly the constant the
        # magnitude head exists to replace. Standardising the target fixes the
        # conditioning; predictions are mapped back before they leave here.
        self.target_scaler = StandardScaler()
        # Raw forecasts overshoot badly at the top of their range, which is
        # the only part of the range the EV gate ever looks at. See
        # MoveCalibrator.
        self.calibrator = MoveCalibrator()
        self.is_fitted = False
        self.fallback = False
        # Unconditional mean absolute move, used when a row cannot be scored.
        self.baseline_move = None

    # ------------------------------------------------------------------ data

    @staticmethod
    def _xy(df: pd.DataFrame):
        if TARGET_COL not in df.columns:
            return None, None
        X = df[FEATURE_COLS].to_numpy(dtype=float)
        y = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype=float)
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y)) & (y >= 0)
        return X[mask], y[mask]

    def _fit_encode(self, y):
        raw = np.log1p(y) if LOG_TARGET else np.asarray(y, dtype=float)
        return self.target_scaler.fit_transform(raw.reshape(-1, 1)).ravel()

    def _decode(self, y):
        raw = self.target_scaler.inverse_transform(
            np.asarray(y, dtype=float).reshape(-1, 1)
        ).ravel()
        out = np.expm1(raw) if LOG_TARGET else raw
        return np.maximum(out, MIN_PREDICTED_MOVE)

    # ------------------------------------------------------------- training

    def fit_walk_forward(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        """Walk-forward fit, scored out of sample.

        The score that matters is not R^2 against the raw target -- absolute
        returns are noisy enough that R^2 stays small even for a useful
        forecaster. What matters is whether the ordering is right: do the bars
        the model ranks highest actually move more? `spread_ratio` answers
        that directly (top-decile realised move / overall mean), and it is
        what the EV gate depends on.
        """
        X, y = self._xy(df)
        if X is None:
            return {"error": f"missing {TARGET_COL} column"}
        if len(X) < 500:
            return {"error": "insufficient data", "rows": int(len(X))}

        self.baseline_move = float(np.mean(y))

        tscv = TimeSeriesSplit(n_splits=n_splits)
        oos_pred, oos_true = [], []

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            X_tr_s = self.scaler.fit_transform(X_tr)
            X_te_s = self.scaler.transform(X_te)

            self.fallback = False
            try:
                self.model.fit(X_tr_s, self._fit_encode(y_tr))
                pred = self._decode(self.model.predict(X_te_s))
            except Exception as e:
                # A failed neural fit must not take the whole training run
                # down; ridge on the same features is a weak but honest
                # volatility forecaster.
                print(f"  magnitude neural fit failed, falling back to ridge: {e}")
                self.model = Ridge(alpha=1.0)
                self.fallback = True
                self.model.fit(X_tr_s, self._fit_encode(y_tr))
                pred = self._decode(self.model.predict(X_te_s))

            oos_pred.extend(pred)
            oos_true.extend(y_te)

        self.is_fitted = True

        pred = np.asarray(oos_pred, dtype=float)
        true = np.asarray(oos_true, dtype=float)

        raw_tail = tail_ratio(pred, true)
        self.calibrator.fit(pred, true)
        cal = self.calibrator.transform(pred)

        return {
            "magnitude_model": self.name if not self.fallback else "magnitude_ridge_fallback",
            "magnitude_rows": int(len(true)),
            "magnitude_baseline_move": round(float(np.mean(true)), 6),
            "magnitude_mae": round(float(np.mean(np.abs(cal - true))), 6),
            # Ranking is unaffected by a monotone recalibration, so one number
            # covers both.
            "magnitude_rank_ic": round(rank_ic(pred, true), 4),
            "magnitude_spread_ratio": round(spread_ratio(pred, true), 4),
            # The numbers the EV gate actually acts on.
            "magnitude_tail_ratio_raw": round(raw_tail, 4),
            "magnitude_tail_ratio": round(tail_ratio(cal, true), 4),
            "magnitude_calibration_rows": self.calibrator.n_rows,
        }

    fit = fit_walk_forward

    # ------------------------------------------------------------ inference

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predicted E|move| per row, indexed like `df`.

        Rows with any missing feature fall back to the unconditional mean
        rather than to zero -- zero would read as "this bar cannot move",
        which is a much stronger claim than "we don't know".
        """
        if not self.is_fitted:
            raise RuntimeError("MagnitudeModel not fitted. Call fit_walk_forward() first.")

        X = df[FEATURE_COLS].to_numpy(dtype=float)
        mask = ~np.isnan(X).any(axis=1)
        filled = np.where(np.isnan(X), 0.0, X)

        out = np.full(len(df), self.baseline_move or MIN_PREDICTED_MOVE, dtype=float)
        if mask.any():
            scaled = self.scaler.transform(filled[mask])
            raw = self._decode(self.model.predict(scaled))
            out[mask] = self.calibrator.transform(raw)
        return pd.Series(out, index=df.index, name="expected_abs_move")

    # -------------------------------------------------------- serialisation

    def save(self, path: Path | None = None):
        path = path or MODEL_DIR / f"{self.name}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "target_scaler": self.target_scaler,
                         "calibrator": self.calibrator,
                         "baseline_move": self.baseline_move, "fallback": self.fallback}, f)

    def load(self, path: Path | None = None):
        path = path or MODEL_DIR / f"{self.name}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.target_scaler = data["target_scaler"]
        self.calibrator = data.get("calibrator", MoveCalibrator())
        self.baseline_move = data.get("baseline_move")
        self.fallback = data.get("fallback", False)
        self.is_fitted = True


def rank_ic(pred, true) -> float:
    """Spearman correlation between predicted and realised move size."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[mask], true[mask]
    if len(pred) < 3 or np.all(pred == pred[0]) or np.all(true == true[0]):
        return 0.0
    pr = pd.Series(pred).rank().to_numpy()
    tr = pd.Series(true).rank().to_numpy()
    return float(np.corrcoef(pr, tr)[0, 1])


def spread_ratio(pred, true, quantile: float = 0.9) -> float:
    """Realised mean move on the top-decile predicted bars, over the overall mean.

    1.0 means the magnitude head carries no information and the EV gate is
    back to a constant. Above ~1.5 the gate has something to select on.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(true)
    pred, true = pred[mask], true[mask]
    if len(pred) < 10:
        return 1.0
    overall = float(np.mean(true))
    if overall <= 0:
        return 1.0
    cutoff = float(np.quantile(pred, quantile))
    top = true[pred >= cutoff]
    if len(top) == 0:
        return 1.0
    return float(np.mean(top) / overall)
