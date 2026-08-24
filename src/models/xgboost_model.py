import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.models.calibration import ProbabilityCalibrator, calibration_error
from src.models.features import FEATURE_COLS

MODEL_DIR = Path(__file__).parent / "checkpoints"


class XGBoostModel:
    name = "xgboost_v1"

    def __init__(self):
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.calibrator = ProbabilityCalibrator()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> dict:
        X = df[FEATURE_COLS].values
        y = df["target_1h"].values

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        if len(X) < 100:
            return {"error": "insufficient data", "rows": len(X)}

        X_train, X_test = X[: int(len(X) * 0.8)], X[int(len(X) * 0.8) :]
        y_train, y_test = y[: int(len(y) * 0.8)], y[int(len(y) * 0.8) :]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True

        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)

        preds_test = self.model.predict(X_test_scaled)

        up_mask = preds_test == 1
        down_mask = preds_test == 0
        ev_up = float(np.mean(y_test[up_mask])) if up_mask.any() else 0.0
        ev_down = float(np.mean(1 - y_test[down_mask])) if down_mask.any() else 0.0

        importances = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in self.model.feature_importances_]))

        return {
            "model_name": self.name,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "expected_value_long": round(ev_up, 4),
            "expected_value_short": round(ev_down, 4),
            "feature_importances": importances,
        }

    def fit_walk_forward(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
        """Walk-forward CV, mirroring LogisticModel.fit_walk_forward.

        The trainer calls this on whichever model is active. It was missing
        here, so once check_auto_upgrade() selected xgboost_v1 every symbol
        died with AttributeError and no model was ever trained.
        """
        X = df[FEATURE_COLS].values
        y = df["target_1h"].values

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        if len(X) < 500:
            return {"error": "insufficient data", "rows": len(X)}

        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_preds = []
        all_probas = []
        all_y_true = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.model.fit(X_train_scaled, y_train)

            all_preds.extend(self.model.predict(X_test_scaled))
            all_probas.extend(self.model.predict_proba(X_test_scaled)[:, 1])
            all_y_true.extend(y_test)

        self.is_fitted = True

        all_preds = np.array(all_preds)
        all_probas = np.array(all_probas, dtype=float)
        all_y_true = np.array(all_y_true)

        raw_ece = calibration_error(all_probas, all_y_true)
        self.calibrator.fit(all_probas, all_y_true)
        cal_ece = calibration_error(self.calibrator.transform(all_probas), all_y_true)

        acc = float(np.mean(all_preds == all_y_true))

        up_mask = all_preds == 1
        down_mask = all_preds == 0
        ev_up = float(np.mean(all_y_true[up_mask])) if up_mask.any() else 0.0
        ev_down = float(np.mean(1 - all_y_true[down_mask])) if down_mask.any() else 0.0

        importances = dict(zip(FEATURE_COLS, [round(float(v), 4) for v in self.model.feature_importances_]))

        test_rows = len(all_y_true)

        return {
            "model_name": self.name,
            "train_rows": len(X) - test_rows,
            "test_rows": test_rows,
            "train_accuracy": round(acc, 4),
            "test_accuracy": round(acc, 4),
            "precision": round(float(precision_score(all_y_true, all_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(all_y_true, all_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(all_y_true, all_preds, zero_division=0)), 4),
            "expected_value_long": round(ev_up, 4),
            "expected_value_short": round(ev_down, 4),
            "feature_importances": importances,
            # Out-of-sample fold predictions. The trainer must score on
            # these, not on predict() over the full frame -- that frame
            # includes the rows the final fold trained on, which is how
            # an honest 0.52 accuracy got reported as a 0.91 win rate.
            "oos_preds": all_preds.tolist(),
            "oos_y_true": all_y_true.tolist(),
            "oos_probas": all_probas.tolist(),
            "calibration_error_raw": round(float(raw_ece), 4),
            "calibration_error": round(float(cal_ece), 4),
            "calibration_rows": self.calibrator.n_rows,
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = df[FEATURE_COLS].values
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X.copy()
        X_clean[~mask] = 0

        X_scaled = self.scaler.transform(X_clean)
        proba = self.calibrator.transform(self.model.predict_proba(X_scaled)[:, 1])
        preds = (proba > 0.5).astype(float)

        result = df[["symbol", "timestamp"]].copy()
        result["prediction"] = preds
        result["probability_up"] = proba
        result["confidence"] = np.abs(proba - 0.5) * 2
        result.loc[~mask, ["prediction", "probability_up", "confidence"]] = np.nan
        return result

    def save(self, path: Path | None = None):
        path = path or MODEL_DIR / f"{self.name}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "calibrator": self.calibrator}, f)

    def load(self, path: Path | None = None):
        path = path or MODEL_DIR / f"{self.name}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.calibrator = data.get("calibrator", ProbabilityCalibrator())
        self.is_fitted = True
