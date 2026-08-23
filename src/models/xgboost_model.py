import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

FEATURE_COLS = [
    "log_return_1h", "log_return_2h", "log_return_4h",
    "log_return_8h", "log_return_24h",
    "rsi_14", "macd", "macd_signal",
    "bb_upper", "bb_lower", "volume_ratio",
]

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
            use_label_encoder=False,
            random_state=42,
        )
        self.scaler = StandardScaler()
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

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = df[FEATURE_COLS].values
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X.copy()
        X_clean[~mask] = 0

        X_scaled = self.scaler.transform(X_clean)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        preds = self.model.predict(X_scaled)

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
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def load(self, path: Path | None = None):
        path = path or MODEL_DIR / f"{self.name}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_fitted = True
