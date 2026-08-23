import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from src.models.features import FEATURE_COLS

MODEL_DIR = Path(__file__).parent / "checkpoints"


class LogisticModel:
    name = "logistic_v2"

    def __init__(self, C=1.0, penalty="l2", solver="lbfgs", max_iter=2000):
        self.model = LogisticRegression(
            C=C, penalty=penalty, solver=solver, max_iter=max_iter, class_weight="balanced"
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.best_params = None

    def fit_walk_forward(self, df: pd.DataFrame, n_splits: int = 5) -> dict:
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

            preds = self.model.predict(X_test_scaled)
            probas = self.model.predict_proba(X_test_scaled)[:, 1]

            all_preds.extend(preds)
            all_probas.extend(probas)
            all_y_true.extend(y_test)

        # The scaler/model now hold the final fold's fit; mark fitted so
        # predict() works on the caller's side (this was missing, so every
        # walk-forward run died with RuntimeError('Model not fitted')).
        self.is_fitted = True

        all_preds = np.array(all_preds)
        all_probas = np.array(all_probas)
        all_y_true = np.array(all_y_true)

        train_acc = np.mean(all_preds == all_y_true)

        up_mask = all_preds == 1
        down_mask = all_preds == 0
        ev_up = float(np.mean(all_y_true[up_mask])) if up_mask.any() else 0.0
        ev_down = float(np.mean(1 - all_y_true[down_mask])) if down_mask.any() else 0.0

        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(all_y_true, all_preds, zero_division=0)
        recall = recall_score(all_y_true, all_preds, zero_division=0)
        f1 = f1_score(all_y_true, all_preds, zero_division=0)

        test_rows = len(all_y_true)
        train_rows = len(X) - test_rows

        return {
            "model_name": self.name,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(train_acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "expected_value_long": round(ev_up, 4),
            "expected_value_short": round(ev_down, 4),
            "coefficients": dict(zip(FEATURE_COLS, [round(c, 6) for c in self.model.coef_[0]])),
            "intercept": round(float(self.model.intercept_[0]), 6),
        }

    def fit(self, df: pd.DataFrame, use_walk_forward: bool = True) -> dict:
        if use_walk_forward:
            return self.fit_walk_forward(df)
        else:
            return self.fit_simple(df)

    def fit_simple(self, df: pd.DataFrame) -> dict:
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

        proba_test = self.model.predict_proba(X_test_scaled)[:, 1]
        preds_test = self.model.predict(X_test_scaled)

        up_mask = preds_test == 1
        down_mask = preds_test == 0
        ev_up = float(np.mean(y_test[up_mask])) if up_mask.any() else 0.0
        ev_down = float(np.mean(1 - y_test[down_mask])) if down_mask.any() else 0.0

        return {
            "model_name": self.name,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "expected_value_long": round(ev_up, 4),
            "expected_value_short": round(ev_down, 4),
            "coefficients": dict(zip(FEATURE_COLS, [round(c, 6) for c in self.model.coef_[0]])),
            "intercept": round(float(self.model.intercept_[0]), 6),
        }

    def tune_hyperparameters(self, df: pd.DataFrame) -> dict:
        X = df[FEATURE_COLS].values
        y = df["target_1h"].values

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        if len(X) < 500:
            return {"error": "insufficient data", "rows": len(X)}

        X_scaled = self.scaler.fit_transform(X)

        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
            "max_iter": [2000],
            "class_weight": ["balanced"],
        }

        tscv = TimeSeriesSplit(n_splits=3)
        grid = GridSearchCV(
            LogisticRegression(),
            param_grid,
            cv=tscv,
            scoring="f1",
            n_jobs=-1,
        )
        grid.fit(X_scaled, y)

        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.is_fitted = True

        return {
            "best_params": grid.best_params_,
            "best_score": round(grid.best_score_, 4),
            "cv_results": grid.cv_results_,
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
            pickle.dump({"model": self.model, "scaler": self.scaler, "best_params": self.best_params}, f)

    def load(self, path: Path | None = None):
        path = path or MODEL_DIR / f"{self.name}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.best_params = data.get("best_params")
        self.is_fitted = True