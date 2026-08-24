"""The training step failed 8/8 symbols every hour for a full day.

`trainer.train_walk_forward()` calls `model.fit_walk_forward()`, which only
existed on LogisticModel -- so once check_auto_upgrade() selected xgboost_v1,
every symbol died with AttributeError inside a blanket `except`. And the
logistic path was broken too: fit_walk_forward never set is_fitted, so the
follow-up predict() raised RuntimeError.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.features import FEATURE_COLS
from src.models.logistic import LogisticModel
from src.models.xgboost_model import XGBoostModel, HAS_XGBOOST

MODELS = [LogisticModel] + ([XGBoostModel] if HAS_XGBOOST else [])


def _training_frame(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLS})
    df["symbol"] = "BTC/USDT"
    df["timestamp"] = np.arange(n) * 3_600_000
    # Mild learnable signal so the models are not fitting pure noise.
    df["target_1h"] = (df["log_return_1h"] + rng.normal(0, 0.5, n) > 0).astype(int)
    return df


@pytest.mark.parametrize("model_cls", MODELS)
def test_fit_walk_forward_exists_and_runs(model_cls):
    """Both models must satisfy the interface the trainer calls."""
    model = model_cls()
    assert hasattr(model, "fit_walk_forward")

    metrics = model.fit_walk_forward(_training_frame(), n_splits=3)
    assert "error" not in metrics
    assert metrics["model_name"] == model_cls.name
    assert 0.0 <= metrics["test_accuracy"] <= 1.0


@pytest.mark.parametrize("model_cls", MODELS)
def test_predict_works_after_walk_forward(model_cls):
    """This is the RuntimeError('Model not fitted') regression."""
    df = _training_frame()
    model = model_cls()
    model.fit_walk_forward(df, n_splits=3)

    assert model.is_fitted is True
    preds = model.predict(df)
    assert set(["symbol", "timestamp", "prediction", "probability_up", "confidence"]) <= set(preds.columns)
    assert preds["probability_up"].between(0, 1).all()


@pytest.mark.parametrize("model_cls", MODELS)
def test_insufficient_data_returns_error_not_crash(model_cls):
    metrics = model_cls().fit_walk_forward(_training_frame(50), n_splits=3)
    assert metrics["error"] == "insufficient data"


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
def test_both_models_share_one_feature_list():
    """xgboost was stuck on the stale 11-column list while the engineer emitted 26."""
    import src.models.xgboost_model as xgb
    import src.models.logistic as lr
    assert xgb.FEATURE_COLS is lr.FEATURE_COLS is FEATURE_COLS


@pytest.mark.parametrize("model_cls", MODELS)
def test_walk_forward_exposes_out_of_sample_predictions(model_cls):
    """The trainer needs these to score honestly instead of in-sample."""
    metrics = model_cls().fit_walk_forward(_training_frame(), n_splits=3)

    assert len(metrics["oos_preds"]) == len(metrics["oos_y_true"])
    assert len(metrics["oos_preds"]) == metrics["test_rows"]

    # And they must agree with the reported accuracy.
    agree = sum(1 for p, y in zip(metrics["oos_preds"], metrics["oos_y_true"]) if p == y)
    assert agree / len(metrics["oos_preds"]) == pytest.approx(metrics["test_accuracy"], abs=1e-4)
