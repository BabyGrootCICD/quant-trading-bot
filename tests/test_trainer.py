"""The trainer computed predictions and threw them away.

Nothing in the codebase wrote to `predictions`, so the engine kept re-reading
one frozen row per symbol forever.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import trainer


class FakeTable:
    def __init__(self, recorder, name):
        self.recorder, self.name = recorder, name
        self.payload = None

    def upsert(self, record, on_conflict=None):
        self.recorder.append((self.name, "upsert", record, on_conflict))
        return self

    def insert(self, record):
        self.recorder.append((self.name, "insert", record, None))
        return self

    def execute(self):
        return type("Resp", (), {"data": []})()


class FakeClient:
    def __init__(self):
        self.calls = []

    def table(self, name):
        return FakeTable(self.calls, name)


def _predictions_frame():
    return pd.DataFrame({
        "symbol": ["BTC/USDT"] * 3,
        "timestamp": [1000, 2000, 3000],
        "prediction": [1.0, 0.0, 1.0],
        "probability_up": [0.7, 0.3, 0.8],
        "confidence": [0.4, 0.4, 0.6],
    })


def test_latest_prediction_is_persisted():
    client = FakeClient()
    assert trainer.upsert_latest_prediction(client, "BTC/USDT", _predictions_frame(), "logistic_v2")

    writes = [c for c in client.calls if c[0] == "predictions"]
    assert len(writes) == 1
    table, op, record, on_conflict = writes[0]
    assert op == "upsert"
    assert on_conflict == "symbol,timestamp"
    # Must be the newest row, not whichever happened to be first.
    assert record["timestamp"] == 3000
    assert record["probability_up"] == pytest.approx(0.8)
    assert record["model_name"] == "logistic_v2"


def test_no_write_when_all_predictions_are_nan():
    client = FakeClient()
    df = _predictions_frame()
    df["prediction"] = np.nan
    assert trainer.upsert_latest_prediction(client, "BTC/USDT", df, "logistic_v2") is False
    assert not [c for c in client.calls if c[0] == "predictions"]


def test_model_names_are_not_hardcoded_stale_strings():
    """check_auto_upgrade compared against 'logistic_v1' while the class was 'logistic_v2'."""
    from src.models.logistic import LogisticModel
    from src.models.xgboost_model import XGBoostModel
    assert trainer.LOGISTIC_MODEL_NAME == LogisticModel.name
    assert trainer.XGBOOST_MODEL_NAME == XGBoostModel.name
