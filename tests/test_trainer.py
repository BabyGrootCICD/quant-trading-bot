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


# --- out-of-sample scoring -------------------------------------------------

def test_pnl_comes_from_out_of_sample_folds():
    """Scoring on predict() over the full frame reported 0.91 against a true 0.52."""
    metrics = {"oos_preds": [1, 0, 1, 0], "oos_y_true": [1, 0, 0, 1]}
    assert trainer.out_of_sample_pnl(metrics) == [1.0, 1.0, -1.0, -1.0]


def test_win_rate_matches_out_of_sample_accuracy():
    """A 52% model must score ~0.52, not 0.91."""
    from src.utils.metrics import win_rate
    preds = [1] * 100
    truth = [1] * 52 + [0] * 48
    pnl = trainer.out_of_sample_pnl({"oos_preds": preds, "oos_y_true": truth})
    assert win_rate(pnl) == pytest.approx(0.52)


def test_missing_oos_arrays_degrade_to_empty():
    assert trainer.out_of_sample_pnl({}) == []
    assert trainer.out_of_sample_pnl({"oos_preds": [], "oos_y_true": []}) == []


def test_oos_arrays_are_stripped_before_logging():
    """They are large and would bloat every model_metrics row."""
    import inspect
    src = inspect.getsource(trainer.train_walk_forward)
    assert 'metrics.pop("oos_preds", None)' in src
    assert 'metrics.pop("oos_y_true", None)' in src


# --- pagination ------------------------------------------------------------

class PagingClient:
    """Mimics PostgREST: refuses to return more than 1000 rows per request."""

    def __init__(self, n_rows):
        self.n_rows = n_rows
        self.descending = None
        self.rows = [{"symbol": "BTC/USDT", "timestamp": i * 3_600_000, "target_1h": 0}
                     for i in range(n_rows)]

    def table(self, name):
        return self

    def select(self, *a):
        return self

    def eq(self, *a):
        return self

    def order(self, col, desc=False):
        self.descending = desc
        return self

    def range(self, lo, hi):
        ordered = list(reversed(self.rows)) if self.descending else self.rows
        self._slice = ordered[lo : min(hi + 1, lo + 1000)]
        return self

    def execute(self):
        return type("Resp", (), {"data": self._slice})()


def test_fetch_features_pages_past_the_1000_row_cap():
    client = PagingClient(5000)
    df = trainer.fetch_features(client, "BTC/USDT")
    assert len(df) == 5000


def test_fetch_features_returns_the_newest_rows_not_the_oldest():
    """The bug: ascending + no pagination handed back the oldest 1000 rows."""
    client = PagingClient(5000)
    df = trainer.fetch_features(client, "BTC/USDT", max_rows=1000)

    assert len(df) == 1000
    newest = 4999 * 3_600_000
    assert df["timestamp"].max() == newest, "must include the most recent bar"
    assert df["timestamp"].min() == 4000 * 3_600_000


def test_fetch_features_returns_chronological_order():
    """Walk-forward splitting depends on chronological order."""
    df = trainer.fetch_features(PagingClient(3000), "BTC/USDT")
    assert df["timestamp"].is_monotonic_increasing


def test_fetch_features_handles_empty_table():
    assert trainer.fetch_features(PagingClient(0), "BTC/USDT").empty


# --- honest skill metrics --------------------------------------------------

def test_trx_style_imbalance_is_exposed_not_rewarded():
    """TRX scored 0.6502 accuracy purely by always predicting the majority class."""
    y_true = [0] * 641 + [1] * 359          # up-rate 0.359, as measured
    preds = [0] * 1000                       # always "down"
    sk = trainer.skill_metrics({"oos_preds": preds, "oos_y_true": y_true})

    assert sk["majority_baseline"] == pytest.approx(0.641)
    assert sk["edge_over_baseline"] == pytest.approx(0.0)
    # Balanced accuracy sees straight through it.
    assert sk["balanced_accuracy"] == pytest.approx(0.5)


def test_real_edge_is_reported_as_positive():
    y_true = [0, 1] * 500
    preds = [y if i % 10 else 1 - y for i, y in enumerate(y_true)]
    sk = trainer.skill_metrics({"oos_preds": preds, "oos_y_true": y_true})
    assert sk["edge_over_baseline"] > 0.3
    assert sk["balanced_accuracy"] > 0.8


def test_skill_metrics_on_empty_input():
    assert trainer.skill_metrics({}) == {}


def test_promotion_gate_uses_edge_not_sharpe():
    """Gating on sharpe_ratio promoted models on TRX's imbalance artifact."""
    import inspect
    src = inspect.getsource(trainer.check_auto_upgrade)
    assert "edge_over_baseline" in src
    assert "sharpe_ratio\"] > 1" not in src
