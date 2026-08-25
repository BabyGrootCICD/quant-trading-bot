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


# --- the magnitude forecast rides along on the prediction row --------------

class _CandleClient(FakeClient):
    """FakeClient that can also answer the candle lookup for predicted_price."""

    def __init__(self, close=100.0):
        super().__init__()
        self.close = close

    def table(self, name):
        if name != "candles":
            return super().table(name)
        close = self.close

        class _Q:
            def select(self, *_):
                return self

            def eq(self, *_):
                return self

            def limit(self, *_):
                return self

            def execute(self):
                return type("Resp", (), {"data": [{"close": close}]})()

        return _Q()


def _moves(values):
    return pd.Series(values, index=[0, 1, 2])


def test_prediction_row_carries_the_conditional_move():
    """Without this the engine falls back to the symbol's unconditional
    average, which is the constant that made the EV gate refuse every bar."""
    client = _CandleClient(close=100.0)
    trainer.upsert_latest_prediction(client, "BTC/USDT", _predictions_frame(),
                                     "neural_v1", expected_moves=_moves([0.001, 0.002, 0.015]))

    record = [c for c in client.calls if c[0] == "predictions"][0][2]
    assert record["expected_move_pct"] == pytest.approx(0.015)
    # P(up)=0.8 on a 1.5% move: (2*0.8-1)*0.015 = +0.9%
    assert record["expected_return_pct"] == pytest.approx(0.009)
    assert record["predicted_price"] == pytest.approx(100.0 * np.exp(0.009), rel=1e-6)
    assert record["horizon_hours"] == 1


def test_missing_magnitude_is_written_as_null_not_zero():
    """A zero expected move reads as 'this bar cannot move', which would make
    break-even accuracy infinite rather than unknown."""
    client = _CandleClient()
    trainer.upsert_latest_prediction(client, "BTC/USDT", _predictions_frame(), "neural_v1")

    record = [c for c in client.calls if c[0] == "predictions"][0][2]
    assert record["expected_move_pct"] is None
    assert record["expected_return_pct"] is None
    assert record["predicted_price"] is None


def test_a_nonsense_magnitude_is_dropped_rather_than_trusted():
    client = _CandleClient()
    trainer.upsert_latest_prediction(client, "BTC/USDT", _predictions_frame(),
                                     "neural_v1", expected_moves=_moves([0.01, 0.01, np.nan]))
    record = [c for c in client.calls if c[0] == "predictions"][0][2]
    assert record["expected_move_pct"] is None


def test_a_failing_magnitude_head_does_not_take_training_down():
    """Degraded (engine falls back to live volatility) beats dead."""
    client = FakeClient()
    df = pd.DataFrame({c: [0.0] * 10 for c in trainer.FEATURE_COLS})
    moves, metrics = trainer.train_magnitude_head(client, "BTC/USDT", df)
    assert moves is None
    assert "magnitude_error" in metrics


# --- promotion ladder -------------------------------------------------------

def _metrics_rows(model_name, edge, n=8):
    now = pd.Timestamp.now(tz="UTC")
    return [{"model_name": model_name,
             "sharpe_ratio": 0.0,
             "edge_over_baseline": edge,
             "evaluated_at": (now - pd.Timedelta(hours=i)).isoformat()}
            for i in range(n)]


class _MetricsClient:
    def __init__(self, rows):
        self.rows = rows

    def table(self, _name):
        rows = self.rows

        class _Q:
            def select(self, *_):
                return self

            def order(self, *_, **__):
                return self

            def limit(self, *_):
                return self

            def execute(self):
                return type("Resp", (), {"data": rows})()

        return _Q()


def test_the_ladder_climbs_to_the_neural_head_before_xgboost():
    client = _MetricsClient(_metrics_rows(trainer.LOGISTIC_MODEL_NAME, 0.05))
    assert trainer.check_auto_upgrade(client) == trainer.NEURAL_MODEL_NAME


def test_the_ladder_climbs_from_neural_to_xgboost():
    client = _MetricsClient(_metrics_rows(trainer.NEURAL_MODEL_NAME, 0.05))
    assert trainer.check_auto_upgrade(client) == trainer.XGBOOST_MODEL_NAME


def test_the_top_of_the_ladder_stays_put():
    client = _MetricsClient(_metrics_rows(trainer.XGBOOST_MODEL_NAME, 0.05))
    assert trainer.check_auto_upgrade(client) == trainer.XGBOOST_MODEL_NAME


def test_noise_level_edge_does_not_promote():
    """Every real head sits at +0.6pp to +2.0pp -- i.e. noise."""
    client = _MetricsClient(_metrics_rows(trainer.LOGISTIC_MODEL_NAME, 0.008))
    assert trainer.check_auto_upgrade(client) == trainer.LOGISTIC_MODEL_NAME


def test_one_good_run_is_not_a_track_record():
    rows = _metrics_rows(trainer.LOGISTIC_MODEL_NAME, 0.05, n=3)
    assert trainer.check_auto_upgrade(_MetricsClient(rows)) == trainer.LOGISTIC_MODEL_NAME


# --- the skill record must actually be recorded ----------------------------

def test_the_honest_skill_metrics_are_persisted():
    """They were all computed and then dropped on the floor, which left
    check_auto_upgrade() gating on a column that did not exist."""
    client = FakeClient()
    trainer.log_model_metrics(client, {
        "model_name": "xgboost_v1", "symbol": "BTC/USDT", "test_accuracy": 0.53,
        "up_rate": 0.51, "majority_baseline": 0.51, "balanced_accuracy": 0.529,
        "edge_over_baseline": 0.019, "roc_auc": 0.5547,
        "calibration_error": 0.0, "test_rows": 4065, "ev": 0.06,
    })
    row = [c for c in client.calls if c[0] == "model_metrics"][0][2]
    for k in ("edge_over_baseline", "roc_auc", "balanced_accuracy",
              "majority_baseline", "up_rate", "symbol"):
        assert row.get(k) is not None, f"{k} was dropped"
    assert row["edge_over_baseline"] == pytest.approx(0.019)


def test_the_promotion_gate_reads_a_column_that_is_written():
    """The gate and the writer must agree, or the ladder is dead code."""
    import inspect

    written = set()
    src = inspect.getsource(trainer.log_model_metrics)
    for line in src.splitlines():
        line = line.strip()
        if line.startswith('"') and '":' in line:
            written.add(line.split('"')[1])
    gate = inspect.getsource(trainer.check_auto_upgrade)
    assert "edge_over_baseline" in gate
    assert "edge_over_baseline" in written, "gate reads a field nobody writes"


def test_misleading_trading_shaped_metrics_are_no_longer_written():
    """model_metrics.sharpe_ratio was +1/-1 per out-of-sample call annualised
    at sqrt(8760) -- it read 10.66 for a model with 1.5pp of edge -- and
    win_rate was byte-identical to accuracy. Both looked like trading results
    and were not."""
    client = FakeClient()
    trainer.log_model_metrics(client, {"model_name": "x", "test_accuracy": 0.5,
                                       "sharpe": 10.66, "win_rate": 0.53})
    row = [c for c in client.calls if c[0] == "model_metrics"][0][2]
    assert "sharpe_ratio" not in row
    assert "win_rate" not in row


def test_a_missing_migration_does_not_stop_the_bot_trading():
    """The skill record is diagnostic. A strict preflight on `scored_trades`
    already cost one red pipeline; losing model-quality history must not stop
    the book."""
    class _Rejects(FakeClient):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def table(self, name):
            outer = self

            class _T:
                def insert(_s, payload):
                    outer.attempts += 1
                    if outer.attempts == 1 and "roc_auc" in payload:
                        raise Exception("PGRST204 Could not find the 'roc_auc' column")
                    return FakeTable(outer.calls, name).insert(payload)
            return _T()

    client = _Rejects()
    trainer.log_model_metrics(client, {
        "model_name": "xgboost_v1", "test_accuracy": 0.53, "roc_auc": 0.55,
        "edge_over_baseline": 0.02, "ev": 0.06, "total_signals": 100,
    })
    logged = [c for c in client.calls if c[0] == "model_metrics"]
    assert len(logged) == 1, "must fall back, not give up"
    assert "roc_auc" not in logged[0][2]
    assert logged[0][2]["model_name"] == "xgboost_v1"
