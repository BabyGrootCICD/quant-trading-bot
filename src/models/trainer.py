import sys
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import SYMBOLS, env_str
from src.data.supabase_client import get_client
from src.models.features import FEATURE_COLS
from src.models.logistic import LogisticModel
from src.models.magnitude import MagnitudeModel
from src.models.neural import NeuralModel
from src.models.xgboost_model import XGBoostModel, HAS_XGBOOST
from src.utils.metrics import sharpe_ratio, expected_value, win_rate
from src.strategy.economics import (
    breakeven_accuracy, expected_return, predicted_price, round_trip_cost_pct,
)

# Cap on how much history each training run pulls. Two years of hourly bars
# is ~17.5k rows; this keeps the whole window without unbounded growth.
MAX_TRAINING_ROWS = 20000

# Minimum accuracy edge over the majority-class baseline before a model is
# considered good enough to promote. Every current model sits under 2pp.
MIN_PROMOTION_EDGE = 0.02

LOGISTIC_MODEL_NAME = LogisticModel.name
NEURAL_MODEL_NAME = NeuralModel.name
XGBOOST_MODEL_NAME = XGBoostModel.name

# Promotion ladder. A rung is only climbed once the model currently in use has
# shown a real edge over the majority baseline for a sustained stretch --
# "the features carry signal, so a stronger learner is worth the variance".
PROMOTION_LADDER = {
    LOGISTIC_MODEL_NAME: NEURAL_MODEL_NAME,
    NEURAL_MODEL_NAME: XGBOOST_MODEL_NAME,
}

# Consecutive recent evaluations that must all clear MIN_PROMOTION_EDGE.
MIN_PROMOTION_RUNS = 7

# Operator override. Set ACTIVE_MODEL to pin a model and skip the ladder.
ACTIVE_MODEL_OVERRIDE = env_str("ACTIVE_MODEL", "")

DIRECTIONAL_MODELS = {
    LOGISTIC_MODEL_NAME: LogisticModel,
    NEURAL_MODEL_NAME: NeuralModel,
    XGBOOST_MODEL_NAME: XGBoostModel,
}


def fetch_features(client, symbol: str, max_rows: int = MAX_TRAINING_ROWS) -> pd.DataFrame:
    """Most recent `max_rows` feature rows for `symbol`, oldest-first.

    This used to be an unpaginated `.select("*")` ordered ascending. PostgREST
    caps a response at 1000 rows, so the trainer silently received the *oldest*
    1000 rows -- it trained on two-year-old data and stamped its prediction
    with a two-year-old timestamp, which the engine then rejected as stale.

    Paginate explicitly, newest-first, then flip to chronological order for
    the walk-forward split.
    """
    all_data = []
    page = 1000
    offset = 0

    while offset < max_rows:
        limit = min(page, max_rows - offset)
        resp = (
            client.table("features")
            .select("*")
            .eq("symbol", symbol)
            .order("timestamp", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        if not resp.data:
            break
        all_data.extend(resp.data)
        if len(resp.data) < limit:
            break
        offset += limit

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in FEATURE_COLS + ["target_1h", "target_4h", "target_move_1h"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def skill_metrics(metrics: dict) -> dict:
    """Accuracy relative to the majority-class baseline.

    Raw accuracy is meaningless on an imbalanced label. TRX/USDT scored 0.6502
    and a Sharpe of 29.47, which reads as a strong model -- but its up-rate is
    0.3587, so always answering "down" scores 0.6413. The real edge was
    +0.89pp, in line with every other symbol. `check_auto_upgrade()` gates on
    Sharpe, so that artifact drove model promotion too.
    """
    y_true = metrics.get("oos_y_true") or []
    preds = metrics.get("oos_preds") or []
    if not y_true or not preds:
        return {}

    n = len(y_true)
    up_rate = sum(1 for y in y_true if y == 1) / n
    baseline = max(up_rate, 1 - up_rate)
    accuracy = sum(1 for p, y in zip(preds, y_true) if p == y) / n

    # Balanced accuracy: mean of per-class recall, immune to class skew.
    recalls = []
    for cls in (0, 1):
        actual = [i for i, y in enumerate(y_true) if y == cls]
        if actual:
            recalls.append(sum(1 for i in actual if preds[i] == cls) / len(actual))
    balanced = sum(recalls) / len(recalls) if recalls else 0.0

    return {
        "up_rate": round(up_rate, 4),
        "majority_baseline": round(baseline, 4),
        "balanced_accuracy": round(balanced, 4),
        "edge_over_baseline": round(accuracy - baseline, 4),
    }


def out_of_sample_pnl(metrics: dict) -> list[float]:
    """+1 per correct out-of-sample call, -1 per incorrect one.

    Scoring used to run `model.predict(df)` over the whole feature frame,
    most of which the final walk-forward fold had trained on. That reported a
    0.91 win rate and a Sharpe of 135 against a true test accuracy of 0.52 --
    and check_auto_upgrade() gates on `sharpe_ratio > 1`, so the bogus figure
    drove model promotion too.
    """
    preds = metrics.get("oos_preds")
    y_true = metrics.get("oos_y_true")
    if not preds or not y_true:
        return []
    return [1.0 if p == y else -1.0 for p, y in zip(preds, y_true)]


def latest_close(client, symbol: str, timestamp: int) -> float | None:
    """Close of the candle the prediction was made on, for the price forecast."""
    try:
        resp = (
            client.table("candles")
            .select("close")
            .eq("symbol", symbol)
            .eq("timestamp", int(timestamp))
            .limit(1)
            .execute()
        )
    except Exception:
        return None
    if not resp.data:
        return None
    return float(resp.data[0]["close"])


def upsert_latest_prediction(client, symbol: str, predictions: pd.DataFrame, model_name: str,
                             expected_moves: pd.Series | None = None) -> bool:
    """Write the newest prediction for `symbol` to the `predictions` table.

    Nothing in the pipeline used to do this. The trainer computed predictions
    and discarded them, so the paper-trading engine kept re-reading one frozen
    row per symbol and churning the same positions hour after hour, paying the
    round-trip spread on a signal that never changed.

    The row now also carries the magnitude head's forecast. Previously the
    engine re-derived an expected move itself from live candles, as each
    symbol's unconditional average -- a constant, which is what made the EV
    gate refuse every bar. Writing the conditional forecast here means the
    number the model produced is the number the allocator sizes on.
    """
    valid = predictions.dropna(subset=["prediction", "probability_up"])
    if valid.empty:
        return False

    valid = valid.sort_values("timestamp")
    row = valid.iloc[-1]
    ts = int(row["timestamp"])
    prob_up = float(row["probability_up"])

    record = {
        "symbol": symbol,
        "timestamp": ts,
        "model_name": model_name,
        "prediction": float(row["prediction"]),
        "probability_up": prob_up,
        "confidence": float(row["confidence"]),
        "horizon_hours": 1,
        "expected_move_pct": None,
        "expected_return_pct": None,
        "predicted_price": None,
    }

    if expected_moves is not None and row.name in expected_moves.index:
        move = float(expected_moves.loc[row.name])
        if np.isfinite(move) and move > 0:
            record["expected_move_pct"] = round(move, 8)
            record["expected_return_pct"] = round(expected_return(prob_up, move), 8)
            close = latest_close(client, symbol, ts)
            if close:
                record["predicted_price"] = round(predicted_price(close, prob_up, move), 8)

    client.table("predictions").upsert(record, on_conflict="symbol,timestamp").execute()
    return True


def train_magnitude_head(client, symbol: str, df: pd.DataFrame) -> tuple[pd.Series | None, dict]:
    """Fit the conditional E|move| model and score every row of `df`.

    A failure here must not block the directional model: without a magnitude
    forecast the engine falls back to its live-candle estimate, which is the
    pre-patch behaviour -- degraded, not broken.
    """
    model = MagnitudeModel()
    try:
        metrics = model.fit_walk_forward(df)
    except Exception as e:
        return None, {"magnitude_error": str(e)}
    if "error" in metrics:
        return None, {"magnitude_error": metrics["error"]}
    try:
        return model.predict(df), metrics
    except Exception as e:
        return None, {"magnitude_error": str(e)}


def train_walk_forward(client, symbol: str, model, n_splits: int = 5) -> dict:
    df = fetch_features(client, symbol)
    if df.empty or len(df) < 500:
        return {"symbol": symbol, "error": "insufficient data"}

    metrics = model.fit_walk_forward(df, n_splits=n_splits)
    if "error" in metrics:
        return {"symbol": symbol, **metrics}

    predictions = model.predict(df)
    valid = predictions.dropna(subset=["prediction"])
    if valid.empty:
        return {"symbol": symbol, **metrics, "trade_signals": 0}

    expected_moves, mag_metrics = train_magnitude_head(client, symbol, df)
    metrics.update(mag_metrics)

    metrics["prediction_written"] = upsert_latest_prediction(
        client, symbol, valid, metrics.get("model_name", model.name),
        expected_moves=expected_moves,
    )

    all_pnl = out_of_sample_pnl(metrics)

    metrics["symbol"] = symbol
    metrics["total_signals"] = len(all_pnl)
    metrics["long_signals"] = int(sum(1 for p in metrics.get("oos_preds", []) if p == 1))
    metrics["short_signals"] = int(sum(1 for p in metrics.get("oos_preds", []) if p == 0))
    metrics["ev"] = round(expected_value(all_pnl), 4) if all_pnl else 0.0
    metrics["sharpe"] = round(sharpe_ratio(all_pnl), 4) if all_pnl else 0.0
    metrics["win_rate"] = round(win_rate(all_pnl), 4) if all_pnl else 0.0

    metrics.update(skill_metrics(metrics))

    # Bulky and already summarised; do not carry into the metrics row.
    metrics.pop("oos_preds", None)
    metrics.pop("oos_y_true", None)
    metrics.pop("oos_probas", None)

    return metrics


def train_and_evaluate(client, symbol: str, model) -> dict:
    df = fetch_features(client, symbol)
    if df.empty or len(df) < 200:
        return {"symbol": symbol, "error": "insufficient data"}

    metrics = model.train_and_evaluate(client, symbol)
    if "error" in metrics:
        return {"symbol": symbol, **metrics}

    predictions = model.predict(df)
    valid = predictions.dropna(subset=["prediction"])
    if valid.empty:
        return {"symbol": symbol, **metrics, "trade_signals": 0}

    metrics["prediction_written"] = upsert_latest_prediction(
        client, symbol, valid, metrics.get("model_name", model.name)
    )

    all_pnl = out_of_sample_pnl(metrics)

    metrics["symbol"] = symbol
    metrics["total_signals"] = len(all_pnl)
    metrics["long_signals"] = int(sum(1 for p in metrics.get("oos_preds", []) if p == 1))
    metrics["short_signals"] = int(sum(1 for p in metrics.get("oos_preds", []) if p == 0))
    metrics["ev"] = round(expected_value(all_pnl), 4) if all_pnl else 0.0
    metrics["sharpe"] = round(sharpe_ratio(all_pnl), 4) if all_pnl else 0.0
    metrics["win_rate"] = round(win_rate(all_pnl), 4) if all_pnl else 0.0

    metrics.update(skill_metrics(metrics))

    # Bulky and already summarised; do not carry into the metrics row.
    metrics.pop("oos_preds", None)
    metrics.pop("oos_y_true", None)
    metrics.pop("oos_probas", None)

    return metrics


def log_model_metrics(client, metrics: dict):
    if "error" in metrics:
        return
    client.table("model_metrics").insert({
        "model_name": metrics.get("model_name", "unknown"),
        "accuracy": metrics.get("test_accuracy"),
        "precision_up": metrics.get("expected_value_long"),
        "recall_up": metrics.get("expected_value_short"),
        "expected_value": metrics.get("ev"),
        "sharpe_ratio": metrics.get("sharpe"),
        "total_trades": metrics.get("total_signals"),
        "win_rate": metrics.get("win_rate"),
    }).execute()


def check_auto_upgrade(client) -> str:
    resp = (
        client.table("model_metrics")
        .select("model_name,sharpe_ratio,evaluated_at")
        .order("evaluated_at", desc=True)
        .limit(168)
        .execute()
    )
    if not resp.data:
        return LOGISTIC_MODEL_NAME

    df = pd.DataFrame(resp.data)
    df["evaluated_at"] = pd.to_datetime(df["evaluated_at"])

    now = datetime.now(timezone.utc)
    week_ago = now - pd.Timedelta(days=7)
    recent = df[df["evaluated_at"] >= week_ago]

    if recent.empty:
        return LOGISTIC_MODEL_NAME

    current_model = recent.iloc[0]["model_name"]
    next_model = PROMOTION_LADDER.get(current_model)
    if next_model is None:
        return current_model

    current_recent = recent[recent["model_name"] == current_model]
    # Gate on demonstrated edge over the majority-class baseline. Gating on
    # sharpe_ratio promoted models on TRX's class-imbalance artifact.
    if "edge_over_baseline" not in current_recent.columns:
        return current_model

    edge = pd.to_numeric(current_recent["edge_over_baseline"], errors="coerce")
    if len(current_recent) >= MIN_PROMOTION_RUNS and (edge > MIN_PROMOTION_EDGE).all():
        return next_model

    return current_model


def main():
    print("=" * 60)
    print("Quant Bot - Model Training Pipeline")
    print("=" * 60)

    client = get_client()

    if ACTIVE_MODEL_OVERRIDE:
        active_model_name = ACTIVE_MODEL_OVERRIDE
        print(f"Active model: {active_model_name} (pinned via ACTIVE_MODEL)")
    else:
        active_model_name = check_auto_upgrade(client)
        print(f"Active model: {active_model_name}")

    if active_model_name not in DIRECTIONAL_MODELS:
        print(f"WARNING: unknown model '{active_model_name}', falling back to logistic")
        active_model_name = LOGISTIC_MODEL_NAME

    if active_model_name == XGBOOST_MODEL_NAME and not HAS_XGBOOST:
        print("WARNING: xgboost not installed, falling back to logistic")
        active_model_name = LOGISTIC_MODEL_NAME

    print(f"Round-trip cost: {round_trip_cost_pct()*100:.3f}% of position")

    all_metrics = []
    for symbol in SYMBOLS:
        print(f"\nTraining on {symbol}...")
        try:
            model = DIRECTIONAL_MODELS[active_model_name]()

            metrics = train_walk_forward(client, symbol, model, n_splits=5)
            all_metrics.append(metrics)
            log_model_metrics(client, metrics)

            if "error" not in metrics:
                print(f"  Accuracy: {metrics.get('test_accuracy', 'N/A')} "
                      f"(baseline {metrics.get('majority_baseline', 'N/A')}, "
                      f"edge {metrics.get('edge_over_baseline', 'N/A')})")
                print(f"  Balanced accuracy: {metrics.get('balanced_accuracy', 'N/A')}")
                print(f"  Calibration error: {metrics.get('calibration_error', 'N/A')} "
                      f"(raw {metrics.get('calibration_error_raw', 'N/A')})")
                if "magnitude_error" in metrics:
                    print(f"  Magnitude head: FAILED ({metrics['magnitude_error']}) "
                          "- engine will fall back to unconditional volatility")
                else:
                    print(f"  Magnitude head: rank IC {metrics.get('magnitude_rank_ic', 'N/A')}, "
                          f"top-decile move x{metrics.get('magnitude_spread_ratio', 'N/A')}")
                    mv = metrics.get("magnitude_baseline_move")
                    if mv:
                        # The gate's job is to find the bars where this number
                        # is far above its unconditional level; at the average
                        # bar the bar is usually unreachable.
                        print(f"  Unconditional move {mv*100:.3f}% -> break-even accuracy "
                              f"{breakeven_accuracy(mv)*100:.1f}%")
                print(f"  Sharpe: {metrics.get('sharpe', 'N/A')}")
                print(f"  Win rate: {metrics.get('win_rate', 'N/A')}")
                print(f"  Signals: {metrics.get('total_signals', 0)}")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_metrics.append({"symbol": symbol, "error": str(e)})

    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    valid = [m for m in all_metrics if "error" not in m]
    if valid:
        avg_ev = np.mean([m["ev"] for m in valid])
        avg_sharpe = np.mean([m["sharpe"] for m in valid])
        avg_wr = np.mean([m["win_rate"] for m in valid])
        print(f"Models trained: {len(valid)}/{len(SYMBOLS)}")
        print(f"Average EV: {avg_ev:.4f}")
        print(f"Average Sharpe: {avg_sharpe:.4f}")
        print(f"Average Win Rate: {avg_wr:.4f}")
        print(f"Active model: {active_model_name}")

        if avg_sharpe > 1:
            print("\n*** Sharpe > 1 achieved! System is performing well. ***")
        elif avg_ev > 0:
            print("\nPositive EV detected. Keep monitoring.")
        else:
            print("\nWarning: Negative average EV. Review features and data quality.")
    else:
        print("No models trained successfully.")

    print("=" * 60)

    # Fail the pipeline instead of reporting green on a total training
    # failure. Eight consecutive silent AttributeErrors are how the bot ended
    # up trading a frozen prediction set for a full day.
    if not valid:
        print("FATAL: every symbol failed to train. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()