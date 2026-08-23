import sys
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import SYMBOLS
from src.data.supabase_client import get_client
from src.models.features import FEATURE_COLS
from src.models.logistic import LogisticModel
from src.models.xgboost_model import XGBoostModel, HAS_XGBOOST
from src.utils.metrics import sharpe_ratio, expected_value, win_rate

LOGISTIC_MODEL_NAME = LogisticModel.name
XGBOOST_MODEL_NAME = XGBoostModel.name


def fetch_features(client, symbol: str) -> pd.DataFrame:
    resp = (
        client.table("features")
        .select("*")
        .eq("symbol", symbol)
        .order("timestamp", desc=False)
        .execute()
    )
    if not resp.data:
        return pd.DataFrame()
    df = pd.DataFrame(resp.data)
    for col in FEATURE_COLS + ["target_1h"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def upsert_latest_prediction(client, symbol: str, predictions: pd.DataFrame, model_name: str) -> bool:
    """Write the newest prediction for `symbol` to the `predictions` table.

    Nothing in the pipeline used to do this. The trainer computed predictions
    and discarded them, so the paper-trading engine kept re-reading one frozen
    row per symbol and churning the same positions hour after hour, paying the
    round-trip spread on a signal that never changed.
    """
    valid = predictions.dropna(subset=["prediction", "probability_up"])
    if valid.empty:
        return False

    row = valid.sort_values("timestamp").iloc[-1]
    client.table("predictions").upsert({
        "symbol": symbol,
        "timestamp": int(row["timestamp"]),
        "model_name": model_name,
        "prediction": float(row["prediction"]),
        "probability_up": float(row["probability_up"]),
        "confidence": float(row["confidence"]),
    }, on_conflict="symbol,timestamp").execute()
    return True


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

    metrics["prediction_written"] = upsert_latest_prediction(
        client, symbol, valid, metrics.get("model_name", model.name)
    )

    long_signals = valid[valid["prediction"] == 1]
    short_signals = valid[valid["prediction"] == 0]

    merged = valid.merge(df[["timestamp", "target_1h"]], on="timestamp", how="left")

    long_pnl = merged.loc[merged["prediction"] == 1, "target_1h"].apply(lambda x: 1 if x == 1 else -1).tolist()
    short_pnl = merged.loc[merged["prediction"] == 0, "target_1h"].apply(lambda x: 1 if x == 0 else -1).tolist()
    all_pnl = long_pnl + short_pnl

    metrics["symbol"] = symbol
    metrics["total_signals"] = len(valid)
    metrics["long_signals"] = len(long_signals)
    metrics["short_signals"] = len(short_signals)
    metrics["ev"] = round(expected_value(all_pnl), 4) if all_pnl else 0.0
    metrics["sharpe"] = round(sharpe_ratio(all_pnl), 4) if all_pnl else 0.0
    metrics["win_rate"] = round(win_rate(all_pnl), 4) if all_pnl else 0.0

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

    long_signals = valid[valid["prediction"] == 1]
    short_signals = valid[valid["prediction"] == 0]

    merged = valid.merge(df[["timestamp", "target_1h"]], on="timestamp", how="left")

    long_pnl = merged.loc[merged["prediction"] == 1, "target_1h"].apply(lambda x: 1 if x == 1 else -1).tolist()
    short_pnl = merged.loc[merged["prediction"] == 0, "target_1h"].apply(lambda x: 1 if x == 0 else -1).tolist()
    all_pnl = long_pnl + short_pnl

    metrics["symbol"] = symbol
    metrics["total_signals"] = len(valid)
    metrics["long_signals"] = len(long_signals)
    metrics["short_signals"] = len(short_signals)
    metrics["ev"] = round(expected_value(all_pnl), 4) if all_pnl else 0.0
    metrics["sharpe"] = round(sharpe_ratio(all_pnl), 4) if all_pnl else 0.0
    metrics["win_rate"] = round(win_rate(all_pnl), 4) if all_pnl else 0.0

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

    if current_model == LOGISTIC_MODEL_NAME:
        logistic_recent = recent[recent["model_name"] == LOGISTIC_MODEL_NAME]
        if len(logistic_recent) >= 7 and (logistic_recent["sharpe_ratio"] > 1).all():
            return XGBOOST_MODEL_NAME

    return current_model


def main():
    print("=" * 60)
    print("Quant Bot - Model Training Pipeline")
    print("=" * 60)

    client = get_client()

    active_model_name = check_auto_upgrade(client)
    print(f"Active model: {active_model_name}")

    if active_model_name == XGBOOST_MODEL_NAME and not HAS_XGBOOST:
        print("WARNING: xgboost not installed, falling back to logistic")
        active_model_name = LOGISTIC_MODEL_NAME

    all_metrics = []
    for symbol in SYMBOLS:
        print(f"\nTraining on {symbol}...")
        try:
            if active_model_name == XGBOOST_MODEL_NAME:
                model = XGBoostModel()
            else:
                model = LogisticModel()

            metrics = train_walk_forward(client, symbol, model, n_splits=5)
            all_metrics.append(metrics)
            log_model_metrics(client, metrics)

            if "error" not in metrics:
                print(f"  Accuracy: {metrics.get('test_accuracy', 'N/A')}")
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