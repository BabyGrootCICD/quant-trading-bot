import sys
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import SYMBOLS
from src.data.supabase_client import get_client
from src.models.logistic import LogisticModel, FEATURE_COLS
from src.models.xgboost_model import XGBoostModel, HAS_XGBOOST
from src.utils.metrics import sharpe_ratio, expected_value, win_rate


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


def train_and_evaluate(client, symbol: str, model) -> dict:
    df = fetch_features(client, symbol)
    if df.empty or len(df) < 200:
        return {"symbol": symbol, "error": "insufficient data"}

    metrics = model.fit(df)
    if "error" in metrics:
        return {"symbol": symbol, **metrics}

    predictions = model.predict(df)
    valid = predictions.dropna(subset=["prediction"])
    if valid.empty:
        return {"symbol": symbol, **metrics, "trade_signals": 0}

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
        return "logistic_v1"

    df = pd.DataFrame(resp.data)
    df["evaluated_at"] = pd.to_datetime(df["evaluated_at"])

    now = datetime.now(timezone.utc)
    week_ago = now - pd.Timedelta(days=7)
    recent = df[df["evaluated_at"] >= week_ago]

    if recent.empty:
        return "logistic_v1"

    current_model = recent.iloc[0]["model_name"]

    if current_model == "logistic_v1":
        logistic_recent = recent[recent["model_name"] == "logistic_v1"]
        if len(logistic_recent) >= 7 and (logistic_recent["sharpe_ratio"] > 1).all():
            return "xgboost_v1"

    return current_model


def main():
    print("=" * 60)
    print("Quant Bot - Model Training Pipeline")
    print("=" * 60)

    client = get_client()

    active_model_name = check_auto_upgrade(client)
    print(f"Active model: {active_model_name}")

    if active_model_name == "xgboost_v1" and not HAS_XGBOOST:
        print("WARNING: xgboost not installed, falling back to logistic_v1")
        active_model_name = "logistic_v1"

    all_metrics = []
    for symbol in SYMBOLS:
        print(f"\nTraining on {symbol}...")
        try:
            if active_model_name == "xgboost_v1":
                model = XGBoostModel()
            else:
                model = LogisticModel()

            metrics = train_and_evaluate(client, symbol, model)
            all_metrics.append(metrics)
            log_model_metrics(client, metrics)

            if "error" not in metrics:
                print(f"  Accuracy: {metrics.get('test_accuracy', 'N/A')}")
                print(f"  EV: {metrics.get('ev', 'N/A')}")
                print(f"  Sharpe: {metrics.get('sharpe', 'N/A')}")
                print(f"  Win rate: {metrics.get('win_rate', 'N/A')}")
                print(f"  Signals: {metrics.get('total_signals', 0)}")

                df = fetch_features(client, symbol)
                if not df.empty:
                    latest = df.tail(1)
                    pred = model.predict(latest)
                    if not pred.empty and pred["prediction"].notna().any():
                        row = pred.iloc[0]
                        client.table("predictions").upsert({
                            "symbol": symbol,
                            "timestamp": int(row["timestamp"]),
                            "model_name": active_model_name,
                            "prediction": float(row["prediction"]),
                            "probability_up": float(row["probability_up"]),
                            "confidence": float(row["confidence"]),
                        }, on_conflict="symbol,timestamp").execute()
                        print(f"  Prediction: P(up)={row['probability_up']:.3f} -> {'LONG' if row['prediction'] == 1 else 'SHORT'}")
            else:
                print(f"  Skipped: {metrics.get('error')}")

        except Exception as e:
            print(f"  ERROR: {e}")

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


if __name__ == "__main__":
    main()
