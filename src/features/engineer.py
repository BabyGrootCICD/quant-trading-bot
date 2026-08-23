import sys
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import SYMBOLS
from src.data.supabase_client import get_client


def fetch_candles(client, symbol: str) -> pd.DataFrame:
    resp = (
        client.table("candles")
        .select("timestamp,open,high,low,close,volume")
        .eq("symbol", symbol)
        .order("timestamp", desc=False)
        .execute()
    )
    if not resp.data:
        return pd.DataFrame()
    df = pd.DataFrame(resp.data)
    df["timestamp"] = df["timestamp"].astype(int)
    return df


def compute_log_returns(close: pd.Series, lag: int) -> pd.Series:
    return np.log(close / close.shift(lag))


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger(close: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series]:
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower


def compute_volume_ratio(volume: pd.Series, period: int = 24) -> pd.Series:
    avg_vol = volume.rolling(window=period).mean()
    return volume / avg_vol.replace(0, np.nan)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 30:
        return pd.DataFrame()

    close = df["close"]
    volume = df["volume"]

    features = pd.DataFrame()
    features["symbol"] = df["symbol"].iloc[0] if "symbol" in df.columns else "UNKNOWN"
    features["timestamp"] = df["timestamp"]

    features["log_return_1h"] = compute_log_returns(close, 1)
    features["log_return_2h"] = compute_log_returns(close, 2)
    features["log_return_4h"] = compute_log_returns(close, 4)
    features["log_return_8h"] = compute_log_returns(close, 8)
    features["log_return_24h"] = compute_log_returns(close, 24)

    features["rsi_14"] = compute_rsi(close, 14)

    macd_line, signal_line = compute_macd(close)
    features["macd"] = macd_line
    features["macd_signal"] = signal_line

    bb_upper, bb_lower = compute_bollinger(close)
    features["bb_upper"] = bb_upper
    features["bb_lower"] = bb_lower

    features["volume_ratio"] = compute_volume_ratio(volume)

    features["target_1h"] = (features["log_return_1h"].shift(-1) > 0).astype(int)

    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def upsert_features(client, symbol: str, features: pd.DataFrame) -> int:
    if features.empty:
        return 0

    records = features.where(features.notna(), None).to_dict(orient="records")
    clean_records = []
    for r in records:
        rec = {k: v for k, v in r.items() if v is not None and k != "id"}
        clean_records.append(rec)

    if not clean_records:
        return 0

    batch_size = 500
    total = 0
    for i in range(0, len(clean_records), batch_size):
        batch = clean_records[i : i + batch_size]
        client.table("features").upsert(batch, on_conflict="symbol,timestamp").execute()
        total += len(batch)
    return total


def process_symbol(client, symbol: str) -> int:
    df = fetch_candles(client, symbol)
    if df.empty:
        print(f"  No candles for {symbol}")
        return 0
    df["symbol"] = symbol
    features = engineer_features(df)
    count = upsert_features(client, symbol, features)
    print(f"  {symbol}: {count} feature rows")
    return count


def main():
    print("=" * 60)
    print("Quant Bot - Feature Engineering")
    print("=" * 60)
    client = get_client()

    total = 0
    for symbol in SYMBOLS:
        try:
            count = process_symbol(client, symbol)
            total += count
        except Exception as e:
            print(f"  ERROR processing {symbol}: {e}")

    print("=" * 60)
    print(f"Done. Total feature rows: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
