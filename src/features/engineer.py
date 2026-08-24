import sys
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import SYMBOLS
from src.data.supabase_client import get_client


def fetch_candles(client, symbol: str) -> pd.DataFrame:
    all_data = []
    batch_size = 1000
    offset = 0

    while True:
        resp = (
            client.table("candles")
            .select("timestamp,open,high,low,close,volume")
            .eq("symbol", symbol)
            .order("timestamp", desc=False)
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        if not resp.data:
            break
        all_data.extend(resp.data)
        if len(resp.data) < batch_size:
            break
        offset += batch_size

    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
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


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d


def compute_heikin_ashi(close: pd.Series) -> pd.Series:
    ha_close = (close + close.shift(1) + close.shift(2) + close.shift(3)) / 4
    ha_open = close.shift(1)
    ha_high = close.rolling(4).max()
    ha_low = close.rolling(4).min()
    return (ha_close + ha_open + ha_high + ha_low) / 4


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 50:
        return pd.DataFrame()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    features = pd.DataFrame()
    if "symbol" in df.columns:
        features["symbol"] = df["symbol"].values
    else:
        features["symbol"] = "UNKNOWN"
    features["timestamp"] = df["timestamp"]

    features["log_return_1h"] = compute_log_returns(close, 1)
    features["log_return_2h"] = compute_log_returns(close, 2)
    features["log_return_4h"] = compute_log_returns(close, 4)
    features["log_return_8h"] = compute_log_returns(close, 8)
    features["log_return_24h"] = compute_log_returns(close, 24)

    features["rsi_14"] = compute_rsi(close, 14)
    features["rsi_7"] = compute_rsi(close, 7)

    macd_line, signal_line = compute_macd(close)
    features["macd"] = macd_line
    features["macd_signal"] = signal_line
    features["macd_hist"] = macd_line - signal_line

    bb_upper, bb_lower = compute_bollinger(close)
    features["bb_upper"] = bb_upper
    features["bb_lower"] = bb_lower
    features["bb_width"] = (bb_upper - bb_lower) / close
    features["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)

    features["volume_ratio"] = compute_volume_ratio(volume)
    features["volume_ratio_48"] = compute_volume_ratio(volume, 48)

    features["atr_14"] = compute_atr(high, low, close)
    features["atr_14_pct"] = features["atr_14"] / close

    features["williams_r"] = compute_williams_r(high, low, close)
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    features["stoch_k"] = stoch_k
    features["stoch_d"] = stoch_d

    features["ha_trend"] = compute_heikin_ashi(close)

    features["close_pct_ma20"] = close / close.rolling(20).mean() - 1
    features["close_pct_ma50"] = close / close.rolling(50).mean() - 1
    features["vol_20"] = close.pct_change().rolling(20).std()
    features["skew_20"] = close.pct_change().rolling(20).skew()

    # Bars with exactly zero return carry no direction. On thin, tick-quantized
    # pairs they are common -- 29.6% of TRX/USDT hourly bars -- and labelling
    # them "down" skews the class balance to 0.36 up, so always predicting
    # "down" scores 64% and looks like skill. Leave them unlabelled (NaN) so
    # training drops them instead.
    next_ret_1h = features["log_return_1h"].shift(-1)
    features["target_1h"] = np.where(
        next_ret_1h > 0, 1.0, np.where(next_ret_1h < 0, 0.0, np.nan)
    )

    next_ret_4h = features["log_return_4h"].shift(-4)
    features["target_4h"] = np.where(
        next_ret_4h > 0, 1.0, np.where(next_ret_4h < 0, 0.0, np.nan)
    )

    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def upsert_features(client, symbol: str, features: pd.DataFrame) -> int:
    if features.empty:
        return 0

    records = features.to_dict(orient="records")
    clean_records = []
    for r in records:
        # Send NULL explicitly rather than omitting the key. On an UPSERT an
        # omitted key leaves the existing column value in place, so newly
        # unlabelled flat bars kept their old target_1h = 0 and TRX's class
        # balance never actually changed. Explicit keys also keep every record
        # in a batch structurally identical, which PostgREST requires.
        rec = {k: _clean_value(v) for k, v in r.items() if k != "id"}
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
    print(f"  {symbol}: {count} feature rows ({len(features.columns)} features)")
    return count


def main():
    print("=" * 60)
    print("Quant Bot - Feature Engineering (Enhanced)")
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