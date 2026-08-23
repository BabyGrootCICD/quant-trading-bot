import sys
import os
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import SYMBOLS
from src.data.supabase_client import get_client
from src.strategy.signals import generate_signals
from src.strategy.sizing import calculate_position_size
from src.utils.metrics import sharpe_ratio, win_rate

INITIAL_CASH = 10000.0
TAKER_FEE = 0.001
SLIPPAGE_BPS = 5


def create_exchange() -> ccxt.binance:
    return ccxt.binance({"enableRateLimit": True})


def fetch_live_prices(exchange: ccxt.binance, symbols: list[str]) -> dict[str, float]:
    prices = {}
    for symbol in symbols:
        try:
            ticker = exchange.fetch_ticker(symbol)
            prices[symbol] = ticker["last"]
        except Exception as e:
            print(f"  Failed to fetch price for {symbol}: {e}")
    return prices


def fetch_latest_predictions(client, symbols: list[str]) -> pd.DataFrame:
    all_preds = []
    for symbol in symbols:
        resp = (
            client.table("predictions")
            .select("symbol,timestamp,probability_up,confidence,model_name")
            .eq("symbol", symbol)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        if resp.data:
            all_preds.append(resp.data[0])
    return pd.DataFrame(all_preds) if all_preds else pd.DataFrame()


def fetch_open_trades(client) -> pd.DataFrame:
    resp = (
        client.table("paper_trades")
        .select("*")
        .eq("status", "open")
        .execute()
    )
    return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()


def fetch_trade_history(client, limit: int = 500) -> pd.DataFrame:
    resp = (
        client.table("paper_trades")
        .select("pnl,status")
        .neq("status", "open")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()


def fetch_portfolio_history(client, limit: int = 168) -> pd.DataFrame:
    resp = (
        client.table("portfolio")
        .select("equity,timestamp")
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )
    return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()


def close_open_positions(client, open_trades: pd.DataFrame, prices: dict[str, float], now_ms: int):
    if open_trades.empty:
        return

    for _, trade in open_trades.iterrows():
        symbol = trade["symbol"]
        if symbol not in prices:
            continue

        current_price = prices[symbol]
        entry_price = trade["entry_price"]
        size = trade["size"]
        side = trade["side"]

        if side == "long":
            raw_pnl = (current_price - entry_price) / entry_price * size
        else:
            raw_pnl = (entry_price - current_price) / entry_price * size

        fees = size * TAKER_FEE + size * (SLIPPAGE_BPS / 10000)
        net_pnl = raw_pnl - fees

        client.table("paper_trades").update({
            "exit_price": current_price,
            "exit_time": now_ms,
            "pnl": round(net_pnl, 4),
            "fees": round(fees, 4),
            "status": "closed",
        }).eq("id", trade["id"]).execute()

        print(f"  Closed {side} {symbol}: entry={entry_price:.2f} exit={current_price:.2f} pnl={net_pnl:.2f}")


def open_new_positions(client, signals: pd.DataFrame, prices: dict[str, float], now_ms: int, model_name: str):
    if signals.empty:
        return

    for _, row in signals.iterrows():
        symbol = row["symbol"]
        if symbol not in prices:
            continue

        signal = row["signal"]
        if signal == 0:
            continue

        side = "long" if signal == 1 else "short"
        current_price = prices[symbol]
        size = calculate_position_size(row["signal_strength"], method="tanh")

        client.table("paper_trades").insert({
            "symbol": symbol,
            "side": side,
            "entry_price": current_price,
            "size": round(size, 2),
            "entry_time": now_ms,
            "model_name": model_name,
            "prediction_at_entry": round(float(row["probability_up"]), 4),
            "status": "open",
        }).execute()

        print(f"  Opened {side} {symbol} @ {current_price:.2f} size=${size:.2f} confidence={row['signal_strength']:.2f}")


def update_portfolio(client, cash: float, open_trades: pd.DataFrame, prices: dict[str, float],
                     trade_history: pd.DataFrame, now_ms: int):
    positions_value = 0.0
    if not open_trades.empty:
        for _, trade in open_trades.iterrows():
            symbol = trade["symbol"]
            if symbol not in prices:
                continue
            current_price = prices[symbol]
            entry_price = trade["entry_price"]
            size = trade["size"]
            if trade["side"] == "long":
                pnl = (current_price - entry_price) / entry_price * size
            else:
                pnl = (entry_price - current_price) / entry_price * size
            positions_value += size + pnl

    equity = cash + positions_value

    closed_pnl = []
    if not trade_history.empty and "pnl" in trade_history.columns:
        closed_pnl = trade_history["pnl"].dropna().tolist()

    total_pnl = sum(closed_pnl) if closed_pnl else 0.0
    wr = win_rate(closed_pnl)
    total_trades = len(closed_pnl)
    sr = sharpe_ratio(closed_pnl[-168:]) if len(closed_pnl) >= 2 else 0.0

    client.table("portfolio").insert({
        "timestamp": now_ms,
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "positions_value": round(positions_value, 2),
        "total_pnl": round(total_pnl, 2),
        "sharpe_ratio": round(sr, 4),
        "win_rate": round(wr, 4),
        "total_trades": total_trades,
    }).execute()

    return {"equity": equity, "cash": cash, "sharpe": sr, "win_rate": wr, "total_pnl": total_pnl, "total_trades": total_trades}


def log_predictions(client, signals: pd.DataFrame, model_name: str):
    if signals.empty:
        return
    records = []
    for _, row in signals.iterrows():
        records.append({
            "symbol": row["symbol"],
            "timestamp": int(row["timestamp"]),
            "model_name": model_name,
            "prediction": float(row["prediction"]),
            "probability_up": float(row["probability_up"]),
            "confidence": float(row["confidence"]),
        })
    if records:
        client.table("predictions").upsert(records, on_conflict="symbol,timestamp").execute()


def main():
    print("=" * 60)
    print("Quant Bot - Paper Trading Engine")
    print("=" * 60)

    client = get_client()
    exchange = create_exchange()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    print("\n[1/6] Fetching live prices...")
    prices = fetch_live_prices(exchange, SYMBOLS)
    print(f"  Got prices for {len(prices)}/{len(SYMBOLS)} symbols")

    print("\n[2/6] Fetching latest predictions...")
    predictions = fetch_latest_predictions(client, SYMBOLS)
    if predictions.empty:
        print("  No predictions found. Run trainer first.")
        return

    model_name = predictions["model_name"].iloc[0] if "model_name" in predictions else "unknown"
    print(f"  Using model: {model_name}")

    print("\n[3/6] Generating signals...")
    signals = generate_signals(predictions)
    active_signals = signals[signals["signal"] != 0]
    print(f"  Active signals: {len(active_signals)}")
    if not active_signals.empty:
        for _, s in active_signals.iterrows():
            side = "LONG" if s["signal"] == 1 else "SHORT"
            print(f"    {s['symbol']}: {side} (P(up)={s['probability_up']:.2f}, strength={s['signal_strength']:.2f})")

    print("\n[4/6] Closing open positions...")
    open_trades = fetch_open_trades(client)
    print(f"  Open positions: {len(open_trades)}")
    close_open_positions(client, open_trades, prices, now_ms)

    print("\n[5/6] Opening new positions...")
    open_new_positions(client, active_signals, prices, now_ms, model_name)

    print("\n[6/6] Updating portfolio...")
    open_trades_after = fetch_open_trades(client)
    trade_history = fetch_trade_history(client)
    portfolio = update_portfolio(client, INITIAL_CASH, open_trades_after, prices, trade_history, now_ms)
    print(f"  Equity: ${portfolio['equity']:.2f}")
    print(f"  Total P&L: ${portfolio['total_pnl']:.2f}")
    print(f"  Sharpe: {portfolio['sharpe']:.4f}")
    print(f"  Win Rate: {portfolio['win_rate']:.2%}")
    print(f"  Total Trades: {portfolio['total_trades']}")

    print("\n" + "=" * 60)
    print("Paper trading cycle complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
