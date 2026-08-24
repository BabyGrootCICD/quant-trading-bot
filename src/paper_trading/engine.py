import sys
import os
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import SYMBOLS, EXCHANGE_ID, STATS_EPOCH_MS
from src.data.supabase_client import get_client
from src.strategy.signals import generate_signals
from src.strategy.sizing import calculate_position_size
from src.strategy.economics import (
    is_tradeable, expected_value, breakeven_accuracy, round_trip_cost_pct,
)
from src.utils.metrics import sharpe_ratio, win_rate, trade_returns, annualization_factor

INITIAL_CASH = 10000.0
TAKER_FEE = 0.001
SLIPPAGE_BPS = 5

# A prediction older than this is not tradeable. Without this guard the engine
# happily replayed one frozen prediction set for a full day, paying the
# round-trip spread every hour on a signal carrying zero information.
MAX_PREDICTION_AGE_HOURS = 2

# Number of closed trades pulled for portfolio stats. The old 500 cap silently
# truncated total_pnl and total_trades once the bot passed 500 trades.
TRADE_HISTORY_LIMIT = 5000

# How much the expected edge must exceed the round-trip cost before a trade is
# worth taking. At a 1h horizon this blocks nearly everything -- see
# .claude/STRATEGY_PLAN.md Finding 1. Not trading beats paying 0.30% for a
# half-point edge.
MIN_EDGE_MARGIN = float(os.getenv("MIN_EDGE_MARGIN", "1.0"))

# Bars of recent history used to estimate each symbol's typical move.
VOLATILITY_LOOKBACK = 168


def create_exchange():
    exchange_class = getattr(ccxt, EXCHANGE_ID, None)
    if exchange_class is None:
        raise ValueError(f"Unknown exchange: {EXCHANGE_ID}")
    return exchange_class({"enableRateLimit": True})


def fetch_live_prices(exchange, symbols: list[str]) -> dict[str, float]:
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


def fetch_trade_history(client, limit: int = TRADE_HISTORY_LIMIT) -> pd.DataFrame:
    """Closed trades, newest first.

    `size` and `exit_time` are needed to express PnL as a return and to work
    out the real trade frequency for Sharpe annualization.
    """
    resp = (
        client.table("paper_trades")
        .select("pnl,status,actual_pnl_usd,size,fees,entry_time,exit_time")
        .neq("status", "open")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()


def fetch_portfolio_history(client, limit: int = 168) -> pd.DataFrame:
    resp = (
        client.table("portfolio")
        .select("equity,total_asset_usd,timestamp")
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )
    return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()


def estimate_expected_moves(exchange, symbols: list[str],
                            lookback: int = VOLATILITY_LOOKBACK) -> dict[str, float]:
    """Mean absolute hourly log return per symbol, from recent live candles.

    This replaces `estimated_change_pct = (probability_up - 0.5) * 2 * 0.02`,
    a fabricated linear map with a hardcoded 2% ceiling that had nothing to do
    with any symbol's actual volatility. The EV gate needs a real number.
    """
    moves = {}
    for symbol in symbols:
        try:
            candles = exchange.fetch_ohlcv(symbol, "1h", limit=lookback + 1)
            closes = np.array([c[4] for c in candles], dtype=float)
            if len(closes) < 2:
                continue
            rets = np.diff(np.log(closes))
            rets = rets[np.isfinite(rets)]
            if len(rets) == 0:
                continue
            moves[symbol] = float(np.mean(np.abs(rets)))
        except Exception as e:
            print(f"  Failed to estimate volatility for {symbol}: {e}")
    return moves


def round_trip_cost(size: float) -> float:
    """Fee + slippage for a full round trip (entry leg and exit leg).

    The old model charged a single leg, understating the real cost of the
    hourly churn by half.
    """
    per_leg = size * TAKER_FEE + size * (SLIPPAGE_BPS / 10000)
    return 2 * per_leg


def is_prediction_fresh(prediction_ts_ms: int, now_ms: int, max_age_hours: int = MAX_PREDICTION_AGE_HOURS) -> bool:
    """True when a prediction is recent enough to trade on."""
    age_hours = (now_ms - int(prediction_ts_ms)) / 3_600_000
    return 0 <= age_hours <= max_age_hours


def desired_sides(signals: pd.DataFrame) -> dict[str, str]:
    """Map symbol -> desired side for the current (fresh) signal set."""
    if signals is None or signals.empty:
        return {}
    out = {}
    for _, row in signals.iterrows():
        if row["signal"] == 1:
            out[row["symbol"]] = "long"
        elif row["signal"] == -1:
            out[row["symbol"]] = "short"
    return out


def desired_sides_from_trades(open_trades: pd.DataFrame) -> dict[str, str]:
    """Map symbol -> side for currently open positions."""
    if open_trades is None or open_trades.empty:
        return {}
    return {row["symbol"]: row["side"] for _, row in open_trades.iterrows()}


def close_open_positions(client, open_trades: pd.DataFrame, prices: dict[str, float], now_ms: int,
                         wanted: dict[str, str] | None = None):
    """Close positions the current signal no longer wants.

    `wanted` maps symbol -> side for the fresh signals. A position whose side
    still matches is held rather than closed-and-reopened: the old code closed
    every position every hour and paid the round trip again, which is what
    turned a zero-information signal into a steady loss.

    Passing `wanted=None` closes everything (used when no fresh signal exists).
    """
    if open_trades.empty:
        return []

    closed_symbols = []
    for _, trade in open_trades.iterrows():
        symbol = trade["symbol"]
        if symbol not in prices:
            continue

        if wanted is not None and wanted.get(symbol) == trade["side"]:
            print(f"  Holding {trade['side']} {symbol} (signal unchanged)")
            continue

        current_price = prices[symbol]
        entry_price = trade["entry_price"]
        size = trade["size"]
        side = trade["side"]

        if side == "long":
            raw_pnl = (current_price - entry_price) / entry_price * size
        else:
            raw_pnl = (entry_price - current_price) / entry_price * size

        # Both legs: the position was opened and is now being closed.
        fees = round_trip_cost(size)
        net_pnl = raw_pnl - fees
        actual_pnl_usd = net_pnl

        client.table("paper_trades").update({
            "exit_price": current_price,
            "exit_time": now_ms,
            "pnl": round(net_pnl, 4),
            "actual_pnl_usd": round(actual_pnl_usd, 4),
            "fees": round(fees, 4),
            "status": "closed",
        }).eq("id", trade["id"]).execute()

        closed_symbols.append(symbol)
        print(f"  Closed {side} {symbol}: entry={entry_price:.2f} exit={current_price:.2f} pnl=${actual_pnl_usd:.2f}")

    return closed_symbols


def open_new_positions(client, signals: pd.DataFrame, prices: dict[str, float], now_ms: int, model_name: str,
                       total_asset_usd: float, held: dict[str, str] | None = None,
                       expected_moves: dict[str, float] | None = None):
    held = held or {}
    expected_moves = expected_moves or {}
    if signals.empty:
        return

    for _, row in signals.iterrows():
        symbol = row["symbol"]
        if symbol not in prices:
            continue

        signal = row["signal"]
        if signal == 0:
            continue

        # Does this signal clear its own transaction cost?
        strength = float(row["signal_strength"])
        exp_move = expected_moves.get(symbol)
        if exp_move is None:
            print(f"  Skipped {symbol}: no volatility estimate")
            continue

        if not is_tradeable(strength, exp_move, margin=MIN_EDGE_MARGIN):
            ev = expected_value(strength, exp_move)
            need = breakeven_accuracy(exp_move)
            print(f"  Skipped {symbol}: EV {ev*100:+.3f}% "
                  f"(strength {strength:.2f}, E|move| {exp_move*100:.3f}%, "
                  f"needs {need*100:.1f}% accuracy)")
            continue

        side = "long" if signal == 1 else "short"
        if held.get(symbol) == side:
            # Already positioned this way; re-entering would just pay the
            # spread again for no change in exposure.
            print(f"  Skipped {symbol}: already {side}")
            continue

        current_price = prices[symbol]
        # Signed real expected move, not the fabricated (p-0.5)*2*0.02 map.
        estimated_change_pct = exp_move if side == "long" else -exp_move
        size = calculate_position_size(
            strength,
            method="percentage",
            estimated_change_pct=estimated_change_pct,
            total_asset_usd=total_asset_usd,
        )

        if size < 1.0:
            print(f"  Skipped {symbol}: size too small (${size:.2f})")
            continue

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

        print(f"  Opened {side} {symbol} @ {current_price:.2f} size=${size:.2f} est_change={estimated_change_pct:.4f}")


def filter_to_stats_epoch(trade_history: pd.DataFrame, epoch_ms: int = STATS_EPOCH_MS) -> pd.DataFrame:
    """Drop trades entered before the pipeline was fixed.

    The first 75 closed trades were made while the engine replayed a frozen
    prediction set every hour. Including them keeps sharpe_ratio pinned near
    -45 and win_rate near 0.35 no matter how the current system performs,
    because the Sharpe window is only 168 trades deep.
    """
    if trade_history.empty or "entry_time" not in trade_history.columns:
        return trade_history
    times = pd.to_numeric(trade_history["entry_time"], errors="coerce")
    return trade_history[times >= epoch_ms]


def compute_sharpe(trade_history: pd.DataFrame, window: int = 168) -> float:
    """Annualized Sharpe over the most recent `window` closed trades.

    Three bugs used to live here:
      * dollar PnL was fed straight in, so the ratio scaled with position size;
      * it was annualized at sqrt(8760) as if one trade happened per hour,
        while the bot opens one per symbol per hour;
      * `closed_pnl[-168:]` sliced the *oldest* rows, because the query
        returns newest-first.
    """
    if trade_history.empty or "pnl" not in trade_history.columns:
        return 0.0
    if "size" not in trade_history.columns:
        return 0.0

    hist = trade_history.dropna(subset=["pnl", "size"])
    if hist.empty:
        return 0.0

    # Newest-first from the query, so the newest `window` rows are the head.
    recent = hist.head(window)

    rets = trade_returns(recent["pnl"].tolist(), recent["size"].tolist())
    if len(rets) < 2:
        return 0.0

    span_hours = 0.0
    if "exit_time" in recent.columns:
        times = pd.to_numeric(recent["exit_time"], errors="coerce").dropna()
        if len(times) >= 2:
            span_hours = (times.max() - times.min()) / 3_600_000

    ppy = annualization_factor(len(rets), span_hours)
    return sharpe_ratio(rets, periods_per_year=ppy)


def update_portfolio(client, cash: float, open_trades: pd.DataFrame, prices: dict[str, float],
                     trade_history: pd.DataFrame, now_ms: int):
    # Calculate REAL cash balance
    realized_pnl = 0.0
    realized_fees = 0.0
    if not trade_history.empty and "pnl" in trade_history.columns:
        realized_pnl = trade_history["pnl"].sum()
        realized_fees = trade_history["fees"].sum() if "fees" in trade_history.columns else 0.0

    # Cash = initial - capital locked in open positions + realized PnL - fees
    capital_locked = sum(trade["size"] for _, trade in open_trades.iterrows()) if not open_trades.empty else 0.0
    cash_balance = cash - capital_locked + realized_pnl - realized_fees

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

    equity = cash_balance + positions_value
    total_asset_usd = equity

    # Statistics describe the current system, not the frozen-signal era.
    stats_history = filter_to_stats_epoch(trade_history)

    closed_pnl = []
    closed_sizes = []
    if not stats_history.empty and "pnl" in stats_history.columns:
        hist = stats_history.dropna(subset=["pnl"])
        closed_pnl = hist["pnl"].tolist()
        closed_sizes = hist["size"].tolist() if "size" in hist.columns else []

    total_pnl = sum(closed_pnl) if closed_pnl else 0.0
    wr = win_rate(closed_pnl)
    total_trades = len(closed_pnl)
    sr = compute_sharpe(stats_history)

    client.table("portfolio").insert({
        "timestamp": now_ms,
        "equity": round(equity, 2),
        "cash": round(cash_balance, 2),
        "positions_value": round(positions_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_asset_usd": round(total_asset_usd, 2),
        "sharpe_ratio": round(sr, 4),
        "win_rate": round(wr, 4),
        "total_trades": total_trades,
    }).execute()

    return {"equity": equity, "cash": cash_balance, "sharpe": sr, "win_rate": wr, "total_pnl": total_pnl, "total_trades": total_trades, "total_asset_usd": total_asset_usd}


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

    print(f"  Round-trip cost: {round_trip_cost_pct()*100:.2f}% of position")

    print("\n[1b/6] Estimating expected moves...")
    expected_moves = estimate_expected_moves(exchange, SYMBOLS)
    for sym, mv in sorted(expected_moves.items()):
        need = breakeven_accuracy(mv)
        note = "  UNREACHABLE" if need > 1 else ""
        print(f"    {sym}: E|move|={mv*100:.3f}% -> needs {need*100:.1f}% accuracy{note}")

    print("\n[2/6] Fetching latest predictions...")
    predictions = fetch_latest_predictions(client, SYMBOLS)
    if predictions.empty:
        print("  No predictions found. Run trainer first.")
        return

    model_name = predictions["model_name"].iloc[0] if "model_name" in predictions else "unknown"
    print(f"  Using model: {model_name}")

    # Staleness guard. Trading a prediction the trainer wrote hours ago is not
    # trading a model, it is paying the spread on a constant.
    fresh_mask = predictions["timestamp"].apply(lambda ts: is_prediction_fresh(ts, now_ms))
    stale = predictions[~fresh_mask]
    for _, row in stale.iterrows():
        age_h = (now_ms - int(row["timestamp"])) / 3_600_000
        print(f"  STALE: {row['symbol']} prediction is {age_h:.1f}h old (max {MAX_PREDICTION_AGE_HOURS}h) - ignoring")
    predictions = predictions[fresh_mask]

    if predictions.empty:
        print(f"  No predictions fresher than {MAX_PREDICTION_AGE_HOURS}h. "
              "Closing out and skipping new entries -- check the training step.")

    print("\n[3/6] Generating signals...")
    if predictions.empty:
        signals = pd.DataFrame()
        active_signals = pd.DataFrame()
    else:
        signals = generate_signals(predictions)
        active_signals = signals[signals["signal"] != 0]
    print(f"  Active signals: {len(active_signals)}")
    if not active_signals.empty:
        for _, s in active_signals.iterrows():
            side = "BUY" if s["signal"] == 1 else "SELL"
            est = s.get("estimated_change_pct", 0.0)
            print(f"    {s['symbol']}: {side} (P(up)={s['probability_up']:.2f}, est_change={est:.4f}, strength={s['signal_strength']:.2f})")

    print("\n[4/6] Reconciling open positions...")
    open_trades = fetch_open_trades(client)
    print(f"  Open positions: {len(open_trades)}")
    # With no fresh signal at all, wanted={} closes everything rather than
    # holding exposure the model can no longer justify.
    wanted = desired_sides(active_signals)
    close_open_positions(client, open_trades, prices, now_ms, wanted=wanted)

    print("\n[5/6] Opening new positions...")
    open_trades_after = fetch_open_trades(client)
    held = desired_sides_from_trades(open_trades_after)
    trade_history = fetch_trade_history(client)
    total_asset_usd = INITIAL_CASH
    if not trade_history.empty and "actual_pnl_usd" in trade_history.columns:
        total_asset_usd += trade_history["actual_pnl_usd"].sum()
    if active_signals.empty:
        print("  No fresh signals; not opening anything.")
    else:
        open_new_positions(client, active_signals, prices, now_ms, model_name, total_asset_usd,
                           held=held, expected_moves=expected_moves)

    print("\n[6/6] Updating portfolio...")
    open_trades_final = fetch_open_trades(client)
    portfolio = update_portfolio(client, INITIAL_CASH, open_trades_final, prices, trade_history, now_ms)
    print(f"  Total Asset: ${portfolio['total_asset_usd']:.2f}")
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