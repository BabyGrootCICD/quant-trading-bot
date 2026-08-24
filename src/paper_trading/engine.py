import sys
import os
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import SYMBOLS, EXCHANGE_ID, STATS_EPOCH_MS, env_float
from src.data.supabase_client import get_client
from src.strategy.signals import generate_signals
from src.strategy.allocation import Candidate, allocate
from src.strategy.economics import (
    is_tradeable, expected_value, breakeven_accuracy, round_trip_cost_pct,
    expected_return, predicted_price, TAKER_FEE, SLIPPAGE_BPS,
)
from src.utils.metrics import (
    sharpe_ratio, win_rate, trade_returns, annualization_factor, MIN_SHARPE_TRADES,
)

INITIAL_CASH = 10000.0

# TAKER_FEE / SLIPPAGE_BPS are imported from src.strategy.economics so the cost
# the gate reasons about and the cost actually charged on a close cannot drift
# apart. Duplicating them here once let the gate allow trades at a price the
# engine did not charge.

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
MIN_EDGE_MARGIN = env_float("MIN_EDGE_MARGIN", 1.0)

# Bars of recent history used to estimate each symbol's typical move.
# This is only the *fallback* now: it produces one unconditional number per
# symbol, which makes the EV gate a constant. The live path prefers the
# magnitude head's per-bar forecast, carried on the prediction row.
VOLATILITY_LOOKBACK = 168

# --- exit policy -----------------------------------------------------------
# There was none. A position was closed only when the signal flipped, so a
# position whose signal never changed was held forever: rows 25-27 of the
# portfolio table are byte-identical, equity frozen at 9976.35 with the same
# two legacy positions and the same -11.97 of unrealized loss, hour after
# hour. The EV that justified entering was computed for a one-hour horizon;
# holding for twenty is a different bet that nothing ever evaluated.
MAX_HOLDING_HOURS = env_float("MAX_HOLDING_HOURS", 6)
STOP_LOSS_PCT = env_float("STOP_LOSS_PCT", 0.015)
TAKE_PROFIT_PCT = env_float("TAKE_PROFIT_PCT", 0.02)


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
            .select("symbol,timestamp,probability_up,confidence,model_name,"
                    "expected_move_pct,expected_return_pct,predicted_price")
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
    """Fee + slippage in dollars for a full round trip (entry and exit legs).

    The old model charged a single leg, understating the real cost of the
    hourly churn by half. Uses the same rate as the EV gate.
    """
    return size * round_trip_cost_pct()


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


def unrealized_return(side: str, entry_price: float, current_price: float) -> float:
    """Fractional PnL on a position at the current price, before costs."""
    entry_price = float(entry_price)
    if entry_price <= 0:
        return 0.0
    move = (float(current_price) - entry_price) / entry_price
    return move if side == "long" else -move


def position_age_hours(trade, now_ms: int) -> float | None:
    """Hours since entry, or None when the row carries no usable entry_time."""
    entry = trade.get("entry_time") if hasattr(trade, "get") else None
    if entry is None:
        return None
    try:
        entry = float(entry)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(entry):
        return None
    return (now_ms - entry) / 3_600_000


def exit_reason(trade, current_price: float, now_ms: int,
                wanted: dict[str, str] | None = None,
                max_holding_hours: float = MAX_HOLDING_HOURS,
                stop_loss_pct: float = STOP_LOSS_PCT,
                take_profit_pct: float = TAKE_PROFIT_PCT) -> str | None:
    """Why this position should be closed now, or None to keep holding.

    Risk exits are checked before the signal, because "the model still likes
    it" is not a reason to sit through an unbounded drawdown -- and because a
    signal that never changes is exactly the state that froze the book.
    """
    side = trade["side"]
    ret = unrealized_return(side, trade["entry_price"], current_price)

    if stop_loss_pct > 0 and ret <= -stop_loss_pct:
        return "stop_loss"
    if take_profit_pct > 0 and ret >= take_profit_pct:
        return "take_profit"

    age = position_age_hours(trade, now_ms)
    if max_holding_hours > 0 and age is not None and age >= max_holding_hours:
        return "max_holding"

    if wanted is None:
        return "no_signal"
    desired = wanted.get(trade["symbol"])
    if desired is None:
        return "no_signal"
    if desired != side:
        return "signal_flip"

    return None


def close_open_positions(client, open_trades: pd.DataFrame, prices: dict[str, float], now_ms: int,
                         wanted: dict[str, str] | None = None):
    """Close positions the exit policy no longer wants held.

    `wanted` maps symbol -> side for the fresh signals. A position whose side
    still matches is held rather than closed-and-reopened: the old code closed
    every position every hour and paid the round trip again, which is what
    turned a zero-information signal into a steady loss.

    Passing `wanted=None` closes everything (used when no fresh signal exists).

    Beyond the signal, `exit_reason()` now also applies a stop, a target and a
    maximum holding time, so a position can no longer outlive the one-hour
    forecast that justified opening it.
    """
    if open_trades.empty:
        return []

    closed_symbols = []
    for _, trade in open_trades.iterrows():
        symbol = trade["symbol"]
        if symbol not in prices:
            continue

        current_price = prices[symbol]
        reason = exit_reason(trade, current_price, now_ms, wanted=wanted)
        if reason is None:
            age = position_age_hours(trade, now_ms)
            age_note = f", {age:.1f}h old" if age is not None else ""
            print(f"  Holding {trade['side']} {symbol} (signal unchanged{age_note})")
            continue

        entry_price = trade["entry_price"]
        size = trade["size"]
        side = trade["side"]

        raw_pnl = unrealized_return(side, entry_price, current_price) * size

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
            "exit_reason": reason,
        }).eq("id", trade["id"]).execute()

        closed_symbols.append(symbol)
        print(f"  Closed {side} {symbol} [{reason}]: entry={entry_price:.2f} "
              f"exit={current_price:.2f} pnl=${actual_pnl_usd:.2f}")

    return closed_symbols


def resolve_expected_move(row, expected_moves: dict[str, float] | None) -> float | None:
    """Conditional E|move| for this bar, falling back to the symbol average.

    Preference order matters. `expected_move_pct` on the prediction row is the
    magnitude head's forecast *for this bar*; the dict is the unconditional
    mean over the last week of candles. Using the dict alone is what made the
    EV gate a constant and stopped the bot trading entirely.
    """
    move = row.get("expected_move_pct") if hasattr(row, "get") else None
    if move is not None:
        try:
            move = float(move)
        except (TypeError, ValueError):
            move = None
        if move is not None and np.isfinite(move) and move > 0:
            return move

    fallback = (expected_moves or {}).get(row["symbol"])
    if fallback is not None and np.isfinite(fallback) and fallback > 0:
        return float(fallback)
    return None


def build_candidates(signals: pd.DataFrame, prices: dict[str, float],
                     held: dict[str, str] | None = None,
                     expected_moves: dict[str, float] | None = None,
                     margin: float = MIN_EDGE_MARGIN) -> list[Candidate]:
    """Signals that survive the EV gate, priced in expected-return terms.

    Separating selection from sizing is the point: the gate decides *whether*
    a bet is worth making, the allocator decides *how much* of the book it
    gets, and neither can quietly override the other.
    """
    held = held or {}
    candidates = []

    if signals is None or signals.empty:
        return candidates

    for _, row in signals.iterrows():
        symbol = row["symbol"]
        if symbol not in prices:
            continue
        signal = row["signal"]
        if signal == 0:
            continue

        side = "long" if signal == 1 else "short"
        if held.get(symbol) == side:
            # Already positioned this way; re-entering would just pay the
            # spread again for no change in exposure.
            print(f"  Skipped {symbol}: already {side}")
            continue

        strength = float(row["signal_strength"])
        exp_move = resolve_expected_move(row, expected_moves)
        if exp_move is None:
            print(f"  Skipped {symbol}: no expected-move forecast")
            continue

        if not is_tradeable(strength, exp_move, margin=margin):
            ev = expected_value(strength, exp_move)
            need = breakeven_accuracy(exp_move)
            print(f"  Skipped {symbol}: EV {ev*100:+.3f}% "
                  f"(strength {strength:.2f}, E|move| {exp_move*100:.3f}%, "
                  f"needs {need*100:.1f}% accuracy)")
            continue

        candidates.append(Candidate(
            symbol=symbol,
            side=side,
            ev=expected_value(strength, exp_move),
            expected_abs_move=exp_move,
            probability_up=float(row["probability_up"]),
        ))

    return candidates


def open_new_positions(client, signals: pd.DataFrame, prices: dict[str, float], now_ms: int,
                       model_name: str, total_asset_usd: float, held: dict[str, str] | None = None,
                       expected_moves: dict[str, float] | None = None,
                       available_cash: float | None = None,
                       existing_exposure: float = 0.0):
    """Gate on expected value, then allocate capital across what survives.

    Sizing used to be `risk_budget / |estimated_change|` capped at $100, per
    symbol, in isolation: inversely proportional to the predicted move, blind
    to the other candidates, and blind to how much cash the portfolio actually
    had. It is now fractional-Kelly on the EV and variance the model produced,
    scaled to fit the cash and gross-exposure limits, so the money goes where
    the edge per unit of risk is largest and stops when the budget is gone.
    """
    candidates = build_candidates(signals, prices, held=held, expected_moves=expected_moves)
    if not candidates:
        return []

    cash = total_asset_usd if available_cash is None else available_cash
    sizes = allocate(candidates, equity=total_asset_usd, available_cash=cash,
                     existing_exposure=existing_exposure)

    opened = []
    for c in candidates:
        size = sizes.get(c.symbol)
        if not size:
            print(f"  Skipped {c.symbol}: no capital allocated "
                  f"(EV {c.ev*100:+.3f}%, budget exhausted or below minimum)")
            continue

        current_price = prices[c.symbol]
        forecast = predicted_price(current_price, c.probability_up, c.expected_abs_move)

        client.table("paper_trades").insert({
            "symbol": c.symbol,
            "side": c.side,
            "entry_price": current_price,
            "size": round(size, 2),
            "entry_time": now_ms,
            "model_name": model_name,
            "prediction_at_entry": round(c.probability_up, 4),
            "status": "open",
        }).execute()

        opened.append(c.symbol)
        print(f"  Opened {c.side} {c.symbol} @ {current_price:.2f} size=${size:.2f} "
              f"EV={c.ev*100:+.3f}% E|move|={c.expected_abs_move*100:.3f}% "
              f"P(up)={c.probability_up:.3f} -> forecast {forecast:.2f}")

    return opened


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


def compute_sharpe(trade_history: pd.DataFrame, window: int = 168) -> float | None:
    """Annualized Sharpe over the most recent `window` closed trades.

    Three bugs used to live here:
      * dollar PnL was fed straight in, so the ratio scaled with position size;
      * it was annualized at sqrt(8760) as if one trade happened per hour,
        while the bot opens one per symbol per hour;
      * `closed_pnl[-168:]` sliced the *oldest* rows, because the query
        returns newest-first.
    """
    if trade_history.empty or "pnl" not in trade_history.columns:
        return None
    if "size" not in trade_history.columns:
        return None

    hist = trade_history.dropna(subset=["pnl", "size"])
    if hist.empty:
        return None

    # Newest-first from the query, so the newest `window` rows are the head.
    recent = hist.head(window)

    rets = trade_returns(recent["pnl"].tolist(), recent["size"].tolist())
    if len(rets) < MIN_SHARPE_TRADES:
        # Not enough closed trades to say anything. Reporting 0.0 here would be
        # a claim -- "zero risk-adjusted return" -- rather than "unknown".
        return None

    span_hours = 0.0
    if "exit_time" in recent.columns:
        times = pd.to_numeric(recent["exit_time"], errors="coerce").dropna()
        if len(times) >= 2:
            span_hours = (times.max() - times.min()) / 3_600_000

    ppy = annualization_factor(len(rets), span_hours)
    sr = sharpe_ratio(rets, periods_per_year=ppy)
    return sr


def update_portfolio(client, cash: float, open_trades: pd.DataFrame, prices: dict[str, float],
                     trade_history: pd.DataFrame, now_ms: int):
    # `pnl` is already net of the round trip -- close_open_positions() writes
    # `net_pnl = raw_pnl - fees` into it and `fees` into the sibling column.
    # Subtracting `fees` again here charged every closed trade twice and
    # understated cash by the full fee bill: at the time of the screenshot,
    # total_pnl was -11.97 while cash sat at 9876.29 against 100 of locked
    # capital, i.e. $11.74 of phantom cost. `realized_fees` is kept only for
    # reporting.
    realized_pnl = 0.0
    realized_fees = 0.0
    if not trade_history.empty and "pnl" in trade_history.columns:
        realized_pnl = trade_history["pnl"].sum()
        realized_fees = trade_history["fees"].sum() if "fees" in trade_history.columns else 0.0

    # Cash = initial - capital locked in open positions + realized PnL (net).
    capital_locked = sum(trade["size"] for _, trade in open_trades.iterrows()) if not open_trades.empty else 0.0
    cash_balance = cash - capital_locked + realized_pnl

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

    # total_pnl and total_trades are accounting facts and stay lifetime, so they
    # reconcile with equity. sharpe_ratio and win_rate are quality measures of
    # the *strategy*, and the 75 frozen-signal trades are not this strategy --
    # those reset at the epoch.
    lifetime_pnl = []
    if not trade_history.empty and "pnl" in trade_history.columns:
        lifetime_pnl = trade_history["pnl"].dropna().tolist()

    total_pnl = sum(lifetime_pnl) if lifetime_pnl else 0.0
    total_trades = len(lifetime_pnl)

    stats_history = filter_to_stats_epoch(trade_history)
    closed_pnl = []
    if not stats_history.empty and "pnl" in stats_history.columns:
        closed_pnl = stats_history["pnl"].dropna().tolist()

    scored_trades = len(closed_pnl)
    wr = win_rate(closed_pnl) if scored_trades else None
    sr = compute_sharpe(stats_history)

    client.table("portfolio").insert({
        "timestamp": now_ms,
        "equity": round(equity, 2),
        "cash": round(cash_balance, 2),
        "positions_value": round(positions_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_asset_usd": round(total_asset_usd, 2),
        "sharpe_ratio": round(sr, 4) if sr is not None else None,
        "win_rate": round(wr, 4) if wr is not None else None,
        "total_trades": total_trades,
    }).execute()

    return {"equity": equity, "cash": cash_balance, "sharpe": sr, "win_rate": wr,
            "total_pnl": total_pnl, "total_trades": total_trades,
            "scored_trades": scored_trades, "total_asset_usd": total_asset_usd,
            "realized_fees": realized_fees, "capital_locked": capital_locked,
            "positions_value": positions_value}


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

    print("\n[1b/6] Estimating expected moves (fallback -- unconditional)...")
    # Only used for symbols whose prediction row carries no magnitude forecast.
    # These are unconditional averages, so the break-even numbers below are the
    # *typical* bar, not the bar being traded.
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
        for _, sig in active_signals.iterrows():
            side = "BUY" if sig["signal"] == 1 else "SELL"
            move = resolve_expected_move(sig, expected_moves)
            price = prices.get(sig["symbol"])
            if move is None:
                print(f"    {sig['symbol']}: {side} (P(up)={sig['probability_up']:.3f}, "
                      "no magnitude forecast)")
                continue
            exp_ret = expected_return(sig["probability_up"], move)
            forecast = predicted_price(price, sig["probability_up"], move) if price else float("nan")
            print(f"    {sig['symbol']}: {side} P(up)={sig['probability_up']:.3f} "
                  f"E|move|={move*100:.3f}% E[ret]={exp_ret*100:+.3f}% "
                  f"-> {forecast:.4f} (needs {breakeven_accuracy(move)*100:.1f}% accuracy)")

    print("\n[4/6] Reconciling open positions...")
    open_trades = fetch_open_trades(client)
    print(f"  Open positions: {len(open_trades)}")
    # With no fresh signal at all, wanted={} closes everything rather than
    # holding exposure the model can no longer justify.
    wanted = desired_sides(active_signals)
    close_open_positions(client, open_trades, prices, now_ms, wanted=wanted)

    print("\n[5/6] Allocating capital...")
    open_trades_after = fetch_open_trades(client)
    held = desired_sides_from_trades(open_trades_after)
    trade_history = fetch_trade_history(client)

    # `pnl` is already net of fees, so equity is initial capital plus realized
    # PnL -- no separate fee subtraction (that double-charge is what pushed the
    # recorded cash $11.74 below its true value).
    total_asset_usd = INITIAL_CASH
    if not trade_history.empty and "pnl" in trade_history.columns:
        total_asset_usd += float(trade_history["pnl"].fillna(0).sum())

    existing_exposure = (
        float(open_trades_after["size"].sum()) if not open_trades_after.empty else 0.0
    )
    available_cash = max(0.0, total_asset_usd - existing_exposure)
    print(f"  Equity ${total_asset_usd:.2f} | committed ${existing_exposure:.2f} "
          f"| free ${available_cash:.2f}")

    if active_signals.empty:
        print("  No fresh signals; not opening anything.")
    else:
        open_new_positions(client, active_signals, prices, now_ms, model_name, total_asset_usd,
                           held=held, expected_moves=expected_moves,
                           available_cash=available_cash,
                           existing_exposure=existing_exposure)

    print("\n[6/6] Updating portfolio...")
    open_trades_final = fetch_open_trades(client)
    portfolio = update_portfolio(client, INITIAL_CASH, open_trades_final, prices, trade_history, now_ms)
    print(f"  Total Asset: ${portfolio['total_asset_usd']:.2f}")
    print(f"  Equity: ${portfolio['equity']:.2f}")
    print(f"  Cash: ${portfolio['cash']:.2f} "
          f"| positions ${portfolio['positions_value']:.2f}")
    print(f"  Total P&L: ${portfolio['total_pnl']:.2f} "
          f"(fees paid ${portfolio['realized_fees']:.2f}, already inside P&L)")
    scored = portfolio["scored_trades"]
    if portfolio["sharpe"] is None:
        print(f"  Sharpe: n/a ({scored}/{MIN_SHARPE_TRADES} scored trades needed)")
    else:
        print(f"  Sharpe: {portfolio['sharpe']:.4f}")
    if portfolio["win_rate"] is None:
        print("  Win Rate: n/a (no trades closed since the stats epoch)")
    else:
        print(f"  Win Rate: {portfolio['win_rate']:.2%} (over {scored} trades)")
    print(f"  Total Trades: {portfolio['total_trades']} "
          f"(scored since epoch: {portfolio['scored_trades']})")

    print("\n" + "=" * 60)
    print("Paper trading cycle complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()