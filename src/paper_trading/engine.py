import sys
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

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
    expected_return, predicted_price, horizon_fraction, collectable_move,
    MIN_HORIZON_FRACTION, TAKER_FEE, SLIPPAGE_BPS,
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
#
# Three hours, not two. A prediction is now stamped with the last *complete*
# bar (the fetcher no longer stores the forming one), so a cycle running at
# :55 reads a prediction 1.92h old through no fault of anything -- and
# GitHub's scheduler adds its own 47-136 minute drift on top. Two hours would
# reject perfectly good predictions and silently flatten the book.
MAX_PREDICTION_AGE_HOURS = env_float("MAX_PREDICTION_AGE_HOURS", 3)

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

# How far the probability must cross 0.5 *against* an open position before the
# signal alone closes it. Entry is decided by the EV gate, so the directional
# threshold sits at 0.5 -- without a band here a probability oscillating
# 0.499/0.501 would round-trip the whole book every hour, which is precisely
# the churn that produced the original bleed.
SIGNAL_EXIT_BAND = env_float("SIGNAL_EXIT_BAND", 0.02)
STOP_LOSS_PCT = env_float("STOP_LOSS_PCT", 0.015)
TAKE_PROFIT_PCT = env_float("TAKE_PROFIT_PCT", 0.02)

# Every value `exit_reason` may take. The column is VARCHAR(24), so a longer
# label would be truncated on the way in and silently mis-read on the way out.
#
# `no_signal` is legacy: it was written whenever the symbol was absent from
# `wanted`, which conflated "the model said flat", "the model said nothing"
# and "there were no predictions at all". Those are now `flat_signal` and a
# preserved position respectively. Historical rows still carry it.
EXIT_REASONS = frozenset({
    "stop_loss", "take_profit", "max_holding", "signal_flip", "flat_signal",
    "no_signal",
})


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
    """Legacy signal map. Superseded by `desired_side_by_symbol()`.

    Cannot represent "the model said flat": a signal of 0 produces no key, the
    same as a symbol that was never scored. Kept only for
    `close_open_positions()` and the tests that drive it.
    """
    if signals is None or signals.empty:
        return {}
    out = {}
    for _, row in signals.iterrows():
        if row["signal"] == 1:
            out[row["symbol"]] = "long"
        elif row["signal"] == -1:
            out[row["symbol"]] = "short"
    return out


def desired_side_by_symbol(signals: pd.DataFrame) -> dict[str, str | None]:
    """Map symbol -> "long" / "short" / None over the FULL signals frame.

    The distinction this exists for: `None` means the model looked and had no
    directional view (signal == 0), while a symbol *absent from the dict* means
    the model never spoke for it at all. `desired_sides()` was fed
    `active_signals`, so both arrived downstream as a missing key and both
    forced a close -- a stale prediction and a genuine flat call were
    indistinguishable, in the code and in the persisted `exit_reason`.
    """
    if signals is None or signals.empty:
        return {}
    out: dict[str, str | None] = {}
    for _, row in signals.iterrows():
        signal = row["signal"]
        if signal == 1:
            out[row["symbol"]] = "long"
        elif signal == -1:
            out[row["symbol"]] = "short"
        else:
            out[row["symbol"]] = None
    return out


def open_by_symbol(open_trades: pd.DataFrame) -> tuple[dict[str, dict], dict[str, list]]:
    """Split open positions into one-per-symbol, plus the symbols that have more.

    `desired_sides_from_trades()` was a dict comprehension, so two open rows
    for one symbol collapsed to whichever came last -- and `fetch_open_trades`
    has no `.order(...)`, so *which* side survived was not deterministic. With
    two opposite-side rows that neither trigger an exit, the surviving side
    could then pass `build_candidates`' `held` guard and open a third
    position. Duplicated symbols are excluded here so nothing downstream can
    guess; the caller fails closed on them instead.

    The duplicated rows are still returned, because refusing to *act* on a
    position is not a reason to stop *counting* it -- its capital is really
    committed, and dropping it from the portfolio mark would understate the
    book by its whole notional.
    """
    if open_trades is None or open_trades.empty:
        return {}, {}

    grouped: dict[str, list[dict]] = {}
    for _, row in open_trades.iterrows():
        grouped.setdefault(row["symbol"], []).append(row.to_dict())

    singles = {sym: rows[0] for sym, rows in grouped.items() if len(rows) == 1}
    duplicates = {sym: rows for sym, rows in sorted(grouped.items()) if len(rows) > 1}
    return singles, duplicates


class Action(str, Enum):
    KEEP = "keep"        # desired side matches; leave the row completely alone
    CLOSE = "close"      # close, open nothing
    OPEN = "open"        # nothing open, the signal wants a side
    REVERSE = "reverse"  # close first, open the other side only if that succeeds
    SKIP = "skip"        # deliberately no transition
    ERROR = "error"      # fail closed


@dataclass(frozen=True)
class Decision:
    """What should happen to one symbol this cycle, decided before any write."""
    symbol: str
    action: Action
    reason: str
    price: float | None = None
    current_side: str | None = None
    desired_side: str | None = None
    trade: dict | None = None
    signal_row: dict | None = None
    # Every open row for a symbol that has more than one. Carried so the
    # portfolio still marks capital the cycle refuses to trade.
    duplicate_rows: tuple = ()


def signal_wants_exit(current_side: str, prob_up: float,
                      band: float = SIGNAL_EXIT_BAND) -> bool:
    """True when the probability has moved far enough against an open position.

    Entry direction is `sign(p - 0.5)`, so without a band a probability
    drifting across 0.5 would reverse the book every hour and pay the round
    trip each time for no change in conviction.
    """
    if not np.isfinite(prob_up):
        return False
    if current_side == "long":
        return prob_up < 0.5 - band
    return prob_up > 0.5 + band


def classify_positions(open_trades: pd.DataFrame, signals: pd.DataFrame,
                       prices: dict[str, float], now_ms: int, *,
                       signals_available: bool = True,
                       exit_band: float = SIGNAL_EXIT_BAND,
                       max_holding_hours: float = MAX_HOLDING_HOURS,
                       stop_loss_pct: float = STOP_LOSS_PCT,
                       take_profit_pct: float = TAKE_PROFIT_PCT) -> list[Decision]:
    """Decide every symbol's transition before touching the database.

    Pure: no client, no writes, no ordering side effects. Separating this from
    execution is what makes the contract testable -- previously the close pass
    and the open pass each decided independently, with a re-read of
    `paper_trades` in between, so no single place knew what the cycle intended.

    Classification decides *direction and transition* only. Whether an OPEN is
    worth taking is still `build_candidates`' EV gate, and how large it is is
    still `allocate()`.
    """
    open_map, duplicates = open_by_symbol(open_trades)
    desired_map = desired_side_by_symbol(signals)

    prob_by_symbol = {}
    if signals is not None and not signals.empty and "probability_up" in signals.columns:
        for _, row in signals.iterrows():
            prob_by_symbol[row["symbol"]] = float(row["probability_up"])

    signal_rows = {}
    if signals is not None and not signals.empty:
        for _, row in signals.iterrows():
            signal_rows[row["symbol"]] = row.to_dict()

    decisions: list[Decision] = []

    for symbol, rows in duplicates.items():
        decisions.append(Decision(symbol=symbol, action=Action.ERROR,
                                  reason="duplicate_open",
                                  duplicate_rows=tuple(rows)))

    for symbol in sorted(set(open_map) | set(desired_map)):
        if symbol in duplicates:
            continue

        trade = open_map.get(symbol)
        has_signal = symbol in desired_map
        desired = desired_map.get(symbol)
        signal_row = signal_rows.get(symbol)

        price = prices.get(symbol)
        if price is None:
            # No price means no honest mark, so no transition in either
            # direction -- an existing position is left exactly as it is.
            if trade is not None:
                decisions.append(Decision(symbol=symbol, action=Action.SKIP,
                                          reason="no_price", current_side=trade["side"],
                                          trade=trade))
            continue

        if trade is not None:
            risk = risk_exit_reason(trade, price, now_ms,
                                    max_holding_hours=max_holding_hours,
                                    stop_loss_pct=stop_loss_pct,
                                    take_profit_pct=take_profit_pct)
            if risk is not None:
                # Checked before any signal comparison: a stop must not be
                # overridden by a model that still likes the position.
                decisions.append(Decision(symbol=symbol, action=Action.CLOSE,
                                          reason=risk, price=price,
                                          current_side=trade["side"], trade=trade))
                continue

        if not signals_available or not has_signal:
            # The model said nothing about this symbol -- no prediction row, or
            # the whole set was stale. Preserve the position and record no
            # decision; the stop and MAX_HOLDING_HOURS above still bound it, so
            # it cannot be orphaned. Flattening here would pay a round trip on
            # every open position for a transient trainer or database failure.
            if trade is not None:
                decisions.append(Decision(symbol=symbol, action=Action.SKIP,
                                          reason="no_decision", price=price,
                                          current_side=trade["side"], trade=trade))
            continue

        if desired is None:
            if trade is not None:
                decisions.append(Decision(symbol=symbol, action=Action.CLOSE,
                                          reason="flat_signal", price=price,
                                          current_side=trade["side"], trade=trade))
            continue

        if trade is None:
            decisions.append(Decision(symbol=symbol, action=Action.OPEN,
                                      reason="signal", price=price,
                                      desired_side=desired, signal_row=signal_row))
            continue

        current_side = trade["side"]
        if desired == current_side:
            decisions.append(Decision(symbol=symbol, action=Action.KEEP,
                                      reason="signal_unchanged", price=price,
                                      current_side=current_side, desired_side=desired,
                                      trade=trade))
            continue

        prob_up = prob_by_symbol.get(symbol, float("nan"))
        if not signal_wants_exit(current_side, prob_up, band=exit_band):
            # The side flipped, but only just. Holding costs nothing; churning
            # costs the round trip.
            decisions.append(Decision(symbol=symbol, action=Action.KEEP,
                                      reason="within_exit_band", price=price,
                                      current_side=current_side, desired_side=desired,
                                      trade=trade))
            continue

        decisions.append(Decision(symbol=symbol, action=Action.REVERSE,
                                  reason="signal_flip", price=price,
                                  current_side=current_side, desired_side=desired,
                                  trade=trade, signal_row=signal_row))

    return decisions


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


def risk_exit_reason(trade, current_price: float, now_ms: int,
                     max_holding_hours: float = MAX_HOLDING_HOURS,
                     stop_loss_pct: float = STOP_LOSS_PCT,
                     take_profit_pct: float = TAKE_PROFIT_PCT) -> str | None:
    """Why risk alone says close this position, or None.

    Deliberately knows nothing about the signal. The classifier consults this
    *before* comparing sides, so "the model still likes it" can never keep a
    position through an unbounded drawdown or past its forecast horizon.
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

    return None


def exit_reason(trade, current_price: float, now_ms: int,
                wanted: dict[str, str] | None = None,
                max_holding_hours: float = MAX_HOLDING_HOURS,
                stop_loss_pct: float = STOP_LOSS_PCT,
                take_profit_pct: float = TAKE_PROFIT_PCT) -> str | None:
    """Legacy single-position exit rule. `main()` no longer calls this.

    Superseded by `classify_positions()`, which can tell "the model said flat"
    apart from "the model said nothing" -- a distinction this function cannot
    make, because `wanted` is built from active signals only and both cases
    arrive here as a missing key. Retained for `close_open_positions()` and
    the tests that pin the risk rules directly.
    """
    risk = risk_exit_reason(trade, current_price, now_ms,
                            max_holding_hours=max_holding_hours,
                            stop_loss_pct=stop_loss_pct,
                            take_profit_pct=take_profit_pct)
    if risk is not None:
        return risk

    if wanted is None:
        return "no_signal"
    desired = wanted.get(trade["symbol"])
    if desired is None:
        return "no_signal"
    if desired != trade["side"]:
        return "signal_flip"

    return None


@dataclass(frozen=True)
class ClosedPosition:
    """A close this process performed and the database confirmed.

    The engine keeps its own ledger of these so it can work out post-close
    cash and exposure without re-reading `paper_trades`. A re-read would also
    pick up a concurrent run's writes, which is exactly the race the partial
    unique index from migration 005 exists to arbitrate.
    """
    symbol: str
    trade_id: object
    side: str
    size: float
    entry_price: float
    exit_price: float
    entry_time: object
    exit_time: int
    fees: float
    net_pnl: float
    reason: str


def close_trade(client, trade, current_price: float, now_ms: int,
                reason: str) -> ClosedPosition | None:
    """Close one position. Returns None if the write failed.

    The `.update()` used to be bare inside a loop: one failure propagated out
    of `main()`, so the remaining symbols never closed *and* the portfolio row
    for that hour was never written. A missing equity record is worse than a
    wrong one, so a failure here is now contained to its own symbol.
    """
    if reason not in EXIT_REASONS:
        # A label longer than exit_reason's VARCHAR(24), or one no reader
        # knows, would be silently truncated or silently unrecognised.
        raise ValueError(f"unknown exit reason {reason!r}; add it to EXIT_REASONS")

    symbol = trade["symbol"]
    side = trade["side"]
    size = float(trade["size"])
    entry_price = float(trade["entry_price"])

    raw_pnl = unrealized_return(side, entry_price, current_price) * size
    # Both legs: the position was opened and is now being closed.
    fees = round_trip_cost(size)
    net_pnl = raw_pnl - fees

    try:
        client.table("paper_trades").update({
            "exit_price": current_price,
            "exit_time": now_ms,
            "pnl": round(net_pnl, 4),
            "actual_pnl_usd": round(net_pnl, 4),
            "fees": round(fees, 4),
            "status": "closed",
            "exit_reason": reason,
        }).eq("id", trade["id"]).execute()
    except Exception as e:
        print(f"  ERROR closing {side} {symbol}: {e}")
        return None

    print(f"  Closed {side} {symbol} [{reason}]: entry={entry_price:.2f} "
          f"exit={current_price:.2f} pnl=${net_pnl:.2f}")

    return ClosedPosition(
        symbol=symbol, trade_id=trade["id"], side=side, size=size,
        entry_price=entry_price, exit_price=float(current_price),
        entry_time=trade.get("entry_time") if hasattr(trade, "get") else None,
        exit_time=now_ms, fees=fees, net_pnl=net_pnl, reason=reason,
    )


def close_open_positions(client, open_trades: pd.DataFrame, prices: dict[str, float], now_ms: int,
                         wanted: dict[str, str] | None = None):
    """Legacy close pass. `main()` now goes through `reconcile_positions()`.

    Kept because it is the entry point the existing exit-policy tests drive.
    It cannot express the reconciliation contract -- `wanted` conflates "the
    model said flat" with "the model said nothing" -- which is why it was
    superseded rather than extended.
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

        if close_trade(client, trade, current_price, now_ms, reason) is not None:
            closed_symbols.append(symbol)

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
                     margin: float = MIN_EDGE_MARGIN,
                     now_ms: int | None = None,
                     min_horizon: float = MIN_HORIZON_FRACTION) -> list[Candidate]:
    """Signals that survive the EV gate, priced in expected-return terms.

    Separating selection from sizing is the point: the gate decides *whether*
    a bet is worth making, the allocator decides *how much* of the book it
    gets, and neither can quietly override the other.

    The forecast is discounted by how much of the bar is still ahead. A
    prediction is made for the next bar, but the pipeline does not enter at the
    top of it -- observed starts run 40-80% into the bar -- and the round trip
    is paid in full regardless. Pricing a part-bar holding period at the full
    move is what let the gate approve trades whose EV it could not collect:
    a setup clearing by +0.100% at the top of the hour is worth -0.137% if
    entered fifty minutes in.
    """
    held = held or {}
    candidates = []

    if signals is None or signals.empty:
        return candidates

    fraction = 1.0 if now_ms is None else horizon_fraction(now_ms)
    if now_ms is not None and fraction < min_horizon:
        # Rounded percentages read as "25% < 25%" at the boundary, which looks
        # like a bug rather than a rule; print what was actually compared.
        print(f"  Opening nothing: {fraction*60:.1f} min of the forecast bar remain "
              f"(fraction {fraction:.3f} below the {min_horizon:.3f} floor). Too "
              "little of the horizon left for the sqrt(t) model to mean anything.")
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
        forecast_move = resolve_expected_move(row, expected_moves)
        if forecast_move is None:
            print(f"  Skipped {symbol}: no expected-move forecast")
            continue

        # What is actually still collectable, not what the whole bar was worth.
        exp_move = collectable_move(forecast_move, fraction)

        if not is_tradeable(strength, exp_move, margin=margin):
            ev = expected_value(strength, exp_move)
            need = breakeven_accuracy(exp_move)
            horizon_note = ""
            if fraction < 1.0:
                horizon_note = (f", {forecast_move*100:.3f}% over a full bar but "
                                f"{fraction:.0%} of it left")
            print(f"  Skipped {symbol}: EV {ev*100:+.3f}% "
                  f"(strength {strength:.2f}, E|move| {exp_move*100:.3f}%{horizon_note}, "
                  f"needs {need*100:.1f}% accuracy)")
            continue

        # The discounted move is what the allocator sizes on too -- Kelly reads
        # both the edge and the variance off it, so passing the undiscounted
        # figure here would size a part-bar bet as though it were a full one.
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
                       existing_exposure: float = 0.0,
                       result=None):
    """Gate on expected value, then allocate capital across what survives.

    Sizing used to be `risk_budget / |estimated_change|` capped at $100, per
    symbol, in isolation: inversely proportional to the predicted move, blind
    to the other candidates, and blind to how much cash the portfolio actually
    had. It is now fractional-Kelly on the EV and variance the model produced,
    scaled to fit the cash and gross-exposure limits, so the money goes where
    the edge per unit of risk is largest and stops when the budget is gone.
    """
    candidates = build_candidates(signals, prices, held=held,
                                  expected_moves=expected_moves, now_ms=now_ms)
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

        payload = {
            "symbol": c.symbol,
            "side": c.side,
            "entry_price": current_price,
            "size": round(size, 2),
            "entry_time": now_ms,
            "model_name": model_name,
            "prediction_at_entry": round(c.probability_up, 4),
            "status": "open",
        }
        try:
            client.table("paper_trades").insert(payload).execute()
        except Exception as e:
            if is_duplicate_open_error(e):
                # A concurrent run already opened this symbol. The partial
                # unique index paper_trades_one_open_per_symbol (migration
                # 005) rejected the duplicate; losing that race is routine.
                print(f"  Skipped {c.symbol}: already open (unique index) -- concurrent run")
            else:
                # Anything else -- auth, schema drift, network -- used to be
                # reported as "likely concurrent run" and swallowed just as
                # quietly. Name it, and make the cycle exit non-zero.
                print(f"  ERROR opening {c.side} {c.symbol}: {e}")
                if result is not None:
                    result.open_failures.append(c.symbol)
            continue

        opened.append(c.symbol)
        if result is not None:
            # The engine's local book, so the portfolio row can be built
            # without re-reading paper_trades after writing to it.
            result.opened.append(payload)
        print(f"  Opened {c.side} {c.symbol} @ {current_price:.2f} size=${size:.2f} "
              f"EV={c.ev*100:+.3f}% E|move|={c.expected_abs_move*100:.3f}% "
              f"P(up)={c.probability_up:.3f} -> forecast {forecast:.2f}")

    return opened


_UNIQUE_MARKERS = ("23505", "duplicate key value",
                   "paper_trades_one_open_per_symbol", "unique constraint")


def is_duplicate_open_error(exc: Exception) -> bool:
    """True when an insert failed because the symbol is already open.

    Scans SQLSTATE, message, details and the repr rather than matching a
    driver exception class, so it works whether PostgREST raises an APIError,
    a dict payload, or a plain Exception. The distinction matters: a lost race
    is routine, but an auth, schema or network failure was being reported with
    the same "likely concurrent run" message and swallowed just as quietly.
    """
    parts = [
        str(getattr(exc, "code", "")),
        str(getattr(exc, "message", "")),
        str(getattr(exc, "details", "")),
        str(getattr(exc, "args", "")),
        str(exc),
    ]
    blob = " ".join(parts).lower()
    return any(marker in blob for marker in _UNIQUE_MARKERS)


@dataclass
class ReconcileResult:
    """Everything this cycle decided and everything the database confirmed."""
    decisions: list = field(default_factory=list)
    closed: list = field(default_factory=list)
    opened: list = field(default_factory=list)
    kept: list = field(default_factory=list)
    duplicates: list = field(default_factory=list)
    close_failures: list = field(default_factory=list)
    open_failures: list = field(default_factory=list)

    def had_errors(self) -> bool:
        return bool(self.duplicates or self.close_failures or self.open_failures)

    def exposure(self) -> float:
        """Notional still committed after the close phase."""
        return float(sum(float(row["size"]) for row in self.kept))

    def realized_delta(self) -> float:
        """Net PnL from the closes this cycle actually wrote."""
        return float(sum(c.net_pnl for c in self.closed))


def execute_decisions(client, decisions: list, prices: dict[str, float], now_ms: int, *,
                      model_name: str, equity_before: float,
                      expected_moves: dict[str, float] | None = None) -> ReconcileResult:
    """Report errors, close, then open -- in that order, never interleaved.

    A REVERSE's open leg is enqueued only once its close is confirmed, so a
    failed close can never leave the book both still long and freshly short.
    """
    result = ReconcileResult(decisions=list(decisions))

    for d in decisions:
        if d.action is Action.ERROR:
            result.duplicates.append(d.symbol)
            # Counted, not traded: the capital is committed either way.
            result.kept.extend(d.duplicate_rows)
            sides = ", ".join(f"#{r['id']} {r['side']}" for r in d.duplicate_rows)
            print(f"  DUPLICATE OPEN {d.symbol}: {len(d.duplicate_rows)} rows "
                  f"({sides}). Refusing to act on it -- apply migration 005.")

    # --- hold ------------------------------------------------------------
    for d in decisions:
        if d.action in (Action.KEEP, Action.SKIP):
            result.kept.append(d.trade)
            if d.action is Action.SKIP and d.reason == "no_decision":
                print(f"  No decision for {d.symbol}: holding {d.current_side} "
                      "(model said nothing this cycle)")
            elif d.action is Action.SKIP and d.reason == "no_price":
                print(f"  No price for {d.symbol}: holding {d.current_side}, no transition")
            else:
                age = position_age_hours(d.trade, now_ms) if d.trade else None
                age_note = f", {age:.1f}h old" if age is not None else ""
                note = "within exit band" if d.reason == "within_exit_band" else "signal unchanged"
                print(f"  Holding {d.current_side} {d.symbol} ({note}{age_note})")

    # --- close -----------------------------------------------------------
    reversible = []
    for d in decisions:
        if d.action not in (Action.CLOSE, Action.REVERSE):
            continue
        closed = close_trade(client, d.trade, d.price, now_ms, d.reason)
        if closed is None:
            # Contained to this symbol: the position stays open in the
            # database, so it must stay in `kept` for exposure to match, and
            # its replacement must not be opened.
            result.close_failures.append(d.symbol)
            result.kept.append(d.trade)
            continue
        result.closed.append(closed)
        if d.action is Action.REVERSE:
            reversible.append(d)

    # --- open ------------------------------------------------------------
    open_decisions = [d for d in decisions if d.action is Action.OPEN] + reversible
    if open_decisions:
        rows = [d.signal_row for d in open_decisions if d.signal_row is not None]
        if rows:
            equity = equity_before + result.realized_delta()
            exposure = result.exposure()
            held = {row["symbol"]: row["side"] for row in result.kept if row}
            open_new_positions(
                client, pd.DataFrame(rows), prices, now_ms, model_name, equity,
                held=held, expected_moves=expected_moves,
                available_cash=max(0.0, equity - exposure),
                existing_exposure=exposure,
                result=result,
            )

    return result


def reconcile_positions(client, open_trades: pd.DataFrame, signals: pd.DataFrame,
                        prices: dict[str, float], now_ms: int, *,
                        model_name: str, equity_before: float,
                        expected_moves: dict[str, float] | None = None,
                        signals_available: bool = True) -> ReconcileResult:
    """Classify every symbol, then execute. The only thing `main()` calls.

    Replaces the old unconditional close pass + re-read + open pass. The
    re-read is gone: everything the open phase needs about the post-close book
    is derived from writes this process performed and confirmed, so equity,
    exposure and the portfolio row are consistent by construction rather than
    describing a book a concurrent run may have altered mid-cycle.
    """
    decisions = classify_positions(open_trades, signals, prices, now_ms,
                                   signals_available=signals_available)
    return execute_decisions(client, decisions, prices, now_ms,
                             model_name=model_name, equity_before=equity_before,
                             expected_moves=expected_moves)


def augment_history(trade_history: pd.DataFrame, closed: list) -> pd.DataFrame:
    """Pre-close history plus this cycle's confirmed closes.

    Same fields the post-close re-query used to return, so `total_pnl`,
    `total_trades`, `filter_to_stats_epoch` and `compute_sharpe` see exactly
    what they saw before the re-read was removed.
    """
    if not closed:
        return trade_history
    rows = [{
        "pnl": c.net_pnl,
        "actual_pnl_usd": c.net_pnl,
        "size": c.size,
        "fees": c.fees,
        "entry_time": c.entry_time,
        "exit_time": c.exit_time,
        "status": "closed",
    } for c in closed]
    added = pd.DataFrame(rows)
    if trade_history is None or trade_history.empty:
        return added
    return pd.concat([added, trade_history], ignore_index=True)


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
            size = float(trade["size"])
            if symbol not in prices:
                # No live mark. `size` has already been subtracted from cash as
                # locked capital, so skipping the position here dropped its
                # whole notional out of equity -- a $100 position made equity
                # read $100 low for as long as the price fetch kept failing.
                # Carry it at cost instead: unknown P&L, not vanished capital.
                print(f"  No price for {symbol}: marking position at cost (${size:.2f})")
                positions_value += size
                continue
            current_price = prices[symbol]
            entry_price = trade["entry_price"]
            positions_value += size + unrealized_return(
                trade["side"], entry_price, current_price) * size

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
        # The denominator behind win_rate. total_trades is lifetime while
        # win_rate and sharpe_ratio are post-epoch, so without this the row
        # cannot be read: a win_rate of 0.5 over two trades and over two
        # hundred look identical, and the first is noise.
        "scored_trades": scored_trades,
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

    print("\n[4/6] Reconciling positions (classify -> close -> allocate -> open)...")
    open_trades = fetch_open_trades(client)
    trade_history = fetch_trade_history(client)
    print(f"  Open positions: {len(open_trades)}")

    # `pnl` is already net of fees, so equity is initial capital plus realized
    # PnL -- no separate fee subtraction (that double-charge is what pushed the
    # recorded cash $11.74 below its true value).
    equity_before = INITIAL_CASH
    if not trade_history.empty and "pnl" in trade_history.columns:
        equity_before += float(trade_history["pnl"].fillna(0).sum())

    # One pass: classify every symbol from the FULL signals frame before any
    # write, then execute. `signals_available` is False when nothing fresh
    # arrived at all, which preserves open positions rather than flattening
    # them -- the stop and MAX_HOLDING_HOURS still bound the risk, and paying a
    # round trip on every position for a transient trainer failure does not.
    result = reconcile_positions(
        client, open_trades, signals, prices, now_ms,
        model_name=model_name, equity_before=equity_before,
        expected_moves=expected_moves,
        signals_available=not predictions.empty,
    )

    print("\n[5/6] Cycle summary...")
    committed = result.exposure() + sum(float(r["size"]) for r in result.opened)
    print(f"  Equity ${equity_before + result.realized_delta():.2f} "
          f"| committed ${committed:.2f} "
          f"| held {len(result.kept)} | closed {len(result.closed)} "
          f"| opened {len(result.opened)}")

    print("\n[6/6] Updating portfolio...")
    # Built from this process's own confirmed writes rather than a re-read.
    # A re-read would also pick up a concurrent run's rows, so the portfolio
    # snapshot could describe a book this cycle did not create.
    open_rows = [row for row in result.kept if row] + list(result.opened)
    open_trades_final = pd.DataFrame(open_rows) if open_rows else pd.DataFrame()
    portfolio = update_portfolio(client, INITIAL_CASH, open_trades_final, prices,
                                 augment_history(trade_history, result.closed), now_ms)
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

    # Reported after the portfolio row is written, never instead of it: a
    # missing equity record is worse than a recorded bad hour. The non-zero
    # exit turns the workflow step red rather than reporting green on a
    # half-reconciled book.
    if result.had_errors():
        print("\n" + "=" * 60)
        print("RECONCILE ERRORS")
        if result.duplicates:
            print(f"  Duplicate open rows: {', '.join(result.duplicates)} "
                  "(apply migration 005)")
        if result.close_failures:
            print(f"  Failed to close: {', '.join(result.close_failures)}")
        if result.open_failures:
            print(f"  Failed to open: {', '.join(result.open_failures)}")
        print("=" * 60)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Paper trading cycle complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()