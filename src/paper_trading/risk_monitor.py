"""Risk-only pass: check stops, targets and holding limits. Never opens.

The full pipeline is hourly because the data is hourly -- 1h candles, a
next-1h target -- so running it more often re-derives the same prediction from
the same closed bar at six times the cost. Risk is different: a stop is about
where price is *now*, and that changes continuously.

Today an open position's stop is only consulted when the full pipeline runs.
Against a `0 * * * *` cron, GitHub actually delivered gaps of 47 to 136
minutes, so "a 1.5% stop" has in practice meant "a 1.5% stop, checked at some
point in the next one to two hours". This module closes that window without
retraining anything.

Deliberately narrow:

  * it never opens a position, so it cannot act on a stale prediction;
  * it never consults a signal, so it cannot disagree with the hourly cycle
    about direction -- only `risk_exit_reason()` decides;
  * it writes a `portfolio` row so the equity curve reflects the exit, using
    the same accounting as the full cycle.

Run it as often as you like. It costs one ticker call per symbol and two
database reads.
"""

import sys
import os
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.supabase_client import get_client
from src.paper_trading.engine import (
    INITIAL_CASH, MAX_HOLDING_HOURS, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    augment_history, close_trade, create_exchange, fetch_live_prices,
    fetch_open_trades, fetch_trade_history, open_by_symbol, position_age_hours,
    risk_exit_reason, unrealized_return, update_portfolio,
)


def evaluate_risk(open_trades: pd.DataFrame, prices: dict[str, float], now_ms: int):
    """(trade, price, reason) for every position risk says to close.

    Duplicated symbols are reported rather than acted on, matching the full
    cycle: with more than one open row the engine cannot tell which position a
    price refers to, and guessing is how a third position got opened.
    """
    singles, duplicates = open_by_symbol(open_trades)

    exits = []
    for symbol, trade in sorted(singles.items()):
        price = prices.get(symbol)
        if price is None:
            print(f"  {symbol}: no price, holding")
            continue
        reason = risk_exit_reason(trade, price, now_ms)
        ret = unrealized_return(trade["side"], trade["entry_price"], price)
        age = position_age_hours(trade, now_ms)
        age_note = f"{age:.1f}h" if age is not None else "age unknown"
        if reason is None:
            print(f"  {symbol}: {trade['side']} {ret*100:+.2f}%, {age_note} -- holding")
            continue
        print(f"  {symbol}: {trade['side']} {ret*100:+.2f}%, {age_note} -- {reason}")
        exits.append((trade, price, reason))

    return exits, duplicates


def main():
    print("=" * 60)
    print("Quant Bot - Risk Monitor (exits only, no training)")
    print("=" * 60)
    print(f"  stop {STOP_LOSS_PCT*100:.2f}% | target {TAKE_PROFIT_PCT*100:.2f}% "
          f"| max hold {MAX_HOLDING_HOURS:.0f}h")

    client = get_client()
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    open_trades = fetch_open_trades(client)
    if open_trades.empty:
        print("\nNo open positions. Nothing to check.")
        print("=" * 60)
        return

    print(f"\n[1/3] Marking {len(open_trades)} open position(s)...")
    exchange = create_exchange()
    symbols = sorted({row["symbol"] for _, row in open_trades.iterrows()})
    prices = fetch_live_prices(exchange, symbols)

    exits, duplicates = evaluate_risk(open_trades, prices, now_ms)

    # Read the history BEFORE closing anything. Reading it afterwards would
    # already contain this pass's closes, and `augment_history` would then add
    # them a second time -- a stop-out booked at twice its loss.
    trade_history = fetch_trade_history(client)

    print(f"\n[2/3] Closing {len(exits)} position(s)...")
    closed, failures = [], []
    for trade, price, reason in exits:
        result = close_trade(client, trade, price, now_ms, reason)
        if result is None:
            failures.append(trade["symbol"])
        else:
            closed.append(result)

    if not closed and not failures:
        print("  Nothing to close.")

    print("\n[3/3] Updating portfolio...")
    # Only write an equity row when something actually changed. A risk pass
    # every ten minutes would otherwise bury the hourly cycle's rows under
    # six times as many identical ones, and `compute_sharpe` reads the trade
    # log rather than this table, so the extra rows buy nothing.
    if closed:
        # Built locally from rows this pass did not close, rather than
        # re-reading -- same reasoning as the hourly cycle, and it keeps the
        # equity row consistent with the writes this process confirmed.
        closed_ids = {c.trade_id for c in closed}
        still_open = pd.DataFrame(
            [row.to_dict() for _, row in open_trades.iterrows()
             if row["id"] not in closed_ids])
        portfolio = update_portfolio(client, INITIAL_CASH, still_open, prices,
                                     augment_history(trade_history, closed), now_ms)
        print(f"  Equity: ${portfolio['equity']:.2f} "
              f"| cash ${portfolio['cash']:.2f} "
              f"| positions ${portfolio['positions_value']:.2f}")
    else:
        print("  No position changed; leaving the equity curve to the hourly cycle.")

    if duplicates:
        print(f"\n  DUPLICATE OPEN: {', '.join(sorted(duplicates))} "
              "-- not acted on. Apply migration 005.")
    if failures:
        print(f"  Failed to close: {', '.join(failures)}")

    print("\n" + "=" * 60)
    print(f"Risk pass complete. Closed {len(closed)}, held "
          f"{len(open_trades) - len(closed) - len(failures)}.")
    print("=" * 60)

    if duplicates or failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
