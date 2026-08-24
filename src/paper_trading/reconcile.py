"""Pure position reconciliation logic (issue #3).

Kept free of ccxt / supabase / network imports so it can be unit-tested in
isolation. The engine imports these symbols and adds the DB-writing execution.
"""
import pandas as pd

# Reconciliation actions.
KEEP = "KEEP"
CLOSE = "CLOSE"
OPEN = "OPEN"
REVERSE = "REVERSE"
ERROR = "ERROR"


def side_from_signal(signal: int) -> str | None:
    """1 -> long, -1 -> short, 0 -> None (flat/close)."""
    if signal == 1:
        return "long"
    if signal == -1:
        return "short"
    return None


def plan_reconciliation(signals: pd.DataFrame, open_trades: pd.DataFrame, prices: dict) -> list[dict]:
    """Pure classification (no DB writes) of what each symbol should do.

    Uses the FULL signals frame (not just active signals) so that signal==0 can
    explicitly flatten an existing position. Each item is one of:
      KEEP / CLOSE / OPEN / REVERSE / ERROR.

    Contract:
      - no signal row for a symbol -> KEEP the open trade ("no decision");
      - signal 0 -> CLOSE an existing trade, open nothing;
      - desired side == existing side -> KEEP unchanged (no fee);
      - desired side != existing side -> REVERSE (close first, then open);
      - current price missing -> no transition (KEEP);
      - >1 open row for a symbol -> ERROR (fail closed, do not guess).
    """
    desired: dict[str, dict] = {}
    for _, row in signals.iterrows():
        desired[row["symbol"]] = {"side": side_from_signal(int(row["signal"])), "row": row}

    open_by: dict[str, pd.Series] = {}
    duplicates: set[str] = set()
    if open_trades is not None and not open_trades.empty:
        for _, trade in open_trades.iterrows():
            symbol = trade["symbol"]
            if symbol in open_by:
                duplicates.add(symbol)
            open_by[symbol] = trade

    plan: list[dict] = []
    for symbol in sorted(set(desired) | set(open_by)):
        if symbol in duplicates:
            plan.append({"symbol": symbol, "action": ERROR, "reason": "multiple open rows"})
            continue

        existing = open_by.get(symbol)
        decision = desired.get(symbol)
        has_price = symbol in prices

        # No signal for this symbol: preserve whatever is open.
        if decision is None:
            if existing is not None:
                plan.append({"symbol": symbol, "action": KEEP, "reason": "no signal"})
            continue

        desired_side = decision["side"]

        if existing is None:
            if desired_side is None:
                continue  # signal 0 and nothing open -> no-op
            if not has_price:
                plan.append({"symbol": symbol, "action": KEEP, "reason": "no price"})
            else:
                plan.append({"symbol": symbol, "action": OPEN, "side": desired_side, "row": decision["row"]})
            continue

        existing_side = existing["side"]
        if desired_side is None:
            if not has_price:
                plan.append({"symbol": symbol, "action": KEEP, "reason": "no price"})
            else:
                plan.append({"symbol": symbol, "action": CLOSE, "trade": existing})
        elif desired_side == existing_side:
            plan.append({"symbol": symbol, "action": KEEP, "reason": "same side"})
        else:
            if not has_price:
                plan.append({"symbol": symbol, "action": KEEP, "reason": "no price"})
            else:
                plan.append({"symbol": symbol, "action": REVERSE, "side": desired_side,
                             "trade": existing, "row": decision["row"]})
    return plan
