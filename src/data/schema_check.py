"""Preflight schema check for the hourly pipeline.

The bot silently bled for a full day because commit 1d1ec52 added 16 feature
columns without a matching migration: every `features` upsert was rejected with
PGRST204, the trainer then had nothing to train on, and the engine kept
replaying one frozen prediction set. A green workflow hid all of it.

This module fails loudly and early instead.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.supabase_client import get_client
from src.models.features import FEATURE_COLS

REQUIRED_COLUMNS = {
    "features": FEATURE_COLS + ["symbol", "timestamp", "target_1h", "target_4h",
                                "target_move_1h"],
    "candles": ["symbol", "timestamp", "open", "high", "low", "close", "volume"],
    "predictions": ["symbol", "timestamp", "model_name", "prediction", "probability_up",
                    "confidence", "expected_move_pct", "expected_return_pct",
                    "predicted_price", "horizon_hours"],
    "paper_trades": ["symbol", "side", "entry_price", "size", "status", "pnl", "fees",
                     "actual_pnl_usd", "exit_reason"],
    "portfolio": ["timestamp", "equity", "cash", "positions_value", "total_pnl",
                  "sharpe_ratio", "win_rate", "total_trades", "total_asset_usd"],
}


def missing_columns(client, table: str, required: list[str]) -> list[str]:
    """Columns in `required` that the live table does not expose.

    PostgREST returns column names in the response, but only for rows that
    exist -- so we ask for exactly the required columns and let it tell us
    which one it cannot find.
    """
    missing = []
    for col in required:
        try:
            client.table(table).select(col).limit(1).execute()
        except Exception as e:
            if "PGRST204" in str(e) or "does not exist" in str(e) or "schema cache" in str(e):
                missing.append(col)
            else:
                raise
    return missing


def check(client) -> dict[str, list[str]]:
    problems = {}
    for table, required in REQUIRED_COLUMNS.items():
        missing = missing_columns(client, table, required)
        if missing:
            problems[table] = missing
    return problems


def main():
    print("=" * 60)
    print("Quant Bot - Schema Preflight")
    print("=" * 60)

    problems = check(get_client())

    if not problems:
        print("  Schema OK: all required columns present.")
        print("=" * 60)
        return

    for table, missing in problems.items():
        print(f"  MISSING on '{table}': {', '.join(missing)}")
    print()
    print("  A migration has not been applied. Run:")
    print("    python -m migrations.run_migrations")
    print("  then paste the pending SQL into the Supabase SQL Editor.")
    print("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    main()
