import os
from dotenv import load_dotenv

load_dotenv(override=False)

def env_float(name: str, default: float) -> float:
    """Float from the environment, treating blank as unset.

    GitHub Actions substitutes an *empty string* for an undefined `vars.X`, so
    a plain `float(os.getenv(...))` on a tuning knob that has never been set in
    the repo raises ValueError and takes the whole step down.
    """
    raw = os.getenv(name, "")
    raw = raw.strip() if raw else ""
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        print(f"WARNING: {name}={raw!r} is not a number; using {default}")
        return float(default)


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name, "")
    raw = raw.strip() if raw else ""
    return raw or default


SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binanceus")

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
    "XRP/USDT", "ADA/USDT", "DOGE/USDT", "TRX/USDT",
]

TIMEFRAME = "1h"
HISTORY_YEARS = 2

# Trades entered before this instant were made while the pipeline was broken:
# training failed 8/8 every hour and the engine replayed one frozen prediction
# set, churning the same four positions and paying the spread each time.
#
# Migration 006 moved those 77 trades into `paper_trades_archive` and cleared
# the live table, so nothing before this instant remains to filter. The epoch
# is kept as a floor -- it costs nothing, and it means restoring rows from the
# archive for analysis cannot silently contaminate `sharpe_ratio` and
# `win_rate` again.
#
# 2026-08-24T00:25:00Z -- the first cycle that traded on a fresh prediction.
STATS_EPOCH_MS = int(os.getenv("STATS_EPOCH_MS", "1787531100000"))
