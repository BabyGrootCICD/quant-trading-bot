import os
from dotenv import load_dotenv

load_dotenv(override=False)

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
# set, churning the same four positions and paying the spread each time. They
# are real recorded trades but they measure the old bug, not the current
# system, so portfolio statistics start here.
#
# 2026-08-24T00:25:00Z -- the first cycle that traded on a fresh prediction.
STATS_EPOCH_MS = int(os.getenv("STATS_EPOCH_MS", "1787531100000"))
