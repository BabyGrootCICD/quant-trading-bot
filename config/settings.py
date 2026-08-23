import os
from dotenv import load_dotenv

load_dotenv(override=False)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binanceus")

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
    "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
    "DOT/USDT", "LINK/USDT", "USDC/USDT", "DAI/USDT",
]

TIMEFRAME = "1h"
HISTORY_YEARS = 2
