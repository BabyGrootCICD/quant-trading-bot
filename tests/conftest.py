import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The supabase client refuses to build without these; tests never hit the network.
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")

# --- the suite pins its own economics --------------------------------------
#
# Every strategy knob is read once at import time from the environment
# (`config.settings.env_float` / `env_str`), and the hourly workflow exports
# all of them to *every* step -- including the unit-test step. So setting the
# repo variable EXECUTION_MODE=maker for production silently rewrote the round
# trip from 0.30% to 0.08% underneath the tests and turned 12 of them red:
# `assert 0.0008 == 0.003`, `is_tradeable(0.652, 0.004)` flipping to True, and
# so on. Nothing was wrong with the code; the assertions had simply been
# measuring the environment.
#
# A test suite whose results depend on a production config value is not a test
# suite, so clear the knobs before any `src` module is imported. pytest loads
# this file before collecting test modules, which is what makes that possible.
# Tests that care about a non-default value pass it explicitly -- e.g.
# `round_trip_cost_pct(mode="maker")` or `is_tradeable(..., cost=...)`.
STRATEGY_ENV_KNOBS = (
    "EXECUTION_MODE", "TAKER_FEE", "MAKER_FEE", "SLIPPAGE_BPS", "MAKER_SLIPPAGE_BPS",
    "MIN_EDGE_MARGIN", "KELLY_SCALE", "MAX_POSITION_FRAC", "MAX_GROSS_EXPOSURE",
    "MIN_POSITION_USD", "MAX_HOLDING_HOURS", "STOP_LOSS_PCT", "TAKE_PROFIT_PCT",
    "SIGNAL_EXIT_BAND", "MIN_HORIZON_FRACTION", "MAX_PREDICTION_AGE_HOURS",
    "STATS_EPOCH_MS", "ACTIVE_MODEL",
)

for _knob in STRATEGY_ENV_KNOBS:
    os.environ.pop(_knob, None)
