-- 004: Expected-value allocation.
--
-- The bot had a direction model and no magnitude model, so the EV gate scored
-- every bar against its symbol's *unconditional* mean absolute move -- one
-- constant per symbol. EV then reduced to a function of probability alone, the
-- answer was "no" on every bar of every symbol, and the engine has opened
-- nothing since the gate was introduced.
--
-- This migration adds the storage for the missing half:
--   * features.target_move_1h  -- the magnitude label to train on
--   * predictions.*            -- the magnitude head's output, so the engine
--                                 sizes on a conditional forecast instead of
--                                 recomputing a constant from live candles
--   * paper_trades.exit_reason -- positions had no exit policy at all; they
--                                 were closed only on a signal flip, so a
--                                 position whose signal never changed was held
--                                 indefinitely and equity froze.

ALTER TABLE features ADD COLUMN IF NOT EXISTS target_move_1h DOUBLE PRECISION;

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS expected_move_pct DOUBLE PRECISION;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS expected_return_pct DOUBLE PRECISION;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS predicted_price DOUBLE PRECISION;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS horizon_hours INT DEFAULT 1;

ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS exit_reason VARCHAR(24);

-- `upsert_latest_prediction()` targets on_conflict="symbol,timestamp"; without
-- this constraint PostgREST rejects the upsert outright.
CREATE UNIQUE INDEX IF NOT EXISTS predictions_symbol_timestamp_key
    ON predictions(symbol, timestamp);
