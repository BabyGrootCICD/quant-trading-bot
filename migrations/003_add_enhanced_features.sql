-- 003: Add the enhanced feature columns introduced in commit 1d1ec52.
--
-- Without these columns every `features` upsert fails wholesale with
-- PGRST204 ("Could not find the 'atr_14' column of 'features' in the schema
-- cache"), which froze the features table and silently broke training.
--
-- Also adds the UNIQUE(symbol, timestamp) that `predictions` upserts rely on
-- for their `on_conflict` target.

ALTER TABLE features ADD COLUMN IF NOT EXISTS rsi_7 DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS macd_hist DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS bb_width DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS bb_position DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS volume_ratio_48 DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS atr_14 DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS atr_14_pct DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS williams_r DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS stoch_k DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS stoch_d DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS ha_trend DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS close_pct_ma20 DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS close_pct_ma50 DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS vol_20 DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS skew_20 DOUBLE PRECISION;
ALTER TABLE features ADD COLUMN IF NOT EXISTS target_4h SMALLINT;

-- Older code inserted into `predictions` with no uniqueness guarantee, so
-- duplicate (symbol, timestamp) rows may already exist -- and the unique index
-- below would fail on them. Drop the older copy of each duplicate pair,
-- keeping the highest id (the most recently written row).
--
-- Inspect first if you want to see what this will remove:
--   SELECT symbol, timestamp, count(*) FROM predictions
--   GROUP BY symbol, timestamp HAVING count(*) > 1;
DELETE FROM predictions p
USING predictions q
WHERE p.symbol = q.symbol
  AND p.timestamp = q.timestamp
  AND p.id < q.id;

-- `predictions` upserts use on_conflict="symbol,timestamp"; without this
-- constraint PostgREST rejects them with 42P10.
CREATE UNIQUE INDEX IF NOT EXISTS predictions_symbol_timestamp_key
    ON predictions(symbol, timestamp);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol_time
    ON predictions(symbol, timestamp DESC);
