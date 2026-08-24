-- 004: At most one open position per symbol (issue #5).
--
-- Overlapping pipeline runs (schedule + manual dispatch, or a run that outlives
-- its hour because `concurrency.cancel-in-progress` is false) can both read
-- "no open trade" for a symbol and then both INSERT a `status='open'` row,
-- leaving duplicate open positions. The directional reconciliation in the engine
-- cannot prevent that race on its own, so enforce it at the DB level.

-- Older code could already have created duplicate open rows. The partial unique
-- index below would fail to build while they exist, so collapse them first:
-- keep the most recent open row per symbol and close the rest.
WITH ranked AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY symbol
               ORDER BY entry_time DESC, id DESC
           ) AS rn
    FROM paper_trades
    WHERE status = 'open'
)
UPDATE paper_trades p
SET status = 'closed'
FROM ranked r
WHERE p.id = r.id
  AND r.rn > 1;

-- Enforce the invariant. A concurrent second insert now fails with a unique
-- violation instead of silently creating a duplicate; the engine treats that
-- as "already open" and skips (see open_new_positions).
CREATE UNIQUE INDEX IF NOT EXISTS paper_trades_one_open_per_symbol
    ON paper_trades (symbol)
    WHERE status = 'open';
