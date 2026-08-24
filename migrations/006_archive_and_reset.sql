-- 006: Archive the broken-era books and restart paper trading at $10,000.
--
-- Why a reset needs to touch `paper_trades` and not just `portfolio`:
-- every portfolio row is *recomputed from scratch* each cycle --
--
--     cash = INITIAL_CASH - capital_locked + realized_pnl
--
-- with `realized_pnl` summed over the whole `paper_trades` history (see
-- `update_portfolio` in src/paper_trading/engine.py). Clearing `portfolio`
-- alone therefore changes nothing: the next hourly run recomputes the same
-- equity, the same -11.78 of lifetime P&L and the same 77 trades from the
-- trades table and writes them straight back.
--
-- The 77 recorded trades are real, and docs/01-04 are written against them, so
-- they are copied out rather than dropped.
--
-- Also adds portfolio.scored_trades: the denominator behind `win_rate`. It was
-- computed and printed but never persisted, so a stored `win_rate` of 0.5 over
-- two trades was indistinguishable from 0.5 over two hundred -- and the first
-- is noise. `total_trades` cannot substitute: it is lifetime, while `win_rate`
-- and `sharpe_ratio` are measured since STATS_EPOCH_MS.

BEGIN;

-- 1. Archive. INCLUDING ALL carries the column types and defaults across; the
--    archives deliberately take no indexes from migration 005, since a frozen
--    audit copy has no invariant to maintain.
CREATE TABLE IF NOT EXISTS paper_trades_archive (LIKE paper_trades INCLUDING DEFAULTS);
CREATE TABLE IF NOT EXISTS portfolio_archive (LIKE portfolio INCLUDING DEFAULTS);

INSERT INTO paper_trades_archive SELECT * FROM paper_trades;
INSERT INTO portfolio_archive SELECT * FROM portfolio;

-- 2. Refuse to clear anything unless every row arrived safely.
DO $$
DECLARE
    live_trades BIGINT;
    archived_trades BIGINT;
    live_portfolio BIGINT;
    archived_portfolio BIGINT;
BEGIN
    SELECT count(*) INTO live_trades FROM paper_trades;
    SELECT count(*) INTO archived_trades FROM paper_trades_archive;
    SELECT count(*) INTO live_portfolio FROM portfolio;
    SELECT count(*) INTO archived_portfolio FROM portfolio_archive;

    IF archived_trades < live_trades OR archived_portfolio < live_portfolio THEN
        RAISE EXCEPTION
            'archive incomplete: paper_trades %/%, portfolio %/% -- refusing to clear',
            archived_trades, live_trades, archived_portfolio, live_portfolio;
    END IF;
END $$;

-- 3. Clean slate. With paper_trades empty the next cycle recomputes
--    equity = 10000, cash = 10000, total_pnl = 0, total_trades = 0, and
--    sharpe_ratio / win_rate as NULL -- unknown rather than zero.
DELETE FROM paper_trades;
DELETE FROM portfolio;

-- 4. Make the persisted win_rate readable.
ALTER TABLE portfolio ADD COLUMN IF NOT EXISTS scored_trades INT;

COMMIT;

-- Re-run this to inspect what was preserved:
--   SELECT count(*), min(entry_time), max(entry_time) FROM paper_trades_archive;
--   SELECT count(*), min(timestamp), max(timestamp) FROM portfolio_archive;
