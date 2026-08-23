-- 002: Add actual_pnl_usd to paper_trades, total_asset_usd to portfolio
-- Applied: 2026-08-23

ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS actual_pnl_usd DOUBLE PRECISION DEFAULT 0;
ALTER TABLE portfolio ADD COLUMN IF NOT EXISTS total_asset_usd DOUBLE PRECISION;
