-- Phase 1-3: Core tables for quant trading bot

CREATE TABLE IF NOT EXISTS candles (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp BIGINT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_candles_symbol_time ON candles(symbol, timestamp DESC);

CREATE TABLE IF NOT EXISTS features (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp BIGINT NOT NULL,
    log_return_1h DOUBLE PRECISION,
    log_return_2h DOUBLE PRECISION,
    log_return_4h DOUBLE PRECISION,
    log_return_8h DOUBLE PRECISION,
    log_return_24h DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    bb_upper DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    volume_ratio DOUBLE PRECISION,
    rsi_7 DOUBLE PRECISION,
    macd_hist DOUBLE PRECISION,
    bb_width DOUBLE PRECISION,
    bb_position DOUBLE PRECISION,
    volume_ratio_48 DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION,
    atr_14_pct DOUBLE PRECISION,
    williams_r DOUBLE PRECISION,
    stoch_k DOUBLE PRECISION,
    stoch_d DOUBLE PRECISION,
    ha_trend DOUBLE PRECISION,
    close_pct_ma20 DOUBLE PRECISION,
    close_pct_ma50 DOUBLE PRECISION,
    vol_20 DOUBLE PRECISION,
    skew_20 DOUBLE PRECISION,
    target_1h SMALLINT,
    target_4h SMALLINT,
    target_move_1h DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp BIGINT NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    prediction DOUBLE PRECISION,
    probability_up DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    expected_move_pct DOUBLE PRECISION,
    expected_return_pct DOUBLE PRECISION,
    predicted_price DOUBLE PRECISION,
    horizon_hours INT DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol_time ON predictions(symbol, timestamp DESC);

CREATE TABLE IF NOT EXISTS paper_trades (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(6) NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_price DOUBLE PRECISION,
    size DOUBLE PRECISION NOT NULL,
    entry_time BIGINT NOT NULL,
    exit_time BIGINT,
    pnl DOUBLE PRECISION,
    fees DOUBLE PRECISION DEFAULT 0,
    slippage DOUBLE PRECISION DEFAULT 0,
    model_name VARCHAR(50),
    prediction_at_entry DOUBLE PRECISION,
    actual_pnl_usd DOUBLE PRECISION DEFAULT 0,
    exit_reason VARCHAR(24),
    status VARCHAR(10) DEFAULT 'open',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Mirrors migration 005. Without it a database bootstrapped from this file
-- silently lacks the guard, and the engine's duplicate-insert handler never
-- fires because the insert simply succeeds.
CREATE UNIQUE INDEX IF NOT EXISTS paper_trades_one_open_per_symbol
    ON paper_trades (symbol)
    WHERE status = 'open';

CREATE TABLE IF NOT EXISTS portfolio (
    id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    equity DOUBLE PRECISION NOT NULL,
    cash DOUBLE PRECISION NOT NULL,
    positions_value DOUBLE PRECISION NOT NULL,
    total_pnl DOUBLE PRECISION NOT NULL,
    sharpe_ratio DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    total_asset_usd DOUBLE PRECISION,
    total_trades INT DEFAULT 0,
    -- Denominator behind win_rate. total_trades is lifetime while win_rate and
    -- sharpe_ratio are post-epoch, so the row is unreadable without it.
    scored_trades INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    evaluated_at TIMESTAMPTZ DEFAULT NOW(),
    accuracy DOUBLE PRECISION,
    precision_up DOUBLE PRECISION,
    recall_up DOUBLE PRECISION,
    expected_value DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    total_trades INT,
    win_rate DOUBLE PRECISION,
    avg_pnl_per_trade DOUBLE PRECISION
);

-- Enable RLS on all tables
ALTER TABLE candles ENABLE ROW LEVEL SECURITY;
ALTER TABLE features ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE paper_trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_metrics ENABLE ROW LEVEL SECURITY;

-- Allow anon full access (for GHA cron jobs using the anon key)
CREATE POLICY "Allow anon all operations on candles" ON candles FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow anon all operations on features" ON features FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow anon all operations on predictions" ON predictions FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow anon all operations on paper_trades" ON paper_trades FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow anon all operations on portfolio" ON portfolio FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow anon all operations on model_metrics" ON model_metrics FOR ALL USING (true) WITH CHECK (true);
