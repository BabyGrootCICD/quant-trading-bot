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
    target_1h SMALLINT,
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
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL,
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
    status VARCHAR(10) DEFAULT 'open',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS portfolio (
    id BIGSERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    equity DOUBLE PRECISION NOT NULL,
    cash DOUBLE PRECISION NOT NULL,
    positions_value DOUBLE PRECISION NOT NULL,
    total_pnl DOUBLE PRECISION NOT NULL,
    sharpe_ratio DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    total_trades INT DEFAULT 0,
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
