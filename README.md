# Quant Trading Bot

Hourly crypto trading bot using ML predictions on BinanceUS, scheduled via GitHub Actions with Supabase storage.

## Strategy

- **Model**: Logistic Regression → auto-upgrade to XGBoost when Sharpe > 1 for 7 days
- **Assets**: BTC, ETH, BNB, SOL, XRP, ADA, DOGE, TRX (all /USDT)
- **Timeframe**: 1H hourly candles
- **Logic**: Predict P(up) for next hour → BUY if P(up) > 0.55, SELL if P(up) < 0.45
- **Sizing**: Percentage-based — position size scales with estimated change magnitude

## Dashboard Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| `total_asset_usd` | Real USD value of all assets | `portfolio.total_asset_usd` |
| `win_rate` | % of profitable closed trades | `portfolio.win_rate` |
| `actual_pnl_usd` | Real USD gain/loss per trade | `paper_trades.actual_pnl_usd` |
| `total_pnl` | Cumulative P&L in USD | `portfolio.total_pnl` |
| `sharpe_ratio` | Risk-adjusted return | `portfolio.sharpe_ratio` |

## Schedule (UTC)

| Time | Workflow | Action |
|------|----------|--------|
| Hourly (:00) | `hourly_pipeline.yml` | Fetch candles → compute features → train models → paper trade (single run) |
| Sun 6AM (UTC) | `quantum_optimize.yml` | Portfolio opt, risk analysis, kernel comparison |

> Fetch, feature computation, training, and paper trading all run together in the hourly pipeline; there is no separate weekly training workflow. Whether training *should* run hourly vs weekly is tracked in #2.

## Quick Start

```bash
# Local development
python -m src.paper_trading.engine

# Manual trigger
gh workflow run "Paper Trade (Hourly)"
gh workflow run "Train Model (Weekly)"
gh workflow run "Quantum Optimization (Weekly)"
```

## Database Tables

- `candles` — 1H OHLCV from BinanceUS
- `features` — log returns, RSI, MACD, Bollinger Bands
- `predictions` — model outputs per symbol
- `paper_trades` — open/closed positions with actual P&L
- `portfolio` — equity curve, total asset USD, win rate
- `model_metrics` — training evaluation results

## Quantum Integration

Three quantum modules for when the classical system scales past 50+ assets:

### Portfolio Optimization (QAOA)
- Selects optimal subset of assets under budget/cardinality constraints
- Maps mean-variance optimization to QUBO → solves with QAOA
- Compares quantum vs classical (NumPy exact solver) results

### Monte Carlo Risk Simulation (Amplitude Estimation)
- Estimates VaR and CVaR across 1000+ simulated price paths
- Quadratic speedup over classical Monte Carlo
- Provides worst-case/best-case scenarios

### Pattern Recognition (Quantum Kernel)
- Quantum SVM for market regime classification
- Compares quantum kernel vs classical RBF kernel
- Identifies quantum advantage in high-dimensional feature spaces

### Configuration

Set in GitHub Secrets:
- `IBM_QUANTUM_TOKEN` — IBM Quantum API key
- `IBM_QUANTUM_INSTANCE` — IBM Quantum instance CRN

### When to Use Quantum

| Condition | Use Case |
|-----------|----------|
| > 50 assets | Portfolio optimization |
| > 5 hard constraints | QUBO formulation |
| > 1000 Monte Carlo paths | Amplitude estimation |
| > 50 features | Quantum kernel methods |

## Configuration

Set in GitHub Secrets:
- `SUPABASE_URL` — Your Supabase project URL
- `SUPABASE_KEY` — Your Supabase anon/service key
- `EXCHANGE_ID` — Exchange ID (default: `binanceus`)
- `IBM_QUANTUM_TOKEN` — IBM Quantum API key (optional)
- `IBM_QUANTUM_INSTANCE` — IBM Quantum instance CRN (optional)
