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
| :07 | Fetch Hourly Data | Pull candles, compute features |
| :12 | Paper Trade | Fetch prices, predict, trade |
| Sun 3AM | Train Model | Retrain all 8 models |

## Quick Start

```bash
# Local development
python -m src.paper_trading.engine

# Manual trigger
gh workflow run "Paper Trade (Hourly)"
gh workflow run "Train Model (Weekly)"
```

## Database Tables

- `candles` — 1H OHLCV from BinanceUS
- `features` — log returns, RSI, MACD, Bollinger Bands
- `predictions` — model outputs per symbol
- `paper_trades` — open/closed positions with actual P&L
- `portfolio` — equity curve, total asset USD, win rate
- `model_metrics` — training evaluation results

## Configuration

Set in GitHub Secrets:
- `SUPABASE_URL` — Your Supabase project URL
- `SUPABASE_KEY` — Your Supabase anon/service key
- `EXCHANGE_ID` — Exchange ID (default: `binanceus`)
