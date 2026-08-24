# The pipeline outage

**Period:** roughly 2026-08-23 11:00Z to 2026-08-24 00:25Z
**Symptom:** `total_pnl` -0.05 -> -9.58, `win_rate` ~0.35, exactly 4 open
positions worth $400 every hour, `sharpe_ratio` between -8 and -49.
**Status:** fixed. Evidence from Actions runs `32673592721`, `32670662532`,
`32661221888`.

Every stage of the workflow reported success throughout.

## The failure chain

Four independent breaks, each hiding the next.

### Link 1 — feature writes failed 8/8, every hour

```
ERROR processing BTC/USDT: {'message': "Could not find the 'atr_14' column of
'features' in the schema cache", 'code': 'PGRST204'}
... (x8)
Done. Total feature rows: 0
```

Commit `1d1ec52` added 16 feature columns to `engineer.py` — `rsi_7`,
`macd_hist`, `bb_width`, `bb_position`, `volume_ratio_48`, `atr_14`,
`atr_14_pct`, `williams_r`, `stoch_k`, `stoch_d`, `ha_trend`,
`close_pct_ma20`, `close_pct_ma50`, `vol_20`, `skew_20`, `target_4h` — but no
migration was ever written for them. PostgREST rejects the whole upsert when
one column is unknown, so the `features` table was frozen from that commit
onward.

**Fix:** `migrations/003_add_enhanced_features.sql`, plus a schema preflight
step (see [04](04-silent-failures.md)).

### Link 2 — training failed 8/8, every hour

```
Active model: xgboost_v1
Training on BTC/USDT...
  ERROR: 'XGBoostModel' object has no attribute 'fit_walk_forward'
... (x8)
No models trained successfully.
```

`trainer.train_walk_forward()` calls `model.fit_walk_forward()`, which existed
only on `LogisticModel`. `check_auto_upgrade()` had already selected
`xgboost_v1`, so every symbol raised `AttributeError` inside a blanket
`except Exception`. The step exited 0.

The logistic path was broken too: `fit_walk_forward()` never set
`self.is_fitted`, so the following `predict()` would have raised
`RuntimeError("Model not fitted")`.

**Fix:** implemented `XGBoostModel.fit_walk_forward()` against the same
contract; set `is_fitted` in the logistic version; unified both models onto one
`FEATURE_COLS` list in `src/models/features.py` (xgboost was still on a stale
11-column list while the engineer emitted 26).

### Link 3 — nothing ever wrote the `predictions` table

`grep -rn predictions src/` found exactly one writer:
`engine.log_predictions()` — **dead code, never called from `main()`**. It
would have failed anyway, since `predictions` had no
`UNIQUE(symbol, timestamp)` for its `on_conflict` target.

The trainer computed `model.predict(df)` and discarded the result.

**Fix:** the trainer now persists the newest prediction per symbol; the dead
function was deleted; migration 003 adds the unique index.

### Link 4 — the engine replayed one frozen prediction set

`fetch_latest_predictions()` took `ORDER BY timestamp DESC LIMIT 1` per symbol
with no freshness check. The same four signals appeared in every run at 19:26Z,
22:27Z and 23:26Z:

```
BNB/USDT: BUY  (P(up)=0.60)
XRP/USDT: BUY  (P(up)=0.86)
DOGE/USDT: SELL (P(up)=0.35)
TRX/USDT: BUY  (P(up)=0.58)
```

Each hour the engine closed all four and reopened the identical four at $100
each.

**Fix:** predictions older than `MAX_PREDICTION_AGE_HOURS` (2h) are rejected;
positions whose signal is unchanged are held rather than round-tripped.

## Why that produced a steady loss

A round trip costs `2 x (0.10% taker + 5bps slippage) = 0.30%` of position
size — $0.30 per $100. Four positions per hour is **$1.20/hour of guaranteed
cost against a signal carrying exactly zero information**, because it was a
constant. Over 75 trades that is roughly the observed -$9.58.

The 35% win rate and the monotonic decline are precisely what a constant signal
churned hourly through a 0.30% spread produces.

## A hypothesis that was wrong

The initial suspicion was that the fetch stage was inserting duplicate candles
and corrupting the database. **It was not.** `candles` has
`UNIQUE(symbol, timestamp)` and the fetcher uses
`upsert(on_conflict="symbol,timestamp")`.

It *was* wasteful: it re-downloaded and re-upserted the full 2-year window —
17,520 candles x 8 symbols, ~140k rows — every hour. Now it resumes from the
newest stored candle, with a 2-bar overlap so the still-forming candle gets
corrected. First run after the fix: **17,520 -> 4 candles** per symbol.

## Two further bugs the first green run exposed

Fixing the chain revealed problems that had been invisible underneath it.

**The trainer was reading the oldest 1000 rows.** `fetch_features()` was an
unpaginated `.select("*")` ordered *ascending*. PostgREST caps a response at
1000 rows, so the trainer silently received the oldest 1000 feature rows per
symbol — it trained on two-year-old data and stamped its prediction with a
two-year-old timestamp:

```
STALE: BTC/USDT prediction is 16533.2h old (max 2h) - ignoring
```

The staleness guard from Link 4 caught it. Fixed by paginating explicitly,
newest-first, capped at `MAX_TRAINING_ROWS`, then sorting chronologically for
the walk-forward split. Signals per symbol went from 950 to ~14,500.

**The model was grading its own homework** — see
[03 Metrics integrity](03-metrics-integrity.md).

## Verification

| Stage | Before | After |
|---|---|---|
| Schema preflight | did not exist | passes |
| Fetch | 17,520 candles x 8, hourly | 4 candles x 8 |
| Features | `Total feature rows: 0` | 125,849 rows |
| Train | `ERROR ... 8/8` | 8/8 trained |
| Paper trade | churned 4 stale positions | refuses to trade (see [02](02-trade-economics.md)) |
