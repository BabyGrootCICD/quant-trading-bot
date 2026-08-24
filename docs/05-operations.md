# Operations

## The pipeline

`.github/workflows/hourly_pipeline.yml` runs hourly and on manual dispatch:

```
Schema preflight -> Fetch OHLCV -> Compute features -> Train model -> Paper trade
```

Each stage fails the run rather than passing bad state downstream. Trigger
manually with:

```bash
gh workflow run hourly_pipeline.yml --ref main
```

## The risk monitor

`.github/workflows/risk_monitor.yml` runs every 15 minutes and does one thing:
mark open positions and close the ones that hit a stop, a target or the
holding limit. It never opens a position, never reads a prediction, and never
trains, so it cannot act on stale state or disagree with the hourly cycle
about direction — `risk_exit_reason()` is the only rule it consults, imported
from the engine rather than reimplemented.

It installs four packages instead of the full requirements file, which is what
keeps it near a minute against the pipeline's seven.

```bash
gh workflow run risk_monitor.yml --ref main
python -m src.paper_trading.risk_monitor      # locally
```

Why it exists: against a `0 * * * *` cron, GitHub delivered the hourly
pipeline with gaps of 47 to 136 minutes. A stop that is only consulted when
the pipeline runs is a stop with an unbounded delay.

It writes a `portfolio` row only when something actually closed — otherwise a
15-minute cadence would bury the hourly cycle's rows under identical ones.

## Why the pipeline is not run more often

The data is hourly: 1h candles, a next-1h label. Six runs an hour re-derive
one prediction from one closed bar, and the newest candle in the table is the
*forming* bar, so running earlier in the hour feeds the model a feature vector
(volume ratio especially) further off the distribution it was trained on.

Separately, a prediction is only collectable over what is left of the bar it
was made for. `horizon_fraction()` scales the forecast by that remainder and
`MIN_HORIZON_FRACTION` (default 0.25) refuses entries in the last quarter
outright.

## Migrations are manual

This repo has no service-role database connection — only the anon REST key —
so `migrations/run_migrations.py` prints the pending SQL rather than applying
it.

1. `python -m migrations.run_migrations` to list what is pending
2. Paste the `.sql` file into **Supabase Dashboard -> SQL Editor -> Run**
3. Optionally mark it: `python -m migrations.run_migrations --mark 003 add_enhanced_features`

If a migration has not been applied, the **schema preflight fails the run** and
names the missing columns. That is by design — see
[04](04-silent-failures.md).

Migration `003` contains a `DELETE FROM predictions` that removes older
duplicate `(symbol, timestamp)` rows before creating a unique index. Inspect
first if you want:

```sql
SELECT symbol, timestamp, count(*) FROM predictions
GROUP BY symbol, timestamp HAVING count(*) > 1;
```

If that returns nothing, the delete is a no-op.

## Reading the paper-trade output

```
Round-trip cost: 0.30% of position

[1b/6] Estimating expected moves...
    BTC/USDT: E|move|=0.354% -> needs 92.4% accuracy
    TRX/USDT: E|move|=0.283% -> needs 102.9% accuracy  UNREACHABLE
```

`UNREACHABLE` means that symbol's typical move is smaller than the round-trip
cost, so no accuracy can make it profitable at this horizon.

```
  Skipped BTC/USDT: EV -0.244% (strength 0.58, E|move| 0.354%, needs 92.3% accuracy)
```

The EV gate refusing a trade. **This is the expected steady state** — see
[02](02-trade-economics.md). A run that opens no positions is working
correctly.

```
  Holding long ADA/USDT (signal unchanged)
```

The position is kept rather than closed and reopened, which would pay the
round trip again for no change in exposure.

```
  STALE: BTC/USDT prediction is 16533.2h old (max 2h) - ignoring
```

The trainer did not write a fresh prediction. Check the training step — this
means the engine is being protected from something upstream being broken.

```
  Total Trades: 75 (scored since epoch: 0)
  No trades closed since the stats epoch yet -- sharpe/win_rate stay 0 until
  the first post-fix trade closes.
```

`sharpe_ratio` and `win_rate` of 0 with zero scored trades is not a failure;
there is simply nothing to measure yet.

## Reading the training output

```
  Accuracy: 0.5211 (baseline 0.5048, edge 0.0163)
  Balanced accuracy: 0.5212
```

**`edge` is the number that matters** — accuracy over the majority-class
baseline. Raw accuracy is meaningless on an imbalanced label; TRX once
reported 0.6502 against a 0.6413 baseline. Anything under ~2pp is noise.

`balanced_accuracy` is the mean of per-class recall and is immune to class
skew; a large gap between it and raw accuracy signals imbalance.

The per-symbol `Sharpe` line in training output is computed on +-1 unit
outcomes and is *not* comparable to the portfolio Sharpe. Treat `edge` as the
authority.

## Configuration

| setting | location | purpose |
|---|---|---|
| `MIN_EDGE_MARGIN` | env, default 1.0 | multiple of round-trip cost an edge must clear |
| `STATS_EPOCH_MS` | env / `config/settings.py` | start of the statistics window |
| `MAX_PREDICTION_AGE_HOURS` | `engine.py`, 2 | staleness cutoff |
| `MAX_TRAINING_ROWS` | `trainer.py`, 20000 | history pulled per symbol |
| `MIN_PROMOTION_EDGE` | `trainer.py`, 0.02 | edge needed to promote a model |
| `TAKER_FEE`, `SLIPPAGE_BPS` | `economics.py` | the cost model |

`TAKER_FEE` and `SLIPPAGE_BPS` decide what the gate allows. They should reflect
real execution, not optimism — lowering them to unlock trades re-creates the
original loss.

## Tests

```bash
pytest              # 80 tests, config in pyproject.toml
```

They run in CI on tagged releases (`release.yml`). They do not currently run on
the hourly pipeline.

## Known constraints

- **Migrations require a human.** No service-role connection exists.
- **No demonstrated edge.** The models sit within ~2pp of their majority
  baselines. The gate correctly refuses to trade on that, so the bot will
  normally hold no positions.
- **First feature run after a schema change is slow** (~6 min): it rewrites all
  ~125k rows. Subsequent fetches are incremental (~4 candles/symbol).
- **`total_pnl` is lifetime; `sharpe_ratio`/`win_rate` start at the stats
  epoch.** They are deliberately different scopes — see
  [03](03-metrics-integrity.md).
