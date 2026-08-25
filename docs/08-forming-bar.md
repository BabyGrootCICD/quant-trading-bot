# The traded prediction was computed from a bar that had not finished

Date: 2026-08-25
Evidence: hourly run #46, plus 1428 five-minute bars from OKX reconstructed
into 119 complete hourly bars.

## Summary

Exchanges return the bar **currently forming** as the last element of an OHLCV
response. The fetcher stored it like any other candle. Everything downstream
inherited it, and the row that becomes *the traded prediction* was computed
from a fraction of an hour's data.

The run that prompted this was **2.4 minutes into the bar** — the log's
`96% of it left` gives that away.

This is a genuine defect and it is now fixed. It is **not** why the bot is not
trading; see the last section before changing anything else.

## How it reached the traded prediction

Three links, each individually reasonable:

1. **`fetcher.fetch_and_store`** upserted every candle the exchange returned,
   including the one still forming.
2. **`engineer_features`** computed a feature row for it, from partial data.
3. **`model.predict()`** masks on NaN *features*, not on the NaN *target* — so
   that row was scored — and `upsert_latest_prediction` takes
   `valid.sort_values("timestamp").iloc[-1]`, the newest timestamp. The partial
   bar therefore became the prediction the engine trades.

## How wrong the inputs were

Reconstructing partial hourly bars from 5-minute data and scoring the final
row at each cut-off. Percentile is against the distribution of **completed**
bars — the only thing the model ever saw in training.

| as of | volume_ratio | volume_ratio_48 | log_return_1h | rsi_14 |
|---|---|---|---|---|
| 5 min | **0.0%** | **0.0%** | 54.9% | 71.8% |
| 10 min | **0.0%** | **0.0%** | 57.7% | 71.8% |
| 20 min | **0.0%** | 7.0% | 54.9% | 71.8% |
| 30 min | 2.8% | 11.3% | 67.6% | 73.2% |
| 45 min | 7.0% | 25.4% | 77.5% | 77.5% |
| 60 min (real) | 14.1% | 32.4% | 80.3% | 77.5% |

Raw values at 5 minutes: `volume_ratio` 0.0193 against a true 0.3313.

The damage is concentrated exactly where you would expect — in the features
that depend on the current bar. `rsi_14`, `atr_14_pct` and `bb_width` barely
move, because they are dominated by prior completed bars. But
`volume_ratio` and `volume_ratio_48` sit at the **zeroth percentile**: the
model is being asked to score a point from a region of feature space it has
never observed. A tree ensemble does not extrapolate there; it routes the
sample down whichever branch the extreme value happens to hit.

### It suppressed the magnitude forecast

Feeding the magnitude head a partial-bar vector (volume scaled to ~3 minutes'
worth) against the same bar complete:

| symbol | complete bar | partial bar | ratio |
|---|---|---|---|
| BTC/USDT | 0.341% | 0.253% | 0.74 |
| ADA/USDT | 0.592% | 0.458% | 0.77 |
| XRP/USDT | 1.147% | 1.147% | 1.00 |

The head has correctly learned "low volume precedes small moves", and a
2-minute-old bar always looks like low volume. So it forecast a quiet hour,
every hour, structurally.

That matches production, where the conditional forecast printed *below* the
unconditional average of the last 168 real bars for six of eight symbols:

| | BTC | ETH | BNB | SOL | XRP | ADA | DOGE | TRX |
|---|---|---|---|---|---|---|---|---|
| unconditional | 0.387% | 0.514% | 0.430% | 0.588% | 1.029% | 0.965% | 0.807% | 0.295% |
| head | 0.324% | 0.431% | 0.340% | 0.666% | 0.496% | 0.623% | 0.511% | 0.349% |
| ratio | 0.84 | 0.84 | 0.79 | 1.13 | **0.48** | **0.65** | **0.63** | 1.18 |

A conditional forecast should straddle the unconditional mean, not sit under
it. (The partial bar explains roughly 25pp of that gap; the rest is the
magnitude calibrator being anchored to a two-year distribution while the
current regime is more volatile — a separate issue, noted below.)

## The semantic error, which is worse than the noise

The label is `sign(return of the NEXT bar)`. Features at bar *T* forecast bar
*T+1*.

With the forming bar stored, the newest feature row was *T+1* (partial), so
its forecast was for ***T+2*** — a bar that had not started, and would not
start for up to an hour. Meanwhile `horizon_fraction()` discounted that
forecast by how much of the bar *in progress* remained.

**The two disagreed by a whole bar.** The engine was opening a position now,
priced against a move due to begin up to sixty minutes later, and discounting
it as though it were already underway.

## The fix

`drop_incomplete_bars()` in `src/data/fetcher.py`: store only bars whose
interval has fully elapsed. Fixing it at the source means training, features
and prediction all see the same invariant, rather than each guarding
separately.

After the fix the newest stored bar is the last closed bar *T*; its forecast
is for *T+1*; and *T+1* is the bar now forming — which is exactly what the
engine trades and exactly what `horizon_fraction()` measures. All three line
up.

The change is self-healing for already-stored partial rows: `resolve_since()`
re-fetches two bars of overlap, so a partial bar written before the fix is
overwritten with its completed values on the next run.

**`MAX_PREDICTION_AGE_HOURS` 2 → 3.** Predictions are now stamped with the
last complete bar, so a cycle running at :55 legitimately reads one 1.92h old,
and GitHub's scheduler adds 47–136 minutes of drift on top. At 2h the guard
would have rejected valid predictions and silently flattened the book — trading
one bug for a worse one.

## This does not make the bot trade, and must not be sold as if it does

| symbol | EV today | EV with the fix (1.35× move) | EV at double the move |
|---|---|---|---|
| BTC/USDT | −0.294% | −0.291% | −0.287% |
| SOL/USDT | −0.233% | −0.210% | −0.167% |
| TRX/USDT | −0.223% | −0.196% | −0.146% |
| **clears the gate** | **0/8** | **0/8** | **0/8** |

`EV = (2p − 1) × E|move| − cost`. The fix improves `E|move|`, which is the
term that was *not* binding. The edge multiplier `(2p − 1)` runs 0.02–0.22, so
at p = 0.52 the gate would need a **7.5% hourly move** to clear a 0.30% round
trip.

`docs/07` measured the edge on clean, complete bars and found ~1pp,
indistinguishable from zero at every horizon tested. That measurement stands —
it never touched the live pipeline. What this bug means is that the live system
was not even achieving that 1pp baseline, because it was scoring
out-of-distribution inputs.

So: two separate facts, both true.

* The pipeline had a real defect and now does not.
* The strategy still has no measurable alpha, and zero trades remains the
  correct output.

## Still open

**The magnitude calibrator is anchored to its training distribution.**
`MoveCalibrator` is isotonic, so its output is bounded by the range of realised
moves it was fitted against — the top bucket's fitted value is the *mean* of
that bucket. Fitted over two years while the current regime is more volatile,
it shrinks forecasts toward the historical average, which is part of the
remaining gap in the table above. Worth a rolling-window refit or an explicit
regime scalar. It is not urgent: `E|move|` is not the binding term.
