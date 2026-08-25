# The model-generation process was bleeding, and blind to it

Date: 2026-08-25
Evidence: live database (276 predictions, 432 model_metrics rows, the first
two paper trades), plus replayed hourly retrains on 2400 bars per symbol.

## The bleed, with a receipt

The first trade to close:

```
VET/USDT long   entry 0.005825   exit 0.005825   held 0.20h
                pnl -0.80        fees 0.80       exit_reason signal_flip
```

Entry and exit prices are **identical**. The position paid a full round trip
for zero price movement, twelve minutes after opening, because the model
changed its mind. That is not a losing trade; it is a trade that should never
have been closed.

Measured across every prediction in the database:

| symbol | predictions | side reversals | flip rate |
|---|---|---|---|
| DOGE/USDT | 27 | 11 | 42.3% |
| ETH/USDT | 27 | 10 | 38.5% |
| ADA/USDT | 27 | 10 | 38.5% |
| BTC/USDT | 27 | 10 | 38.5% |
| SOL/USDT | 27 | 8 | 30.8% |
| BNB/USDT | 27 | 6 | 23.1% |

**The model reverses its directional call on 29.9% of consecutive hours.** At
a 0.08% maker round trip, that is ~0.024% per hour of pure churn on every open
position — bleeding the account to pay for the model disagreeing with itself.

## Cause: only the last fold was ever served

`fit_walk_forward` looped the walk-forward splits refitting **the same
object**, so after the loop `self.model` held whichever fit the final fold
produced — trained on about five-sixths of the data, at a boundary that moves
by an hour on every run. The pipeline retrains from scratch hourly, so each
run served a materially different model.

Two further consequences of the same line:

* `self.scaler` was likewise the last fold's, so features were standardised
  against a different distribution than any other fold saw.
* The calibrator is fitted on the **pooled out-of-sample predictions of all
  folds**, then applied to one fold's model. Calibrating an ensemble's output
  distribution and serving a single member of it is a mismatch.

**Fix.** Every fold's `(scaler, model)` is kept, and `predict()` averages their
probabilities. That uses all the data, matches what the calibrator was fitted
on, and — being an average — has strictly lower variance.

Replayed against real hourly retrains, 24 consecutive hours per symbol:

| symbol | flip rate, last fold | flip rate, fold ensemble |
|---|---|---|
| DOGE/USDT | 26.1% | 8.7% |
| ADA/USDT | 17.4% | **0.0%** |
| BTC/USDT | 0.0% | 0.0% |
| ETH/USDT | 0.0% | 0.0% |
| **mean** | **10.9%** | **2.2%** |

Churn cost per open position per hour: 0.0087% → 0.0017%. A five-fold
reduction, from a change that adds no parameters and no assumptions.

## The promotion ladder has never been able to fire

`check_auto_upgrade()` gates on `edge_over_baseline`. `log_model_metrics()`
never wrote it. Neither did it write `up_rate`, `majority_baseline` or
`balanced_accuracy` — `skill_metrics()` computed all four every run and every
one was discarded.

```
skill_metrics computes:  up_rate, majority_baseline, balanced_accuracy, edge_over_baseline
log_model_metrics wrote: model_name, accuracy, precision_up, recall_up,
                         expected_value, sharpe_ratio, total_trades, win_rate
DISCARDED every run:     all four
```

So the ladder (logistic → neural → xgboost) is dead code, and always has been.
The active model only reads `xgboost_v1` because the gate falls through to
"whatever was logged last", which is self-perpetuating.

It also meant **432 rows of model history with no way to tell whether any model
was any good.** `accuracy 0.5301` is not a fact about skill without the
baseline beside it — TRX once scored 0.6502 against a 0.6413 majority
baseline, which reads as strong and is +0.89pp of noise.

## Two metrics in that table were actively lying

`sharpe_ratio` ran 1.40 to 10.66 across recent rows. It was
`sharpe_ratio(±1 per out-of-sample call)` annualised at √8760 — a quantity
with no trading meaning at all, presented in a column named for one. A model
carrying 1.5pp of edge was recording a Sharpe of 10.66.

`win_rate` was byte-identical to `accuracy` in every row, because a "win" was
defined as a correct call. Two columns, one number, both shaped like trading
results and neither being one.

Both are now dropped from the write. Trading results live in `portfolio`.

## What is recorded instead

`up_rate`, `majority_baseline`, `balanced_accuracy`, `edge_over_baseline`,
`calibration_error`, `oos_rows`, `symbol` — and **`roc_auc`**.

AUC earns its place: it is invariant under any monotone transform, so it
measures discrimination *before* calibration can flatten or inflate it. That
matters here specifically, because `docs/09` found the gate firing on isotonic
artifacts — accuracy and the old Sharpe both moved with that, and AUC did not.
It is the right quantity to gate promotion on.

Live check on BTC/USDT after the change:

```
accuracy 0.5296   baseline 0.5122   edge +0.0174   roc_auc 0.555
folds kept: 5
```

## Migration 007 is advisory, not blocking

The columns are diagnostic: without them `check_auto_upgrade()` degrades to
"keep the current model" and `log_model_metrics()` falls back to the legacy
subset with a warning. Neither stops the bot trading.

So the preflight **reports** them and continues, rather than failing the run.
A strict preflight on `scored_trades` already cost one red pipeline; stopping
the book to protect a diagnostic is the wrong trade. Load-bearing columns
(`target_move_1h`, `exit_reason`) stay strict.

Apply it when convenient:

```sql
-- migrations/007_honest_model_metrics.sql
```

Until then the pipeline runs, trades, and logs the legacy columns.

## What this does not fix

Nothing here creates edge. AUC is still ~0.53 and `docs/09`'s conclusion
stands: a little real discrimination, nowhere near enough to pay a round trip.
What changed is that the model no longer pays 0.024%/hour to argue with
itself, and there is finally a record of whether it is improving.
