# Metrics integrity

Every performance number this system reported was wrong, in four independent
ways. Each is worth understanding separately, because each has a different
failure mode.

## 1. The model was grading its own homework

`train_walk_forward()` ran walk-forward CV, then scored the result by calling
`model.predict(df)` over the **entire** feature frame — most of which the final
fold had trained on.

Reported for BTC/USDT:

```
test_accuracy: 0.5278      <- honest, out-of-sample
win_rate:      0.9053      <- in-sample
sharpe:      129.4534      <- in-sample
```

A 0.91 win rate against a 0.52 accuracy is internally contradictory; the two
numbers were measuring different things.

This mattered beyond reporting: `check_auto_upgrade()` promotes models on
`sharpe_ratio > 1`, so the inflated figure was driving model selection.

**Fix.** Both models now return their out-of-sample fold predictions
(`oos_preds`, `oos_y_true`) and the trainer scores on those alone.

Reproduced on a near-noise synthetic set:

```
test_accuracy (honest):        0.5532
win_rate from OOS:             0.5532   <- now matches
win_rate in-sample (old bug):  0.8865
```

## 2. Raw accuracy hid class imbalance

TRX/USDT reported `accuracy 0.6502, sharpe 29.47`, which reads as a strong
model. It was not.

**29.63% of TRX hourly bars have exactly zero return** — tick quantization on a
low-priced, thin pair. The label `target_1h = (next_return > 0)` scored all of
them as "down", giving an up-rate of 0.3587. Always answering "down" therefore
scores **0.6413**.

| symbol | up rate | majority baseline | model | edge over baseline |
|---|---|---|---|---|
| BTC/USDT | 0.5100 | 0.5100 | 0.5155 | +0.55pp |
| ETH/USDT | 0.5060 | 0.5060 | 0.5236 | +1.76pp |
| ADA/USDT | 0.4739 | 0.5261 | 0.5156 | **-1.05pp** |
| DOGE/USDT | 0.5040 | 0.5040 | 0.5208 | +1.68pp |
| TRX/USDT | 0.3587 | 0.6413 | 0.6502 | +0.89pp |

Every model sits within ~2pp of simply predicting the majority class, and ADA
is *worse* than it.

**Fix.** Flat bars are left unlabelled (NaN) rather than scored as "down", so
training drops them. The trainer reports `majority_baseline`,
`edge_over_baseline` and `balanced_accuracy` alongside raw accuracy, and
promotion gates on edge over baseline instead of Sharpe.

After the fix, TRX's baseline fell **0.6367 -> 0.5071** and its signal count
dropped 2450 -> 1700.

## 3. Sharpe was computed three ways wrong

```python
sr = sharpe_ratio(closed_pnl[-168:])   # the old call
```

- **Dollar PnL, not returns.** The ratio scaled with position size, so the same
  strategy at $100 and $1000 reported different Sharpes.
- **Wrong annualization.** `sqrt(8760)` assumes one observation per hour; the
  bot opened one position *per symbol* per hour, up to 8.
- **The wrong 168 trades.** The query returns newest-first, so `[-168:]` sliced
  the **oldest** rows, not the most recent.

A fourth issue surfaced during the fix: `if std == 0` is too strict. A constant
series of, say, -0.0015 leaves a floating-point residue near 1e-19, and
`mean/std` then explodes. A regression test caught this producing a Sharpe of
**-3.2e17** — precisely the shape of this bot's PnL when a fixed fee is the
only thing moving. The guard is now relative, not exact-zero.

**Fix.** `compute_sharpe()` converts PnL to fractional returns on capital at
risk, annualizes by the observed trade frequency via `annualization_factor()`,
takes the newest `window` trades from the head, and treats near-zero variance
as degenerate.

## 4. Legacy trades contaminated everything

After the pipeline was fixed, `sharpe_ratio` stayed at -44.97 and `win_rate` at
0.3467 — because all 75 closed trades were made during the outage, when the
engine replayed a frozen prediction set. The Sharpe window is 168 trades deep,
so those rows would have dominated for days.

**Fix.** `STATS_EPOCH_MS` in `config/settings.py` marks the first cycle that
traded on a fresh prediction (2026-08-24T00:25:00Z). Statistics start there.

**An important boundary.** The epoch applies only to *quality* metrics:

| field | scope | why |
|---|---|---|
| `total_pnl` | lifetime | accounting fact; must reconcile with `equity` |
| `total_trades` | lifetime | same |
| `sharpe_ratio` | since epoch | measures a strategy those trades weren't running |
| `win_rate` | since epoch | same |

The first attempt applied the epoch to everything, which produced
`total_pnl 0` next to `equity 9978.93` — reading as "no losses" on an account
down $21. Accounting facts and quality measures need different treatment.

The engine now also reports `Total Trades: 75 (scored since epoch: 0)` so the
two scopes are visible side by side.

## 5. "Unknown" was being written as 0.0

The first version of the epoch change wrote `sharpe_ratio = 0` and
`win_rate = 0` whenever there was not enough data to compute them. That is a
different claim: 0.0 means "no risk-adjusted return", not "not enough trades
yet". Rows 21-25 of the `portfolio` table all read `0`, which looks
indistinguishable from a broken metric.

It also would not have resolved on its own. `sharpe_ratio()` needs
`MIN_SHARPE_TRADES` (30) observations, and the EV gate correctly blocks nearly
every trade — so 30 closed post-epoch trades may never accumulate, and that
column would have read `0.0000` indefinitely.

**Fix.** Both columns are nullable, so undefined metrics are written as SQL
`NULL`, and the log distinguishes the cases:

```
Sharpe: n/a (1/30 scored trades needed)
Win Rate: n/a (no trades closed since the stats epoch)
```

A `NULL` in those columns means "not computable yet". A `0` means zero.

## A worked example: when a real edge still isn't alpha

Once flat bars were properly excluded, TRX reported a **+12.18pp** edge with
balanced accuracy 0.6294 — an order of magnitude above every other symbol.

Checked rather than banked:

```
sign autocorrelation (lag 1, non-flat bars): -0.2256
reversal rate:                                61.3%
```

The model had learned "predict the opposite of the last bar" — tick-level mean
reversion on a thin, quantized pair. 61.3% is almost exactly the observed
balanced accuracy. The edge is real and it is not a forecast.

It is also untradeable. On non-flat bars TRX's `E|move|` is 0.319%, so
break-even accuracy is **97.0%** and EV at the reported 0.6288 accuracy is
**-0.218% per trade**. The gate refuses it.

**The lesson:** an unusually good number is a prompt to investigate, not to
celebrate. Both times a model here looked strong, it was an artifact.
