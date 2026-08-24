# Expected-value allocation: from "never trades" to a real position sizer

Date: 2026-08-24
Evidence: `portfolio` rows 20-27, and 720 live hourly bars per symbol pulled
from Kraken.

## Symptom

Portfolio rows 25, 26 and 27 are byte-identical:

| id | equity | cash | positions | total_pnl | sharpe | win_rate | trades |
|---|---|---|---|---|---|---|---|
| 25 | 9976.35 | 9876.29 | 100.06 | -11.97 | 0 | 0 | 76 |
| 26 | 9976.35 | 9876.29 | 100.06 | -11.97 | NULL | 0 | 76 |
| 27 | 9976.35 | 9876.29 | 100.06 | -11.97 | NULL | 0 | 76 |

Hours apart, nothing changes. One trade has been recorded since the stats
epoch, and it lost. The previous round of fixes stopped the bleed by refusing
every trade; what it left behind was a bot that cannot lose because it cannot
act, holding two positions it will never release.

Four separate defects produce that picture.

## Defect 1 — the EV gate was a constant, so it refused everything

`EV = (2p - 1) * E|move| - cost`. The engine had a direction model and no
magnitude model, so `E|move|` came from `estimate_expected_moves()`: the mean
absolute hourly return over the last week of candles. **One number per symbol,
identical on every bar.** EV then reduces to a function of `p` alone, and since
the unconditional move is small on every pair in the universe, the answer was
"no" everywhere. On live data the unconditional gate admits **0 to 1.6%** of
bars, and on the pairs that matter, zero.

That is not a filter. A filter distinguishes.

**Fix — `src/models/magnitude.py`.** Direction is close to unforecastable at
1h; *magnitude* is not, because volatility clusters. An `MLPRegressor` over the
same feature set predicts `|next-hour log return|` per bar. Out of sample on
live data:

| symbol | rank IC | top-decile spread | unconditional move | break-even at unconditional |
|---|---|---|---|---|
| BTC/USDT | 0.181 | 1.47x | 0.216% | 119% |
| ETH/USDT | 0.223 | 1.76x | 0.284% | 103% |
| SOL/USDT | 0.212 | 2.11x | 0.332% | 95% |
| DOGE/USDT | 0.331 | 2.67x | 0.380% | 89% |

A 2x spread turns an impossible 119% break-even accuracy into a merely
demanding one. That is what makes the gate a selector rather than a wall.

Two conditioning details mattered more than the architecture:

* **The target had to be standardised.** Hourly absolute returns live around
  0.002, smaller than adam's default step, so the untouched network converged
  to predicting the mean -- reproducing the exact constant it exists to
  replace. Rank IC went from 0.02 to 0.18 on the same data once the label was
  scaled.
* **The forecast had to be recalibrated.** See Defect 2.

## Defect 2 — the gate selected the bars the model was most wrong about

The raw magnitude head is roughly unbiased in the middle of its range and
grossly optimistic at the top:

| symbol | top-5% predicted | realised | ratio |
|---|---|---|---|
| BTC/USDT | 0.958% | 0.310% | 0.32 |
| ETH/USDT | 1.847% | 0.553% | 0.30 |
| SOL/USDT | 0.776% | 0.548% | 0.71 |
| DOGE/USDT | 5.360% | 0.771% | **0.14** |

The EV gate only ever fires at the top of that range. Selection and bias point
the same way, so every trade the system takes would be priced off the head's
most inflated estimate -- the gate would admit not the volatile bars but the
bars the model is most wrong about, and report a confident EV for each.

**Fix — `MoveCalibrator`**, an isotonic map from predicted to realised move,
fitted on the walk-forward out-of-sample pairs. Top-decile ratio moves from
0.27-0.53 to ~1.0. Ranking is monotone-invariant, so the head's actual skill
survives untouched. Reported as `magnitude_tail_ratio_raw` and
`magnitude_tail_ratio` so the drift is visible.

The same argument applies to the *direction* head: `EV`, the Kelly fraction and
the allocation all read `probability_up` as a real probability, while
`LogisticModel` is fitted with `class_weight="balanced"`, which deliberately
distorts it. Measured calibration error was 0.14-0.22. `ProbabilityCalibrator`
now maps raw scores onto observed frequencies for every directional head.

### What the gate does after both corrections

| symbol | bars admitted, unconditional | bars admitted, conditional + calibrated |
|---|---|---|
| BTC/USDT | 0 / 669 | 1 (0.15%) |
| ETH/USDT | 0 / 667 | 0 (0.00%) |
| SOL/USDT | 2 / 664 | 7 (1.05%) |
| DOGE/USDT | 11 / 669 | 10 (1.49%) |

Roughly 1% of bars, across 8 symbols, is on the order of two trades a day.
That is the honest answer at a 0.30% round trip, and it is a real strategy
rather than a wall. Note that the raw (uncalibrated) head would have admitted
3-5x as many -- all of them mispriced.

## Defect 3 — positions had no exit, so equity froze

A position was closed only when its signal flipped. A position whose signal
never flipped was held forever, on a forecast whose horizon is one hour. That
is rows 25-27: two legacy positions, unbounded holding period, -11.97 of
unrealized loss going nowhere.

**Fix — `engine.exit_reason()`**: stop loss (default 1.5%), take profit
(default 2%), maximum holding period (default 6h), then signal flip, then no
signal. Risk exits are checked before the signal, because "the model still
likes it" is not a reason to sit through an unbounded drawdown. The reason is
written to `paper_trades.exit_reason` so the exit mix is auditable.

## Defect 4 — fees were charged twice

`close_open_positions()` writes `pnl = raw_pnl - fees` and `fees` into their
own columns. `update_portfolio()` then computed:

```python
cash_balance = cash - capital_locked + realized_pnl - realized_fees
```

`realized_pnl` already has the fees inside it. Every closed trade was billed
its round trip twice. In the live table: `total_pnl` -11.97 against cash
9876.29 with 100 of locked capital -- **$11.74 of phantom cost**, more than the
entire recorded loss. The reported equity was wrong, and so was the
`total_asset_usd` that sizing was derived from.

## Sizing: from an inverted rule to expected-value allocation

The old sizer was `percentage_based_size()`: `risk_budget / |estimated_change|`
capped at $100, where `estimated_change` was `(p - 0.5) * 2 * 0.02` -- a
fabricated linear map. It sized *inversely* to conviction, considered each
symbol in isolation, and ignored available cash entirely.

`src/strategy/allocation.py` replaces it with fractional Kelly on the model's
own EV and variance:

```
f_i = KELLY_SCALE * EV_i / sigma_i^2 ,   sigma_i = E|move|_i * sqrt(pi/2)
```

sized proportionally, scaled to fit the budget, capped per symbol, with
capacity freed by a cap redistributed to names still below theirs.

Which constraint binds is worth understanding. Kelly at a one-hour horizon is
enormous -- a 0.3% edge against a 1% move implies many times the account -- so
with a handful of candidates every one saturates `MAX_POSITION_FRAC` and they
come out equal-sized. That is correct, not a ranking failure. The EV ranking
expresses itself when the budget is scarce relative to the caps.

Defaults: 0.25 Kelly, 10% of equity per symbol, 60% gross exposure, $10
minimum. All overridable from repo variables.

## What this does and does not claim

It fixes four defects and makes the EV framework functional: the gate
discriminates, the forecasts it discriminates on are calibrated, capital is
allocated by edge per unit of risk, positions are released, and the books
balance.

It does **not** establish that the strategy is profitable. On the small
samples the gate admits (7-33 bars per symbol), directional hit rates land on
both sides of what break-even requires. The directional heads still carry
0.6-2.0pp of edge over the majority baseline -- close to noise -- and
Finding 1 of `STRATEGY_PLAN.md` is unchanged: a 0.30% round trip against a
0.22% median hourly move is a brutal hurdle. What has changed is that the
system now takes only the bars where that hurdle is clearable, priced on
numbers that are not inflated, and sizes them like a portfolio.

The `magnitude_tail_ratio` of ~1.0 is measured on the pairs the calibrator was
fitted to, so treat it as a correction that is directionally right rather than
as an out-of-sample guarantee; the pre-calibration figures are the honest
measure of how wrong the raw head was.

The levers that would actually change the economics remain the ones already
listed: a longer horizon, maker execution (`EXECUTION_MODE=maker` roughly
halves the cost term), and cost-aware labels.

## Migration

`migrations/004_ev_allocation.sql` adds `features.target_move_1h`, the four
new `predictions` columns, `paper_trades.exit_reason`, and the
`UNIQUE(symbol, timestamp)` index that the prediction upsert has always
depended on. **The schema preflight fails the pipeline until it is applied** --
paste it into the Supabase SQL editor, then:

```
python -m migrations.run_migrations --mark 004 004_ev_allocation
```
