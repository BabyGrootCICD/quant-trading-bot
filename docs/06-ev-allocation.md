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


---

# Addendum, 2026-08-24 evening: the gate was never reached

## The bot went idle, and the calibration patch is why

`portfolio` rows 28-34: equity 9988.22, positions 0, `total_pnl` -11.78,
`total_trades` frozen at 77 for six hours. Three things in that picture are
working as designed and one is not.

Working: `sharpe_ratio` NULL is correct (2 scored trades, needs 30 -- `0.0`
would be a claim rather than an absence). `win_rate` 0.5 is real, and is
literally 1 win / 2 trades since the epoch. Training is fine -- 8/8 symbols
every hour, magnitude head rank IC 0.15-0.35, top-decile spread x1.7-2.1,
calibration error 0.05-0.11 -> 0.00. And the equity step from 9976.18 to
9988.22 with no trades was the fee double-count fix landing: 9988.22 is
exactly 10000 - 11.78.

Not working: `Active signals: 0` or `1`, out of 8, hour after hour.

The cause is the calibration added earlier in this document. Mapping raw
scores onto observed frequencies is correct, and the consequence is that
`probability_up` now sits in a narrow band around 0.50, because the models
genuinely carry about 1.5pp of edge. The `0.55 / 0.45` threshold in
`generate_signals` was chosen when the heads emitted uncalibrated,
overconfident probabilities. Against calibrated ones it is close to absolute.

So the funnel was **8 symbols -> 0-1 past the probability threshold -> 0 past
the EV gate**. The gate this document is about was never reached.

A fixed probability threshold is also the wrong instrument in a system that
has an EV gate. It judges `p` alone and cannot see forecast move size, so it
blocks p=0.53 on a 2% bar while passing p=0.60 on a 0.3% one -- the same
mistake as the constant `E|move|`, in the other direction.

**Fix.** Thresholds to 0.50, so direction is `sign(p - 0.5)` and
`is_tradeable()` is the only entry gate. Churn is held off by
`SIGNAL_EXIT_BAND` (default 0.02) on the *exit* side instead: an open long is
closed on signal grounds only once `p < 0.48`. Without that, entry at 0.50
would reverse the whole book on a probability drifting across the midpoint --
the original bleed, rebuilt.

On the live probabilities from run 32730166876, candidates reaching the EV
gate go from 2 to 8. All 8 are still refused, and that is the honest answer:

| symbol | p | E&#124;move&#124; | EV | needs |
|---|---|---|---|---|
| XRP/USDT | 0.545 | 1.000% | -0.210% | 65.0% |
| DOGE/USDT | 0.523 | 0.760% | -0.265% | 69.7% |
| SOL/USDT | 0.497 | 0.570% | -0.297% | 76.3% |
| TRX/USDT | 0.230 | 0.290% | -0.143% | 101.7% |

What changed is that every symbol is now *evaluated and logged* each hour, and
a volatility spike on any of them can carry it through. The frontier:

| forecast E&#124;move&#124; | calibrated P(up) needed |
|---|---|
| 0.5% | 0.800 |
| 1.0% | 0.650 |
| 2.0% | 0.575 |
| 3.0% | 0.550 |

Trades will be rare. That is what a 0.30% round trip against a sub-1% hourly
move costs, and no amount of threshold tuning changes it -- only a longer
horizon or maker execution does.

## Reconciliation: one pass, decided before any write

The close pass and the open pass each decided independently, with a re-read of
`paper_trades` between them, so no single place knew what the cycle intended.
Three defects followed.

**Flat and silent were the same thing.** `desired_sides()` was fed
`active_signals`, so a symbol the model called flat produced no dict key --
identical to one it never scored. Both closed the position, both recorded
`no_signal`. `desired_side_by_symbol()` now takes the full frame and returns
`None` for a flat call, absent for silence, and the two get different
treatment: flat closes (`flat_signal`), silence *preserves*, bounded by the
stop and `MAX_HOLDING_HOURS`. Flattening on silence would pay a round trip on
every open position for a transient trainer or database failure.

**Duplicate open rows were resolved by luck.** `desired_sides_from_trades()`
was a dict comprehension over a query with no `.order(...)`, so with two open
rows for one symbol the surviving side was not deterministic -- and two
opposite-side rows could pass the `held` guard and open a third position.
`open_by_symbol()` now returns duplicated symbols separately and the cycle
fails closed on them. Migration 005's partial unique index remains the real
guard; this is defense for databases provisioned from `src/data/schema.sql`,
which had drifted and carried no such index (now fixed, with a contract test).

**One failed close aborted the hour.** The `.update()` was bare inside a loop,
so a failure propagated out of `main()` and steps [5/6] and [6/6] never ran --
a hole in the equity curve rather than a wrong value in it. Failures are now
contained per symbol, the portfolio row is always written, and the cycle exits
non-zero afterwards so the step goes red instead of reporting green. The open
path's `except Exception` was the mirror image: it reported auth, schema and
network failures as "likely concurrent run" and swallowed them.
`is_duplicate_open_error()` now separates a lost race from a real error.

Reads of `paper_trades` per cycle: **4 -> 2**. Post-close cash, exposure and
the final open book are derived from writes this process performed and
confirmed. That is not just cheaper -- a re-read picks up a concurrent run's
rows, so the allocator could size against cash another run had already
changed, and the portfolio row could describe a book this cycle did not
create.

### exit_reason vocabulary

`stop_loss`, `take_profit`, `max_holding`, `signal_flip`, `flat_signal`.
Validated against `EXIT_REASONS` before every write, because the column is
`VARCHAR(24)` and a longer label truncates silently. **`no_signal` is legacy**
-- historical rows carry it, and it may mean any of "model said flat", "model
said nothing", or "no predictions at all".

## Reading the portfolio row

`scored_trades` is now persisted. It was computed and printed but never
stored, which is why a `win_rate` of 0.5 was unreadable: `total_trades` is
lifetime while `win_rate` and `sharpe_ratio` are measured since
`STATS_EPOCH_MS`, so the row carried no denominator for its own headline
number. 0.5 over two trades and 0.5 over two hundred looked identical.

## Migration 006: archive and restart

Clearing `portfolio` alone does nothing -- every row is recomputed from
`paper_trades` each cycle (`cash = INITIAL_CASH - locked + realized_pnl`), so
the same equity returns within the hour. `006_archive_and_reset.sql` copies
both tables to `*_archive`, verifies the counts, clears the originals, and
adds `portfolio.scored_trades`. The 77 broken-era trades that docs 01-04 are
written against are preserved, and the next cycle recomputes a clean
$10,000.


---

# Addendum 2: the forecast horizon, and where exits belong

## The question that started it: run the pipeline every 10 minutes?

No, and the 8-minutes-of-runtime-versus-a-10-minute-gap margin is not the
reason. Three things, in increasing order of importance.

**The margin is thinner than it looks.** Over the last 20 scheduled runs:
median 7.1 min, **max 12.9 min** — already past a 10-minute gap. And
`concurrency: cancel-in-progress: false` queues exactly one pending run;
further ones are cancelled, so overlap silently *drops* cycles rather than
delaying them.

**GitHub does not honour the interval anyway.** Against a `0 * * * *` cron:

```
gap between scheduled runs:  min 47m   median 63m   max 136m
```

Deliveries landed at 14:48, 13:00, 11:34, 10:41, 09:54 — nowhere near `:00` —
and the 136-minute gap is an hour skipped outright. A `*/10` cron buys
delivery *attempts*, not deliveries.

**The data is hourly.** 1h candles, a next-1h label. Six runs an hour
re-derive one prediction from one closed bar. Worse, the newest candle the
fetcher stores is the *forming* bar — checked live, 46% complete at run time —
so features like `volume_ratio`, computed from partial volume, drift further
off the training distribution the earlier in the bar you run.

## The real defect it surfaced

The EV gate priced every candidate at its **full-bar** expected move. But the
forecast is for the next bar, and the pipeline does not enter at the top of
it. Volatility scales with √t, so a part-bar holding period collects only
part of the move — while the round trip is paid in full regardless.

A setup clearing the gate by +0.100% at the open:

| minutes into bar | collectable E&#124;move&#124; | realised EV |
|---|---|---|
| 0 | 2.000% | **+0.100%** |
| 20 | 1.633% | +0.027% |
| 30 | 1.414% | −0.017% |
| 50 | 0.816% | −0.137% |

With runs landing at 14:48 and 11:34, the gate was routinely approving trades
whose EV it could not collect. This was live at hourly cadence; a 10-minute
schedule would only have multiplied it.

**Fix** — `horizon_fraction()` and `collectable_move()` in
`src/strategy/economics.py`. `build_candidates` discounts the forecast by
`√(fraction of bar remaining)` before both the gate and the allocator, and
`MIN_HORIZON_FRACTION` (default 0.25) declines outright in the last quarter,
where the remaining-move estimate is microstructure rather than model.

### One counter-intuitive property, pinned in tests

Shortening the horizon *raises* the raw Kelly fraction. Edge falls as √f while
variance falls as f, so edge/variance scales as 1/√f — a shorter bet is less
variable, and Kelly asks for more of it:

| minutes into bar | EV | raw Kelly | after the 10% cap |
|---|---|---|---|
| 0 | 0.700% | 2.785 | 0.100 |
| 30 | 0.407% | 3.240 | 0.100 |
| 40 | 0.277% | 3.311 | 0.100 |

So the horizon discount must **not** be relied on to shrink positions. The
protection is the gate refusing the trade and the per-symbol cap bounding it —
never the sizer trimming it. `test_shortening_the_horizon_raises_raw_kelly_not_lowers_it`
exists so a future "fix" does not quietly remove the cap on the assumption
that sizing handles it.

## Exits moved to their own workflow

A stop is about where price is *now*, which changes continuously — unlike the
model, which cannot learn anything new until a bar closes. Yet a stop was only
consulted when the full pipeline ran, i.e. every 47 to 136 minutes.

`src/paper_trading/risk_monitor.py` + `.github/workflows/risk_monitor.yml` run
every 15 minutes and evaluate stops, targets and holding limits only. The
narrowness is the safety property, and it is enforced by tests rather than by
convention:

* it never inserts a trade (asserted via AST, because a grep for `.insert(`
  matches the `sys.path.insert` import shim and passes for the wrong reason);
* it never imports `generate_signals`, `build_candidates` or
  `fetch_latest_predictions`, so it cannot act on a stale prediction or
  contradict the hourly cycle's direction;
* it shares `risk_exit_reason`, `close_trade` and `update_portfolio` with the
  engine by identity, so two copies of a stop rule cannot drift apart.

It installs four packages rather than the full requirements file, which keeps
it near a minute, and writes a `portfolio` row only when something closed.

**One bug caught while building it:** the first version read `trade_history`
*after* writing the closes, then passed it through `augment_history()` — which
adds those same closes again. A stop-out would have been booked at twice its
loss. The history is now read before any write, matching the hourly cycle, and
`test_the_history_is_read_before_the_closes_are_written` pins the ordering.
