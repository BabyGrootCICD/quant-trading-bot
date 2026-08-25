# The gate was not blocked. It was firing on its own artifacts.

Date: 2026-08-25
Evidence: 4068 out-of-sample bars across six pairs, gate backtested exactly as
the engine applies it.

## The question

Run #47, eight signals, eight refusals, zero trades — again. `docs/07`
concluded "no measurable edge, so refusing is correct". That conclusion was
reached from summary statistics. It had never been tested by running the gate
over history and settling what it picked.

Doing that turned up something `docs/07` missed.

## There is more discrimination than reported

AUC is invariant under any monotone transform, so it measures the model
*before* calibration can touch it:

| symbol | AUC | top-decile observed up-rate |
|---|---|---|
| BTC/USDT | **0.5547** | 0.6435 |
| DOGE/USDT | 0.5308 | 0.5752 |
| ETH/USDT | 0.5290 | 0.5826 |
| XRP/USDT | 0.5210 | 0.4737 |
| ADA/USDT | 0.5158 | 0.5273 |
| SOL/USDT | 0.5135 | 0.4867 |

Mean AUC 0.5275 against a standard error of 0.0086. BTC at 0.5547 is six
standard errors above chance, and the bars it ranks highest genuinely close up
64% of the time. That is weak, but it is not nothing, and `docs/07`'s framing
of "no edge anywhere" was too flat.

## And the gate does fire — it just loses

Backtesting the live gate out of sample, fitting both heads on train only:

| calibration | cost | fires | mean PnL/trade | win rate | total |
|---|---|---|---|---|---|
| isotonic | taker 0.30% | 16 / 4068 (0.39%) | **−0.3447%** | 37.5% | −5.51% |
| isotonic | maker 0.08% | 131 / 4068 (3.22%) | −0.1325% | 47.3% | −17.36% |

So the answer to "why no trades" was never "the gate cannot fire". It fires
about once per 256 bars per symbol — with eight symbols, roughly a trade a
day. Production simply had not reached one yet. **The next trade it took would
have lost money**, and kept doing so.

## Why: the direction head had the magnitude head's disease

`MoveCalibrator` was fixed for tail overfitting in `docs/06`.
`ProbabilityCalibrator` was never checked — and it is the same defect.

Isotonic regression is a step function over pooled-adjacent blocks, and its
fitted value for a block is the *mean of that block*. At the extremes the
blocks are tiny. Live calibrated ranges:

| BTC | ETH | SOL | ADA | XRP | DOGE |
|---|---|---|---|---|---|
| 0.010–0.990 | 0.333–0.990 | 0.010–0.561 | **0.333–0.667** | 0.333–0.990 | 0.010–0.990 |

`0.333` and `0.667` are one-third and two-thirds — blocks of *three
observations*. `0.010` and `0.990` are the y_min/y_max clips. None of these are
probabilities estimated from data, and the EV gate consumed them as fact:

```
at E|move| = 0.4%   the gate needs p > 0.875
at E|move| = 0.6%   the gate needs p > 0.750
at E|move| = 1.0%   the gate needs p > 0.650
```

A three-sample block reaching 0.667, or a clip at 0.990, clears all of those
trivially. And selection points the same way as the bias: **the gate only ever
fires on the most extreme probabilities, which are exactly the blocks with the
least support.** It was selecting its own artifacts — precisely the adverse
selection that `docs/06` identified in the magnitude head and fixed there.

## The fix

`ProbabilityCalibrator` now defaults to sigmoid (Platt) — a two-parameter
logistic fit over the whole score range, which cannot be moved to certainty by
three observations. Isotonic remains available via `method="isotonic"`.

| | isotonic | sigmoid |
|---|---|---|
| max calibrated strength | 0.990 | **0.652** |
| gate fires (taker) | 16 bars, −0.3447%/trade | **0 bars** |
| gate fires (maker) | 131 bars, −0.1325%/trade | 26 bars, −0.0397%/trade, 53.8% win |

The ceiling of 0.652 is the result worth noticing: it lands almost exactly on
BTC's *observed* top-decile up-rate of 0.6435. The calibrated probability now
claims what the data actually shows, and no more.

At taker cost the gate now fires on nothing, which is correct — an honest
ceiling of 0.65 cannot clear a 0.30% round trip against sub-1% hourly moves.
At maker cost it fires 26 times for −0.040% per trade at a 53.8% win rate.
With a per-trade standard deviation around 0.7%, that is −0.040% ± 0.14%:
statistically indistinguishable from breakeven, on 26 trades. Not a strategy,
but a materially different place than −0.13%.

## What changed in the conclusion

`docs/07` said: no edge, refusing is correct. Half right.

* **Refusing is correct** — confirmed properly this time, by backtest rather
  than by inference. At taker cost the honest gate fires zero times in 4068
  bars.
* **"No edge" was too strong.** Mean AUC 0.5275 is six SE above chance on the
  best symbol. There is a little real discrimination; there is just nowhere
  near enough of it to pay a 0.30% round trip.
* **There was a live bug the summary statistics hid.** The bot was one trade
  away from beginning to lose money, and every trade after it, because the
  gate's most confident inputs were its least trustworthy ones.

The order matters: this is why "the system is correctly doing nothing" is not
a safe place to stop investigating. It was doing nothing for the right reason
and about to start doing the wrong thing for a bad one.
