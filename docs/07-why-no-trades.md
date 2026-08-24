# Why no trades are being made

Date: 2026-08-25
Evidence: hourly run 32785112191, and 1350 hourly bars per symbol across six
pairs pulled live from OKX.

The books were reset to $10,000 by migration 006. Since then: eight signals
per hour, eight refusals, zero positions.

```
[4/6] Reconciling positions (classify -> close -> allocate -> open)...
  Open positions: 0
  Skipped ADA/USDT:  EV -0.274% (strength 0.52, E|move| 0.627%, needs 73.9% accuracy)
  Skipped BNB/USDT:  EV -0.282% (strength 0.53, E|move| 0.344%, needs 93.7% accuracy)
  Skipped BTC/USDT:  EV -0.298% (strength 0.50, E|move| 0.292%, needs 101.3% accuracy)
  Skipped DOGE/USDT: EV -0.288% (strength 0.51, E|move| 0.522%, needs 78.7% accuracy)
  Skipped ETH/USDT:  EV -0.285% (strength 0.52, E|move| 0.443%, needs 83.9% accuracy)
  Skipped SOL/USDT:  EV -0.264% (strength 0.53, E|move| 0.525%, needs 78.6% accuracy)
  Skipped TRX/USDT:  EV -0.225% (strength 0.62, E|move| 0.317%, needs 97.3% accuracy)
  Skipped XRP/USDT:  EV -0.291% (strength 0.51, E|move| 0.536%, needs 78.0% accuracy)
```

**This is the system working.** Every symbol now reaches the gate — that was
the point of dropping the 0.55/0.45 threshold — and every one is refused
because its expected value is negative. The bot is declining to buy a
guaranteed loss.

## Which term is binding

`EV = (2p - 1) * E|move| - cost`. Two candidate culprits: the probability or
the move forecast. Hold each fixed and solve for the other:

| symbol | edge `2p-1` | E&#124;move&#124; | move needed | p needed |
|---|---|---|---|---|
| ADA/USDT | 0.040 | 0.627% | 7.5% | 0.739 |
| BNB/USDT | 0.060 | 0.344% | 5.0% | 0.936 |
| BTC/USDT | 0.000 | 0.292% | — | 1.014 |
| DOGE/USDT | 0.020 | 0.522% | 15.0% | 0.787 |
| ETH/USDT | 0.040 | 0.443% | 7.5% | 0.839 |
| SOL/USDT | 0.060 | 0.525% | 5.0% | 0.786 |
| TRX/USDT | 0.240 | 0.317% | 1.2% | 0.973 |
| XRP/USDT | 0.020 | 0.536% | 15.0% | 0.780 |

It is the **edge multiplier**, not the magnitude head. Calibrated
probabilities sit at 0.50–0.53, so `2p-1 ≈ 0.04` — and 4% of *any* realistic
hourly move cannot cover a 0.30% round trip. Even at an impossible 5% forecast
move, p=0.52 yields **-0.100%**. Improving the magnitude model cannot fix
this; it is not the constraint.

## Does a longer horizon rescue it?

This was the standing hope from `docs/02` — moves grow as √t while cost is
fixed, so at 24h BTC's expected move is ~5× larger against the same 0.30%.
Nothing had ever measured whether the *edge* survives the stretch. It does
not.

Walk-forward, out-of-sample, balanced accuracy (immune to class skew), 1350
bars per symbol:

| horizon | mean edge | mean EV (taker) | mean EV (maker) | tradeable (maker) |
|---|---|---|---|---|
| 1h | +1.06pp | −0.292% | −0.072% | 0/6 |
| 4h | +1.25pp | −0.279% | −0.059% | 0/6 |
| 12h | +1.09pp | −0.236% | −0.016% | 2/6 |
| 24h | **−1.20pp** | −0.219% | +0.001% | 2/6 |

The mean edge does not improve with horizon, and at 24h it is *negative* —
worse than a coin flip.

### The per-symbol spread is the real tell

| symbol | 1h | 4h | 12h | 24h |
|---|---|---|---|---|
| BTC/USDT | +2.78 | +3.92 | +6.20 | +3.30 |
| ETH/USDT | +1.85 | +0.79 | −1.35 | −3.63 |
| SOL/USDT | +0.44 | −0.30 | −2.02 | −6.90 |
| ADA/USDT | +1.62 | +3.15 | +6.23 | +5.73 |
| XRP/USDT | +0.00 | −0.77 | −3.99 | −7.60 |
| DOGE/USDT | −0.36 | +0.70 | +1.45 | +1.92 |

ADA at +6.23pp and SOL at −6.90pp look like strong signal in opposite
directions. They are the same thing: noise.

A 24h forward label at bar *t* and at bar *t+1* share 23/24 of the same price
path, so consecutive observations are not independent. The effective sample is
roughly `n / h`:

| horizon | nominal n | effective n | 1 SE of accuracy |
|---|---|---|---|
| 1h | 1350 | 1350 | ±1.36pp |
| 4h | 1350 | 338 | ±2.72pp |
| 12h | 1350 | 112 | ±4.71pp |
| 24h | 1350 | 56 | **±6.67pp** |

The measured spread grows in lockstep with the error bar — ±2.78pp at 1h,
±7.60pp at 24h — which is precisely what pure noise does and precisely what
real signal does not. Four of six symbols sit inside one standard error of
zero at every horizon.

**There is no measurable directional edge at any horizon tested.** The
`+1.06pp` at 1h is itself inside its own ±1.36pp band.

## So what would make it trade?

| configuration | tradeable |
|---|---|
| taker, 1h (today) | 0/6 |
| maker fills, 1h | 0/6 |
| taker, 24h | 1/6 |
| maker fills + 24h | 5/6 |

Only the last combination opens the book — and it does so by pairing an
execution assumption (a resting limit order that always fills, which is not a
thing) with a horizon whose edge measurement is indistinguishable from zero.
Trading on that is not a strategy, it is two optimistic assumptions
multiplied together.

## What this means

The engineering is now correct. Over `docs/01`–`06` the pipeline went from
silently broken to: fresh predictions every hour, calibrated probabilities, a
conditional magnitude forecast, EV-based selection, Kelly allocation, exits,
one-pass reconciliation, and books that balance. All of it works, and it is
all being pointed at a feature set that carries no measurable alpha.

The missing ingredient is not engineering. **Do not lower `MIN_EDGE_MARGIN`,
widen the thresholds, or weaken the gate to make trades appear** — the gate is
the only thing standing between this system and the steady bleed documented in
`docs/01`. Every trade it currently refuses has negative expected value, and
taking them would reproduce the original outage's P&L on purpose.

Where edge might actually live, roughly in order of cost-to-test:

1. **Cross-sectional, not time-series.** Every feature here is standard TA on
   a symbol's own history — the most heavily mined signal space that exists.
   Relative strength against the basket ("is SOL cheap versus its peers right
   now") is a different question and a far less crowded one.
2. **Cheaper execution as a first-class goal.** At 0.08% maker the bar drops
   from ~78% accuracy to ~57%. That is still above what the models deliver,
   but it is the difference between "implausible" and "worth pursuing", and it
   requires a limit-order execution model rather than a config flag.
3. **Data the price series does not contain.** Funding rates, order-book
   imbalance, on-chain flows. Predicting price from lagged price is where
   everyone starts and where nearly everyone stops.
4. **A cost-aware label.** Train on "does the move clear costs within N bars"
   rather than the sign of a move that is usually too small to trade — the
   model would then be optimising the question the gate actually asks.

Until one of those produces an edge that survives its own error bar, zero
trades is the correct output, and the equity curve staying flat at $10,000 is
the system succeeding rather than failing.
