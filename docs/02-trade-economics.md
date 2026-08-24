# Trade economics

**The finding:** at a 1-hour holding period with a 0.30% round-trip cost, a
directional strategy on these pairs cannot be profitable — for two of them, not
at *any* accuracy. This is arithmetic about the cost structure, not a statement
about the model.

## The formula

For a directional bet where being right earns the move and being wrong loses
it:

```
EV per trade = (2p - 1) x E|move| - cost
```

- `p` — probability the direction call is correct
- `E|move|` — expected absolute price move over the holding period
- `cost` — round trip: `2 x (0.10% taker fee + 5bps slippage) = 0.30%`

Setting `EV = 0` gives the accuracy required to break even:

```
break-even p = (cost / E|move| + 1) / 2
```

When `E|move| < cost`, this exceeds 1.0 — no classifier can reach it.

## Measured over 1000 live binanceus hourly bars (2026-08-24)

| symbol | E abs 1h move | model accuracy | break-even accuracy | EV/trade |
|---|---|---|---|---|
| BTC/USDT | 0.220% | 0.5155 | **118.1%** | -0.293% |
| ETH/USDT | 0.306% | 0.5236 | 98.9% | -0.286% |
| ADA/USDT | 0.572% | 0.5156 | 76.2% | -0.282% |
| DOGE/USDT | 0.364% | 0.5208 | 91.3% | -0.285% |
| TRX/USDT | 0.225% | 0.6502 | **116.7%** | -0.232% |

**BTC and TRX require over 100% accuracy to break even.**

Supporting detail: BTC's *median* hourly move is 0.14%, under half the round
trip, and only **21.8%** of BTC bars move more than the cost at all. Trading
every bar is a guaranteed loss regardless of skill. The measured EV of about
-0.28%/trade against an actual -$9.58 over 75 trades (-0.13%/trade at $100
size) is the same phenomenon, softened by trades that happened to win.

## What was implemented

`src/strategy/economics.py` is the single home for the cost model:

```python
round_trip_cost_pct()                      # 0.003
expected_edge(strength, move)              # (2p - 1) * move
expected_value(strength, move)             # edge - cost
breakeven_accuracy(move)                   # (cost/move + 1) / 2
is_tradeable(strength, move, margin=1.0)   # edge > margin * cost
```

The engine estimates each symbol's `E|move|` from the mean absolute hourly log
return over the last 168 live candles, and refuses any entry whose expected
edge does not clear the round trip.

This also replaced the previous sizing input. `estimated_change_pct` had been
`(probability_up - 0.5) * 2 * 0.02` — a fabricated linear map with a hardcoded
2% ceiling, unrelated to any symbol's actual volatility.
`percentage_based_size` divided by it, so position size was noise that almost
always saturated at the $100 cap.

## Live behaviour

Run `32679878553`, 8 active signals, all refused:

```
Round-trip cost: 0.30% of position

    ADA/USDT: E|move|=0.881% -> needs 67.0% accuracy
    BNB/USDT: E|move|=0.394% -> needs 88.0% accuracy
    BTC/USDT: E|move|=0.354% -> needs 92.3% accuracy
   DOGE/USDT: E|move|=0.728% -> needs 70.6% accuracy
    ETH/USDT: E|move|=0.481% -> needs 81.2% accuracy
    SOL/USDT: E|move|=0.533% -> needs 78.1% accuracy
    TRX/USDT: E|move|=0.283% -> needs 102.9% accuracy  UNREACHABLE
    XRP/USDT: E|move|=0.957% -> needs 65.7% accuracy

  Skipped BTC/USDT:  EV -0.244% (strength 0.58, needs 92.3% accuracy)
  Skipped ETH/USDT:  EV -0.198% (strength 0.61, needs 81.2% accuracy)
  Skipped BNB/USDT:  EV -0.259% (strength 0.55, needs 88.0% accuracy)
  Skipped SOL/USDT:  EV -0.223% (strength 0.57, needs 78.1% accuracy)
  Skipped XRP/USDT:  EV -0.199% (strength 0.55, needs 65.7% accuracy)
  Skipped DOGE/USDT: EV -0.223% (strength 0.55, needs 70.6% accuracy)
  Skipped TRX/USDT:  EV -0.222% (strength 0.64, needs 102.9% accuracy)
```

**A bot that opens nothing is the correct outcome here**, not a malfunction.
The best available signal needs 65.7% accuracy and the models deliver ~52%.

## What would change the arithmetic

None of these are implemented. They are the honest options.

**1. A longer holding period.** Moves scale roughly with `sqrt(t)` while the
cost stays fixed, so the ratio improves with horizon:

| horizon | BTC E\|move\| | break-even accuracy |
|---|---|---|
| 1h | 0.220% | 118.1% |
| 4h | ~0.440% | ~84.1% |
| 24h | ~1.078% | ~63.9% |

At 24h the bar is demanding but no longer impossible. Requires a `target_24h`
label and a holding policy that does not re-evaluate hourly.

**2. Trade only high-volatility bars.** Conditioning on the top quintile of ATR
raises `E|move|` several-fold. The EV gate already does this implicitly — it
passes exactly those bars where the move is large enough to pay.

**3. Cheaper execution.** Maker limit orders instead of taker market orders
would roughly halve the cost term. `TAKER_FEE` and `SLIPPAGE_BPS` in
`src/strategy/economics.py` are the levers; changing them changes what the gate
allows, so they should reflect reality rather than optimism.

**4. Cost-aware labels.** Train on "does the move clear costs within N bars?"
(a triple-barrier label) instead of the sign of a move that is usually too
small to trade. The model would then be optimising the question that matters.
