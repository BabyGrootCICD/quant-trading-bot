# Engineering notes

Findings from the 2026-08-23/24 investigation into why the paper-trading
portfolio lost money monotonically (`total_pnl -9.58`, `win_rate 0.35`,
`sharpe_ratio` swinging between -8 and -49).

Read them in order; each builds on the one before.

| # | Document | What it covers |
|---|---|---|
| 01 | [Pipeline outage](01-pipeline-outage.md) | The four-link failure chain that left the bot trading one frozen prediction set for a day |
| 02 | [Trade economics](02-trade-economics.md) | Why a 1h directional strategy on these pairs cannot be profitable at a 0.30% round trip, whatever the model does |
| 03 | [Metrics integrity](03-metrics-integrity.md) | Every way the reported numbers were lying: in-sample scoring, class imbalance, dollar-PnL Sharpe, legacy contamination |
| 04 | [Silent failure patterns](04-silent-failures.md) | Five distinct ways this codebase reported success while doing nothing, and the guards now in place |
| 05 | [Operations](05-operations.md) | Running the pipeline, applying migrations, reading the output, known constraints |
| 06 | [EV allocation](06-ev-allocation.md) | Why "correctly refuses to trade" became "never trades and never exits", and the magnitude head, calibration, exit policy and Kelly allocator that fix it |
| 07 | [Why no trades](07-why-no-trades.md) | The measurement everything hinges on: no directional edge survives its own error bar at any horizon, so zero trades is the correct output |
| 08 | [The forming bar](08-forming-bar.md) | The traded prediction was computed from a bar that had not finished — volume features at the 0th percentile of anything the model was trained on |

## The one-paragraph version

The bot was not losing money because of a bad model. Commit `1d1ec52` added 16
feature columns without a migration, so every feature write failed; training
then failed on all 8 symbols every hour; nothing had ever written the
`predictions` table in the first place; and the engine therefore replayed one
frozen prediction set every hour, closing and reopening the same four
positions and paying 0.30% each time. Every stage reported success. After
fixing the chain, a second question surfaced: at a 1h horizon, with a 0.30%
round trip against a 0.22% mean hourly move, **BTC needs 118% accuracy to
break even**. The pipeline now works and the bot correctly refuses to trade.
It has no demonstrated edge.

That refusal then became its own failure. The EV gate scored every bar against
its symbol's *unconditional* average move -- one constant per symbol -- so it
was not a filter at all, and the bot opened nothing for a day while holding two
positions it had no rule to release; equity froze and fees were being billed
twice on top. Document 06 covers the second round: a conditional magnitude
model (volatility is forecastable even when direction is not), calibration of
both heads, an exit policy, and expected-value position sizing. The gate now
admits roughly 1% of bars -- the ones where the edge can actually pay for its
own execution -- instead of none.


## Where it ended up

The pipeline is correct and the strategy has no alpha. Those are separate
facts and both are load-bearing. Document 07 measures the second one directly:
across six pairs and four horizons, every edge estimate sits inside one
standard error of zero once overlapping labels are accounted for, and the
apparent per-symbol spread grows exactly as the effective sample shrinks --
the signature of noise.

So the EV gate refuses everything, and that is the correct output rather than
a bug to be tuned away. Weakening it would reproduce the bleed in document 01
deliberately.
