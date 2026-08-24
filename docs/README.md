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
