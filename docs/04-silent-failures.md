# Silent failure patterns

The outage in [01](01-pipeline-outage.md) ran for roughly 13 hours with a green
checkmark on every workflow run. That was not one bug — it was five distinct
mechanisms by which this codebase reported success while doing nothing.

They are worth cataloguing, because three of them recurred *during the fix*.

## 1. Blanket `except` around a whole unit of work

```python
for symbol in SYMBOLS:
    try:
        ...
    except Exception as e:
        print(f"  ERROR: {e}")   # and then continue, and then exit 0
```

Eight consecutive `AttributeError`s printed as text and the step exited 0.
Printing an error is not reporting one — nothing reads stdout.

**Guard.** `trainer.main()` and `engineer.main()` now `sys.exit(1)` when every
symbol (trainer) or any symbol (engineer) fails. A frozen table stops the
pipeline rather than poisoning what follows.

## 2. Schema drift with no preflight

Adding columns to the writer without adding them to the database made every
write fail identically and permanently. Nothing checked.

**Guard.** `src/data/schema_check.py` runs before the pipeline touches data,
asserting that every column the code expects exists in PostgREST's schema
cache. It is deliberately the first step after dependency install:

```
MISSING on 'features': rsi_7, macd_hist, bb_width, ... target_4h

A migration has not been applied. Run:
  python -m migrations.run_migrations
```

## 3. Implicit row caps on unpaginated reads

```python
client.table("features").select("*").eq("symbol", s).order("timestamp").execute()
```

PostgREST silently caps a response at 1000 rows. Ordered *ascending*, this
returned the **oldest** 1000 rows out of 17,533 — the trainer trained on
two-year-old data and never knew.

The failure is invisible: valid data, correct schema, no error. It surfaced
only because the prediction it produced carried a two-year-old timestamp and
tripped the staleness guard.

**Guard.** Explicit pagination with an explicit cap
(`MAX_TRAINING_ROWS`), ordered newest-first, then sorted chronologically.
Regression tests assert both the row count and that the newest bar is present.

## 4. Omitted keys on an UPSERT retain the old value

This one recurred during the fix. Labelling flat bars as NaN was supposed to
drop them from training. It changed nothing, because:

```python
cleaned = _clean_value(v)
if cleaned is not None:
    rec[k] = cleaned      # NaN -> key omitted entirely
```

An omitted key in an UPSERT is not "write NULL" — it is "leave the existing
value alone". Every flat bar kept its previous `target_1h = 0`. The run
completed successfully, the numbers were byte-identical to the previous run,
and nothing indicated the change had not landed.

**Guard.** Always send the key with an explicit `None`. This also keeps every
record in a batch structurally identical, which PostgREST requires anyway.

## 5. Type coercion at the database boundary

Also introduced during the fix. Switching labels to `np.where` so they could be
NaN made them floats, and Postgres rejects `"0.0"` for a `SMALLINT`:

```
ERROR processing ETH/USDT: invalid input syntax for type smallint: "0.0"
Done. Total feature rows: 0
```

All 8 symbols failed — and pattern #1 meant the step still exited 0, so
training quietly reused stale features again.

**Guard.** `INT_COLUMNS` declares which columns are integral; those are cast on
the way out while staying float in the DataFrame so NaN can mean "unlabelled".
A test asserts the wire types.

## The shared shape

Every one of these produced **valid-looking output from a no-op**. None threw
where anyone would see it. The reason the outage lasted a day is not that any
single bug was subtle — it is that four layers each absorbed the failure of the
one beneath.

Two working rules came out of it:

1. **A stage that produces zero output should fail.** Zero feature rows, zero
   models trained, zero predictions written — these are never legitimate steady
   states, so they should be loud.
2. **Verify a change landed by its effect on the data, not by the run
   succeeding.** Both times a fix silently failed here, the workflow was green
   and the logs looked normal. What gave it away was a number that should have
   moved and didn't.

## Regression coverage

`tests/` was empty at the start of this work. It now holds 80 tests, with at
least one pinning each failure above:

| file | covers |
|---|---|
| `test_features_contract.py` | engineer output vs schema vs migrations; null handling; SMALLINT wire types |
| `test_models.py` | `fit_walk_forward` on both models; `is_fitted`; out-of-sample arrays |
| `test_trainer.py` | prediction persistence; the 1000-row cap; newest-vs-oldest; skill metrics |
| `test_engine.py` | staleness guard; hold-vs-churn; round-trip cost; Sharpe; stats epoch; EV gate |
| `test_economics.py` | break-even arithmetic against measured market data |
| `test_metrics.py` | returns-vs-dollars; annualization; degenerate variance |
| `test_fetcher.py` | incremental resume; upsert duplicate protection |
