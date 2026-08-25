"""The fetcher re-downloaded and re-upserted 2 years of candles every hour.

Not corrupting -- `candles` has UNIQUE(symbol,timestamp) and the writes are
upserts -- but ~140k redundant rows per run across 8 symbols.
"""

import pytest
from src.data import fetcher

HOUR = 3_600_000
BACKFILL_START = 1_700_000_000_000


def test_empty_table_backfills_full_window():
    assert fetcher.resolve_since(None, BACKFILL_START) == BACKFILL_START


def test_populated_table_resumes_from_the_tail():
    latest = BACKFILL_START + 5000 * HOUR
    since = fetcher.resolve_since(latest, BACKFILL_START)
    assert since == latest - 2 * HOUR


def test_resume_overlaps_so_the_forming_candle_is_corrected():
    latest = BACKFILL_START + 5000 * HOUR
    assert fetcher.resolve_since(latest, BACKFILL_START) < latest


def test_resume_never_predates_the_backfill_window():
    """A stale row older than the retention window must not widen the fetch."""
    assert fetcher.resolve_since(BACKFILL_START, BACKFILL_START) == BACKFILL_START


def test_candles_are_upserted_not_inserted():
    """Duplicate protection: the H1 hypothesis about DB tampering."""
    import inspect
    src = inspect.getsource(fetcher.upsert_candles)
    assert "upsert" in src
    assert 'on_conflict="symbol,timestamp"' in src


# --- the still-forming bar must never be stored ----------------------------
#
# Exchanges return the bar currently forming as the last element. Storing it
# meant the newest feature row -- the one `upsert_latest_prediction` turns into
# THE traded prediction -- was computed from a fraction of an hour. Measured on
# live 5-minute data, five minutes into a bar `volume_ratio` and
# `volume_ratio_48` sit at the 0th percentile of the completed-bar
# distribution: outside anything the model saw in training. The production run
# that prompted this was 2.4 minutes in.

from src.data.fetcher import drop_incomplete_bars

HOUR = 3_600_000


def _bars(n):
    return [[i * HOUR, 1.0, 1.0, 1.0, 1.0, 1.0] for i in range(n)]


def test_the_forming_bar_is_not_stored():
    now = 10 * HOUR + 150_000              # 2.5 minutes into bar 10
    kept = drop_incomplete_bars(_bars(11), now)
    assert kept[-1][0] == 9 * HOUR, "newest stored bar must be the last complete one"
    assert len(kept) == 10


def test_a_bar_becomes_storable_the_instant_it_closes():
    bars = _bars(11)
    assert drop_incomplete_bars(bars, 10 * HOUR - 1)[-1][0] == 8 * HOUR
    assert drop_incomplete_bars(bars, 10 * HOUR)[-1][0] == 9 * HOUR
    assert drop_incomplete_bars(bars, 11 * HOUR)[-1][0] == 10 * HOUR


def test_nothing_is_dropped_when_every_bar_has_closed():
    bars = _bars(5)
    assert drop_incomplete_bars(bars, 100 * HOUR) == bars


def test_all_bars_dropped_rather_than_storing_a_partial_one():
    """Degrading to 'no new candles' is correct; storing a partial bar is not."""
    assert drop_incomplete_bars(_bars(1), 0) == []


def test_empty_input_is_not_a_crash():
    assert drop_incomplete_bars([], 10 * HOUR) == []


def test_the_prediction_bar_now_matches_what_the_engine_trades():
    """Semantics, not just hygiene. The label is sign(return of the NEXT bar).

    With the forming bar stored, the newest feature row was bar T+1 (partial)
    and its forecast was for T+2 -- a bar that had not started -- while
    `horizon_fraction()` discounted it as though it applied to the bar in
    progress. The two disagreed by a whole bar. Storing only complete bars
    makes the newest row T, its forecast T+1, and T+1 is the bar now forming.
    """
    from src.strategy.economics import horizon_fraction

    now = 10 * HOUR + 6 * 60_000           # 6 minutes into bar 10
    newest = drop_incomplete_bars(_bars(11), now)[-1][0]

    assert newest == 9 * HOUR              # prediction is stamped bar 9
    target_bar = newest + HOUR             # its forecast is for bar 10
    assert target_bar == 10 * HOUR         # which is the bar now forming
    assert horizon_fraction(now) == pytest.approx(0.9, abs=0.01)
