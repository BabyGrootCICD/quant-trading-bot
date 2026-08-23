"""The fetcher re-downloaded and re-upserted 2 years of candles every hour.

Not corrupting -- `candles` has UNIQUE(symbol,timestamp) and the writes are
upserts -- but ~140k redundant rows per run across 8 symbols.
"""

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
