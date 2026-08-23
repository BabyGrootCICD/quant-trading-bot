"""The bug that started it all.

`engineer.py` produced 16 columns that did not exist on the `features` table,
so every upsert failed with PGRST204 and the table froze. This locks the
engineer's output, the models' input, and the declared schema together.
"""

import re
from pathlib import Path

import pandas as pd
import numpy as np

from src.features.engineer import engineer_features
from src.models.features import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]


def _schema_columns(table: str) -> set[str]:
    sql = (ROOT / "src" / "data" / "schema.sql").read_text()
    block = sql.split(f"CREATE TABLE IF NOT EXISTS {table} (", 1)[1].split(");", 1)[0]
    cols = set()
    for line in block.splitlines():
        line = line.strip()
        m = re.match(r"^([a-z_][a-z0-9_]*)\s+[A-Z]", line)
        if m:
            cols.add(m.group(1))
    return cols


def _migration_columns() -> set[str]:
    cols = set()
    for f in (ROOT / "migrations").glob("*.sql"):
        for m in re.finditer(r"ADD COLUMN IF NOT EXISTS ([a-z_][a-z0-9_]*)", f.read_text()):
            cols.add(m.group(1))
    return cols


def _synthetic_candles(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    return pd.DataFrame({
        "symbol": "BTC/USDT",
        "timestamp": np.arange(n) * 3_600_000,
        "open": close,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": rng.uniform(1, 100, n),
    })


def test_every_engineered_column_exists_in_schema():
    """This assertion, had it existed, would have caught the PGRST204 outage."""
    produced = set(engineer_features(_synthetic_candles()).columns)
    declared = _schema_columns("features") | _migration_columns()

    missing = produced - declared - {"id", "created_at"}
    assert not missing, f"engineer.py emits columns with no DB home: {sorted(missing)}"


def test_model_feature_cols_are_all_produced():
    produced = set(engineer_features(_synthetic_candles()).columns)
    missing = set(FEATURE_COLS) - produced
    assert not missing, f"models require features the engineer never emits: {sorted(missing)}"


def test_migration_003_covers_the_gap():
    """The 16 columns commit 1d1ec52 forgot must be in a migration."""
    migrated = _migration_columns()
    for col in ["rsi_7", "macd_hist", "bb_width", "bb_position", "volume_ratio_48",
                "atr_14", "atr_14_pct", "williams_r", "stoch_k", "stoch_d",
                "ha_trend", "close_pct_ma20", "close_pct_ma50", "vol_20",
                "skew_20", "target_4h"]:
        assert col in migrated, f"{col} has no migration"


def test_target_uses_next_bar_return_not_current():
    """target_1h must look forward; a lookahead-free label is the whole point."""
    df = _synthetic_candles(100)
    feats = engineer_features(df)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    expected = (log_ret.shift(-1) > 0).astype(int)
    pd.testing.assert_series_equal(
        feats["target_1h"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )
