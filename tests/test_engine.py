"""The mechanism that turned a broken pipeline into a steady cash bleed.

With training dead, the engine read one frozen prediction set and, every hour,
closed all four positions and reopened the identical four -- paying fee +
slippage on a signal that never changed.
"""

import pandas as pd
import pytest

from src.paper_trading import engine


class FakeQuery:
    def __init__(self, recorder, name, op, payload):
        self.recorder, self.name, self.op, self.payload = recorder, name, op, payload
        self.filters = {}

    def eq(self, col, val):
        self.filters[col] = val
        return self

    def execute(self):
        self.recorder.append((self.name, self.op, self.payload, self.filters))
        return type("Resp", (), {"data": []})()


class FakeTable:
    def __init__(self, recorder, name):
        self.recorder, self.name = recorder, name

    def update(self, payload):
        return FakeQuery(self.recorder, self.name, "update", payload)

    def insert(self, payload):
        return FakeQuery(self.recorder, self.name, "insert", payload)


class FakeClient:
    def __init__(self):
        self.calls = []

    def table(self, name):
        return FakeTable(self.calls, name)


NOW = 1_800_000_000_000
PRICES = {"BTC/USDT": 100.0, "ETH/USDT": 50.0}


def _open_trades():
    return pd.DataFrame([
        {"id": 1, "symbol": "BTC/USDT", "side": "long", "entry_price": 100.0, "size": 100.0},
        {"id": 2, "symbol": "ETH/USDT", "side": "short", "entry_price": 50.0, "size": 100.0},
    ])


def _signals(btc=1, eth=-1):
    return pd.DataFrame([
        {"symbol": "BTC/USDT", "signal": btc, "signal_strength": 0.8,
         "probability_up": 0.8, "estimated_change_pct": 0.01},
        {"symbol": "ETH/USDT", "signal": eth, "signal_strength": 0.8,
         "probability_up": 0.2, "estimated_change_pct": -0.01},
    ])


# --- staleness guard -------------------------------------------------------

def test_stale_prediction_is_rejected():
    three_hours_ago = NOW - 3 * 3_600_000
    assert engine.is_prediction_fresh(three_hours_ago, NOW) is False


def test_fresh_prediction_is_accepted():
    thirty_min_ago = NOW - 1_800_000
    assert engine.is_prediction_fresh(thirty_min_ago, NOW) is True


def test_future_timestamp_is_rejected():
    assert engine.is_prediction_fresh(NOW + 3_600_000, NOW) is False


def test_the_exact_outage_prediction_would_now_be_blocked():
    """The live bot traded a prediction that was hours stale, hour after hour."""
    assert engine.is_prediction_fresh(NOW - 19 * 3_600_000, NOW) is False


# --- no churn --------------------------------------------------------------

def test_unchanged_signal_holds_instead_of_round_tripping():
    client = FakeClient()
    wanted = engine.desired_sides(_signals())
    closed = engine.close_open_positions(client, _open_trades(), PRICES, NOW, wanted=wanted)

    assert closed == []
    assert not [c for c in client.calls if c[1] == "update"], "held positions must not be closed"


def test_flipped_signal_does_close():
    client = FakeClient()
    wanted = engine.desired_sides(_signals(btc=-1, eth=-1))
    closed = engine.close_open_positions(client, _open_trades(), PRICES, NOW, wanted=wanted)

    assert closed == ["BTC/USDT"]


def test_no_fresh_signal_closes_everything():
    client = FakeClient()
    closed = engine.close_open_positions(client, _open_trades(), PRICES, NOW, wanted={})
    assert sorted(closed) == ["BTC/USDT", "ETH/USDT"]


def test_already_held_symbol_is_not_reopened():
    client = FakeClient()
    held = {"BTC/USDT": "long"}
    engine.open_new_positions(client, _signals(btc=1, eth=0), PRICES, NOW, "logistic_v2",
                              10000.0, held=held)
    inserts = [c for c in client.calls if c[1] == "insert"]
    assert inserts == [], "re-entering an identical position just pays the spread again"


# A 0.8-strength signal has a 0.6 edge multiplier, so it needs E|move| > 0.5%
# to clear the 0.30% round trip.
TRADEABLE_MOVES = {"BTC/USDT": 0.01, "ETH/USDT": 0.01}
REAL_MOVES = {"BTC/USDT": 0.0022, "ETH/USDT": 0.0031}  # measured hourly moves


def test_new_symbol_is_opened_when_edge_clears_cost():
    client = FakeClient()
    engine.open_new_positions(client, _signals(btc=1, eth=0), PRICES, NOW, "logistic_v2",
                              10000.0, held={}, expected_moves=TRADEABLE_MOVES)
    inserts = [c for c in client.calls if c[1] == "insert"]
    assert len(inserts) == 1
    assert inserts[0][2]["symbol"] == "BTC/USDT"
    assert inserts[0][2]["side"] == "long"


def test_real_hourly_volatility_blocks_the_trade():
    """The core finding: at 1h, BTC's 0.22% move cannot pay a 0.30% round trip."""
    client = FakeClient()
    engine.open_new_positions(client, _signals(btc=1, eth=0), PRICES, NOW, "logistic_v2",
                              10000.0, held={}, expected_moves=REAL_MOVES)
    assert [c for c in client.calls if c[1] == "insert"] == []


def test_missing_volatility_estimate_blocks_the_trade():
    client = FakeClient()
    engine.open_new_positions(client, _signals(btc=1, eth=0), PRICES, NOW, "logistic_v2",
                              10000.0, held={}, expected_moves={})
    assert [c for c in client.calls if c[1] == "insert"] == []


def test_position_size_uses_real_expected_move_not_the_fabricated_map():
    """estimated_change_pct was (p-0.5)*2*0.02 -- unrelated to real volatility."""
    client = FakeClient()
    engine.open_new_positions(client, _signals(btc=-1, eth=0), PRICES, NOW, "logistic_v2",
                              10000.0, held={}, expected_moves=TRADEABLE_MOVES)
    inserts = [c for c in client.calls if c[1] == "insert"]
    assert len(inserts) == 1
    assert inserts[0][2]["side"] == "short"


# --- stats epoch -----------------------------------------------------------

def test_legacy_broken_era_trades_are_excluded_from_stats():
    epoch = 1_787_531_100_000
    hist = pd.DataFrame({
        "pnl": [-0.15] * 3 + [1.0] * 2,
        "size": [100.0] * 5,
        "entry_time": [epoch - 3_600_000, epoch - 7_200_000, epoch - 1, epoch, epoch + 3_600_000],
    })
    kept = engine.filter_to_stats_epoch(hist, epoch_ms=epoch)
    assert len(kept) == 2
    assert (kept["pnl"] == 1.0).all()


def test_stats_epoch_passes_through_when_column_absent():
    hist = pd.DataFrame({"pnl": [1.0], "size": [100.0]})
    assert len(engine.filter_to_stats_epoch(hist)) == 1


def test_stats_epoch_handles_empty_history():
    assert engine.filter_to_stats_epoch(pd.DataFrame()).empty


# --- honest cost accounting -----------------------------------------------

def test_round_trip_charges_both_legs():
    per_leg = 100.0 * engine.TAKER_FEE + 100.0 * (engine.SLIPPAGE_BPS / 10000)
    assert engine.round_trip_cost(100.0) == pytest.approx(2 * per_leg)


def test_round_trip_cost_matches_observed_bleed():
    """$0.30 per $100 round trip; 4 of those an hour is the observed drain."""
    assert engine.round_trip_cost(100.0) == pytest.approx(0.30)


# --- Sharpe ----------------------------------------------------------------

def _history(n=200, pnl=-0.15, size=100.0):
    """Newest-first, as the query returns it."""
    return pd.DataFrame({
        "pnl": [pnl] * n,
        "size": [size] * n,
        "exit_time": [NOW - i * 3_600_000 for i in range(n)],
    })


def test_sharpe_is_independent_of_position_size():
    """Dollar PnL made the ratio scale with size; returns do not."""
    small = engine.compute_sharpe(_history(pnl=-0.15, size=100.0))
    large = engine.compute_sharpe(_history(pnl=-1.5, size=1000.0))
    assert small == pytest.approx(large)


def test_sharpe_uses_newest_trades_not_oldest():
    """`closed_pnl[-168:]` sliced the oldest rows off a newest-first query."""
    # Alternating magnitudes so the winning window has real variance;
    # a constant series is degenerate and correctly scores 0.
    newest_winners = [1.0 if i % 2 else 2.0 for i in range(168)]
    oldest_losers = [-5.0 if i % 2 else -6.0 for i in range(100)]
    recent_wins = pd.DataFrame({
        "pnl": newest_winners + oldest_losers,
        "size": [100.0] * 268,
        "exit_time": [NOW - i * 3_600_000 for i in range(268)],
    })
    # The head (newest 168) is all winners, so Sharpe must be positive.
    assert engine.compute_sharpe(recent_wins, window=168) > 0


def test_sharpe_of_constant_pnl_is_zero_not_huge():
    assert engine.compute_sharpe(_history()) == 0.0


def test_sharpe_handles_empty_history():
    assert engine.compute_sharpe(pd.DataFrame()) is None


def test_sharpe_handles_missing_size_column():
    assert engine.compute_sharpe(pd.DataFrame({"pnl": [1.0, 2.0]})) is None


# --- unknown is not zero ---------------------------------------------------

def test_sharpe_is_unknown_below_the_minimum_trade_count():
    """0.0 is a claim ("no risk-adjusted return"); too-few-trades is not."""
    from src.utils.metrics import MIN_SHARPE_TRADES
    assert engine.compute_sharpe(_history(n=MIN_SHARPE_TRADES - 1, pnl=1.0)) is None


def test_sharpe_becomes_a_number_once_enough_trades_exist():
    from src.utils.metrics import MIN_SHARPE_TRADES
    varied = pd.DataFrame({
        "pnl": [1.0 if i % 2 else -0.5 for i in range(MIN_SHARPE_TRADES)],
        "size": [100.0] * MIN_SHARPE_TRADES,
        "exit_time": [NOW - i * 3_600_000 for i in range(MIN_SHARPE_TRADES)],
    })
    assert isinstance(engine.compute_sharpe(varied), float)


def test_history_limit_is_not_a_silent_500_cap():
    assert engine.TRADE_HISTORY_LIMIT > 500
