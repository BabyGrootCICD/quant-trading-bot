"""The magnitude head, and the accounting bugs it sits next to.

Central claim under test: an EV gate fed an *unconditional* expected move is
not a gate, it is a constant, and the bot had opened nothing since it was
installed. A conditional forecast is what lets the same gate distinguish a
tradeable bar from an untradeable one.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.calibration import ProbabilityCalibrator, calibration_error
from src.models.features import FEATURE_COLS
from src.models.magnitude import (
    MagnitudeModel, MoveCalibrator, rank_ic, spread_ratio, tail_ratio,
)
from src.paper_trading import engine
from src.strategy import economics


# --- the constant-gate problem ---------------------------------------------

def test_unconditional_move_makes_the_gate_a_constant():
    """One number per symbol means EV depends on p alone, on every bar."""
    btc_move = 0.0022  # measured mean absolute hourly move
    calm = economics.expected_value(0.55, btc_move)
    wild = economics.expected_value(0.55, btc_move)
    assert calm == wild
    assert calm < 0, "and at BTC's real hourly move the constant answer is 'no'"


def test_a_conditional_forecast_can_clear_the_same_cost():
    """A volatile bar at the same accuracy is a different trade."""
    quiet = economics.expected_value(0.58, 0.0022)
    volatile = economics.expected_value(0.58, 0.020)
    assert quiet < 0 < volatile
    assert economics.is_tradeable(0.58, 0.020)
    assert not economics.is_tradeable(0.58, 0.0022)


def test_breakeven_falls_as_the_forecast_move_rises():
    assert economics.breakeven_accuracy(0.0022) > 1.0
    assert economics.breakeven_accuracy(0.020) < 0.60


# --- signed forecasts -------------------------------------------------------

def test_expected_return_is_signed_by_the_probability():
    assert economics.expected_return(0.7, 0.01) > 0
    assert economics.expected_return(0.3, 0.01) < 0
    assert economics.expected_return(0.5, 0.01) == pytest.approx(0.0)


def test_predicted_price_moves_the_right_way():
    assert economics.predicted_price(100.0, 0.8, 0.01) > 100.0
    assert economics.predicted_price(100.0, 0.2, 0.01) < 100.0
    assert economics.predicted_price(100.0, 0.5, 0.01) == pytest.approx(100.0)


def test_net_expected_return_is_zero_when_costs_eat_the_edge():
    assert economics.net_expected_return(0.55, 0.0022) == 0.0
    assert economics.net_expected_return(0.8, 0.02) > 0


def test_move_volatility_is_the_normal_relationship():
    assert economics.move_volatility(0.01) == pytest.approx(0.01 * np.sqrt(np.pi / 2))


# --- the engine prefers the conditional forecast ---------------------------

def _row(symbol="BTC/USDT", **kw):
    base = {"symbol": symbol}
    base.update(kw)
    return pd.Series(base)


def test_engine_prefers_the_prediction_row_over_the_symbol_average():
    move = engine.resolve_expected_move(_row(expected_move_pct=0.019),
                                        {"BTC/USDT": 0.0022})
    assert move == pytest.approx(0.019)


def test_engine_falls_back_when_the_forecast_is_missing():
    assert engine.resolve_expected_move(_row(expected_move_pct=None),
                                        {"BTC/USDT": 0.0022}) == pytest.approx(0.0022)
    assert engine.resolve_expected_move(_row(expected_move_pct=float("nan")),
                                        {"BTC/USDT": 0.0022}) == pytest.approx(0.0022)


def test_no_forecast_anywhere_blocks_the_trade():
    assert engine.resolve_expected_move(_row(), {}) is None
    assert engine.resolve_expected_move(_row(expected_move_pct=0.0), {}) is None


def test_a_volatile_bar_becomes_a_candidate_where_the_average_bar_does_not():
    signals = pd.DataFrame([
        {"symbol": "BTC/USDT", "signal": 1, "signal_strength": 0.62,
         "probability_up": 0.62, "expected_move_pct": 0.020},
        {"symbol": "ETH/USDT", "signal": 1, "signal_strength": 0.62,
         "probability_up": 0.62, "expected_move_pct": 0.0031},
    ])
    prices = {"BTC/USDT": 100.0, "ETH/USDT": 50.0}
    cands = engine.build_candidates(signals, prices)
    assert [c.symbol for c in cands] == ["BTC/USDT"]
    assert cands[0].ev > 0


# --- fee double-count -------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.rows = []

    def table(self, name):
        rec = self.rows

        class _Q:
            def __init__(self, payload):
                self.payload = payload

            def execute(self):
                rec.append((name, self.payload))
                return type("R", (), {"data": []})()

        return type("T", (), {"insert": lambda _s, p: _Q(p)})()


def test_fees_are_not_charged_twice():
    """`pnl` is already net of fees; subtracting `fees` again was double-billing.

    The live table showed total_pnl -11.97 alongside cash 9876.29 with $100 of
    locked capital -- $11.74 of cost charged a second time.
    """
    client = _Recorder()
    history = pd.DataFrame({
        "pnl": [-3.0, -3.0],       # already net of the round trip
        "fees": [0.30, 0.30],
        "size": [100.0, 100.0],
        "entry_time": [1, 2],
        "exit_time": [2, 3],
    })
    result = engine.update_portfolio(client, 10_000.0, pd.DataFrame(), {}, history, 12345)

    assert result["cash"] == pytest.approx(10_000.0 - 6.0)
    assert result["equity"] == pytest.approx(9_994.0)
    assert result["realized_fees"] == pytest.approx(0.60)


def test_equity_reconciles_with_open_positions():
    client = _Recorder()
    history = pd.DataFrame({"pnl": [-2.0], "fees": [0.3], "size": [100.0],
                            "entry_time": [1], "exit_time": [2]})
    open_trades = pd.DataFrame([{"symbol": "BTC/USDT", "side": "long",
                                 "entry_price": 100.0, "size": 100.0}])
    result = engine.update_portfolio(client, 10_000.0, open_trades,
                                     {"BTC/USDT": 101.0}, history, 12345)
    # 10000 - 100 locked - 2 realized + (100 + 1 unrealized)
    assert result["equity"] == pytest.approx(9_999.0)


# --- magnitude model --------------------------------------------------------

def _frame(n=2400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLS})
    df["symbol"] = "BTC/USDT"
    df["timestamp"] = np.arange(n) * 3_600_000
    # Volatility clusters and is driven by observable state -- that is the
    # whole premise of the magnitude head. Direction stays a coin flip, which
    # is what the real data looks like too.
    scale = 0.002 + 0.02 * np.abs(df["atr_14_pct"]) + 0.01 * np.abs(df["vol_20"])
    df["target_move_1h"] = np.abs(rng.normal(scale=scale))
    df["target_1h"] = rng.integers(0, 2, size=n).astype(float)
    return df


def test_magnitude_head_ranks_volatile_bars_above_quiet_ones():
    model = MagnitudeModel(hidden_layer_sizes=(32, 16), max_iter=400)
    metrics = model.fit_walk_forward(_frame(), n_splits=3)
    assert "error" not in metrics
    assert metrics["magnitude_rank_ic"] > 0.1
    assert metrics["magnitude_spread_ratio"] > 1.2, "otherwise the gate is still a constant"


def test_magnitude_predictions_are_positive_and_aligned():
    df = _frame()
    model = MagnitudeModel(hidden_layer_sizes=(32, 16), max_iter=400)
    model.fit_walk_forward(df, n_splits=3)
    preds = model.predict(df)
    assert len(preds) == len(df)
    assert (preds > 0).all(), "a zero forecast makes break-even accuracy infinite"


def test_missing_features_fall_back_to_the_symbol_average_not_zero():
    df = _frame()
    model = MagnitudeModel(hidden_layer_sizes=(32, 16), max_iter=400)
    model.fit_walk_forward(df, n_splits=3)
    broken = df.copy()
    broken.loc[0, FEATURE_COLS[0]] = np.nan
    assert model.predict(broken).iloc[0] == pytest.approx(model.baseline_move)


def test_magnitude_head_reports_error_without_the_label():
    df = _frame().drop(columns=["target_move_1h"])
    assert "error" in MagnitudeModel().fit_walk_forward(df)


def test_magnitude_head_refuses_a_short_series():
    assert "error" in MagnitudeModel().fit_walk_forward(_frame(n=100))


def test_rank_ic_of_noise_is_near_zero():
    rng = np.random.default_rng(1)
    assert abs(rank_ic(rng.normal(size=500), rng.normal(size=500))) < 0.2


def test_spread_ratio_of_a_useless_forecast_is_about_one():
    rng = np.random.default_rng(2)
    assert spread_ratio(rng.normal(size=2000), np.abs(rng.normal(size=2000))) \
        == pytest.approx(1.0, abs=0.25)


# --- calibration ------------------------------------------------------------

def test_calibration_pulls_overconfident_scores_toward_reality():
    rng = np.random.default_rng(3)
    truth = rng.random(4000) * 0.2 + 0.4          # real frequency 0.4 - 0.6
    y = (rng.random(4000) < truth).astype(int)
    raw = np.clip((truth - 0.5) * 4 + 0.5, 0.01, 0.99)   # wildly overconfident

    before = calibration_error(raw, y)
    after = calibration_error(ProbabilityCalibrator().fit(raw, y).transform(raw), y)
    assert after < before


def test_calibrator_is_identity_without_enough_data():
    cal = ProbabilityCalibrator().fit([0.4, 0.6], [0, 1])
    assert not cal.is_fitted
    assert list(cal.transform([0.4, 0.6])) == [0.4, 0.6]


def test_calibrator_is_identity_on_a_single_class():
    cal = ProbabilityCalibrator().fit(np.linspace(0.1, 0.9, 500), np.ones(500))
    assert not cal.is_fitted


def test_calibration_error_of_a_perfect_forecaster_is_zero():
    p = np.full(1000, 0.3)
    y = np.zeros(1000)
    y[:300] = 1
    assert calibration_error(p, y) == pytest.approx(0.0, abs=1e-9)


# --- the magnitude head's tail bias ----------------------------------------
#
# Measured on live hourly bars, the *raw* head predicted 5.36% for DOGE's top
# 5% of forecasts and 0.77% actually followed -- a ratio of 0.14. The EV gate
# only ever fires on the top of that range, so selection and bias point the
# same way: without a correction the system trades exactly the bars its
# magnitude model is most wrong about, on an EV it has no basis for.

def _inflated_tail(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    realised = np.abs(rng.normal(scale=0.004, size=n))
    # A forecaster that is fine in the middle and blows up at the top.
    pred = realised * 1.0
    top = realised >= np.quantile(realised, 0.9)
    pred[top] = realised[top] * 5.0
    pred += np.abs(rng.normal(scale=0.0005, size=n))
    return pred, realised


def test_the_raw_tail_bias_is_detected():
    pred, realised = _inflated_tail()
    assert tail_ratio(pred, realised) < 0.5


def test_calibration_removes_the_tail_bias():
    pred, realised = _inflated_tail()
    cal = MoveCalibrator().fit(pred, realised).transform(pred)
    assert tail_ratio(cal, realised) == pytest.approx(1.0, abs=0.25)


def test_calibration_preserves_the_ranking():
    """It corrects the level, not the ordering -- the head's actual skill."""
    pred, realised = _inflated_tail()
    cal = MoveCalibrator().fit(pred, realised).transform(pred)
    assert rank_ic(cal, realised) == pytest.approx(rank_ic(pred, realised), abs=0.05)


def test_move_calibrator_is_identity_without_enough_data():
    cal = MoveCalibrator().fit([0.01, 0.02], [0.01, 0.02])
    assert not cal.is_fitted
    assert list(cal.transform([0.01, 0.02])) == [0.01, 0.02]


def test_calibrated_moves_are_never_zero_or_negative():
    pred, realised = _inflated_tail()
    cal = MoveCalibrator().fit(pred, realised)
    assert (cal.transform([0.0, -1.0, 1e-9]) > 0).all()


def test_the_model_reports_both_tail_ratios():
    model = MagnitudeModel(hidden_layer_sizes=(32, 16), max_iter=400)
    metrics = model.fit_walk_forward(_frame(), n_splits=3)
    assert "magnitude_tail_ratio_raw" in metrics
    assert "magnitude_tail_ratio" in metrics


def test_predictions_go_through_the_calibrator():
    df = _frame()
    model = MagnitudeModel(hidden_layer_sizes=(32, 16), max_iter=400)
    model.fit_walk_forward(df, n_splits=3)
    assert model.calibrator.is_fitted
    # A monotone map cannot produce a forecast above the largest realised move
    # it was ever fitted against, which is the whole point.
    assert model.predict(df).max() <= df["target_move_1h"].max()


# --- the direction head had the magnitude head's disease ---------------------
#
# MoveCalibrator was fixed for tail overfitting; ProbabilityCalibrator was not
# even checked. Isotonic is a step function over pooled-adjacent blocks, and at
# the extremes those blocks are tiny -- live output included exactly 0.333 and
# 0.667, i.e. blocks of three observations, handed to the EV gate as fact.
#
# Selection points the same way: the gate only ever fires on the most extreme
# probabilities, which are exactly the least-supported blocks. Backtested over
# 4068 out-of-sample bars, the isotonic-calibrated gate fired on 0.39% of bars
# and lost 0.34% per trade at a 37.5% win rate. With sigmoid it fires on none
# of them, and max strength falls from 0.990 to 0.652 -- which matches the
# observed top-decile up-rate of the best symbol (0.6435).

def _weak_signal(n=3000, seed=11):
    """A model with a little real skill and a wide spread of raw scores.

    Shaped like the real thing: xgboost emits confident-looking raw
    probabilities (live range 0.029-0.992) even when its actual discrimination
    is AUC 0.53, so the extremes are populated but *sparsely*.
    """
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.03, 0.97, size=n)
    # Barely-there signal: the score shifts the odds by a couple of points.
    y = (rng.uniform(size=n) < 0.5 + 0.06 * (raw - 0.5)).astype(int)
    return raw, y


def test_isotonic_manufactures_confidence_from_thin_blocks():
    """Documents the defect precisely, so the fix cannot be silently reverted.

    The mechanism is not randomness, it is arithmetic: isotonic's fitted value
    for the topmost block is the *mean of that block*. When the block holds
    three observations that happen to be up, the answer is 1.0 (clipped to
    0.99); when it holds two of three, the answer is exactly 0.667. Both were
    observed live. Neither is a probability estimated from data.
    """
    raw, y = _weak_signal()
    # The handful of highest-scoring bars happen to have gone up -- routine.
    raw = np.append(raw, [0.980, 0.985, 0.990])
    y = np.append(y, [1, 1, 1])

    iso = ProbabilityCalibrator(method="isotonic").fit(raw, y).transform(raw)
    sig = ProbabilityCalibrator(method="sigmoid").fit(raw, y).transform(raw)

    assert iso.max() > 0.90, "isotonic claims near-certainty from three bars"
    assert sig.max() < 0.75, "sigmoid cannot be moved that far by three bars"


def test_sigmoid_will_not_claim_more_than_the_data_supports():
    raw, y = _weak_signal()
    sig = ProbabilityCalibrator(method="sigmoid").fit(raw, y).transform(raw)
    assert sig.max() < 0.75
    assert sig.min() > 0.25


def test_the_calibrated_ceiling_tracks_the_observed_top_decile():
    """The honest ceiling: bars the model ranks highest actually go up this
    often, so no calibrated probability should exceed it by much."""
    raw, y = _weak_signal()
    top = raw >= np.quantile(raw, 0.9)
    observed = y[top].mean()
    sig = ProbabilityCalibrator(method="sigmoid").fit(raw, y).transform(raw)
    assert sig.max() < observed + 0.15


def test_sigmoid_is_monotone_so_it_cannot_destroy_ranking():
    """Calibration must fix the level, never the ordering -- the ordering is
    the only thing the model actually knows."""
    from sklearn.metrics import roc_auc_score

    raw, y = _weak_signal()
    sig = ProbabilityCalibrator(method="sigmoid").fit(raw, y).transform(raw)
    assert roc_auc_score(y, sig) == pytest.approx(roc_auc_score(y, raw), abs=1e-9)


def test_sigmoid_still_improves_calibration():
    raw, y = _weak_signal()
    sig = ProbabilityCalibrator(method="sigmoid").fit(raw, y).transform(raw)
    assert calibration_error(sig, y) < calibration_error(raw, y)


def test_the_gate_refuses_a_thin_block_masquerading_as_certainty():
    """End to end: a 0.99 from a three-sample block clears any realistic gate."""
    from src.strategy.economics import is_tradeable

    assert is_tradeable(0.99, 0.004), "the artifact would have opened a position"
    assert not is_tradeable(0.652, 0.004), "the honest ceiling does not"


def test_default_method_is_sigmoid():
    assert ProbabilityCalibrator().method == "sigmoid"
