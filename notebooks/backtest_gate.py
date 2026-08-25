# Reproduces docs/09-the-gate-was-firing-on-artifacts.md.
# Run: python notebooks/backtest_gate.py   (needs network access to OKX)
"""Backtest the live EV gate, out of sample, exactly as the engine applies it.

For each fold: fit the direction head and the magnitude head on train only,
score test, then apply is_tradeable() to every test bar and settle the ones
that fire against the actual next-bar return.
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/dennis_leedennis_lee/Documents/GitHub/quant-trading-bot")
import numpy as np, pandas as pd, ccxt
from sklearn.model_selection import TimeSeriesSplit
from src.features.engineer import engineer_features
from src.models.features import FEATURE_COLS
from src.models.xgboost_model import XGBoostModel
from src.models.magnitude import MagnitudeModel
from src.strategy.economics import is_tradeable, expected_value, round_trip_cost_pct, per_leg_cost_pct

ex = ccxt.okx({"enableRateLimit": True})
def hist(sym, pages=40):
    rows, end = [], ex.milliseconds()
    for _ in range(pages):
        c = ex.fetch_ohlcv(sym, "1h", limit=100, params={"after": end})
        if not c: break
        rows = c + rows; end = c[0][0]; time.sleep(0.05)
    return pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

TAKER, MAKER = 2*per_leg_cost_pct("taker"), 2*per_leg_cost_pct("maker")
SYMS = ["BTC/USDT","ETH/USDT","SOL/USDT","ADA/USDT","XRP/USDT","DOGE/USDT"]
allrows = []

for sym in SYMS:
    h = hist(sym); h["symbol"] = sym
    f = engineer_features(h)
    close = h["close"].reset_index(drop=True)
    f["fwd_ret"] = np.log(close.shift(-1)/close)       # what a 1h position actually earns
    f = f.dropna(subset=["target_1h","target_move_1h","fwd_ret"]+FEATURE_COLS).reset_index(drop=True)
    X = f[FEATURE_COLS].to_numpy(float)
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        trd, ted = f.iloc[tr], f.iloc[te]
        if trd["target_1h"].nunique() < 2: continue
        clf = XGBoostModel()
        if "error" in clf.fit_walk_forward(trd, n_splits=3): continue
        mag = MagnitudeModel()
        if "error" in mag.fit_walk_forward(trd, n_splits=3): continue
        p = clf.predict(ted)["probability_up"].to_numpy(float)
        mv = mag.predict(ted).to_numpy(float)
        for pi, mi, fr in zip(p, mv, ted["fwd_ret"].to_numpy(float)):
            if not (np.isfinite(pi) and np.isfinite(mi)): continue
            side = 1 if pi > 0.5 else -1
            allrows.append({"sym": sym, "p": pi, "strength": max(pi, 1-pi),
                            "move": mi, "side": side, "fwd": fr})

d = pd.DataFrame(allrows)
print("=== SIGMOID (Platt) calibration ===")
print(f"{len(d)} out-of-sample bars across {d.sym.nunique()} symbols\n")
print(f"calibrated strength: median {d.strength.median():.3f}  p90 {d.strength.quantile(.9):.3f}  max {d.strength.max():.3f}")
print(f"forecast E|move|:    median {d.move.median()*100:.3f}%  p90 {d.move.quantile(.9)*100:.3f}%  max {d.move.max()*100:.3f}%\n")

for label, cost in (("taker 0.30%", TAKER), ("maker 0.08%", MAKER)):
    fire = d.apply(lambda r: is_tradeable(r.strength, r.move, cost=cost), axis=1)
    n = int(fire.sum())
    print(f"--- gate at {label}: fires on {n}/{len(d)} bars ({n/len(d)*100:.2f}%) ---")
    if n:
        t = d[fire].copy()
        t["pnl"] = t.side * t.fwd - cost
        print(f"    mean PnL/trade {t.pnl.mean()*100:+.4f}%   win rate {(t.pnl>0).mean()*100:.1f}%"
              f"   total {t.pnl.sum()*100:+.2f}%   median E|move| {t.move.median()*100:.3f}%")
        print(f"    by symbol: " + ", ".join(f"{s}:{len(g)}" for s,g in t.groupby('sym')))
    print()
