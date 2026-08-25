# Reproduces the churn measurement in docs/10-model-generation.md.
"""Does averaging the walk-forward folds stabilise the served prediction?

Simulates the pipeline: retrain from scratch each hour on data up to that
hour, then predict the newest bar. Measures how often the directional call
reverses between consecutive hours -- every reversal on an open position pays
a round trip.
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/dennis_leedennis_lee/Documents/GitHub/quant-trading-bot")
import numpy as np, pandas as pd, ccxt
from src.features.engineer import engineer_features
from src.models.features import FEATURE_COLS
from src.models.xgboost_model import XGBoostModel

ex = ccxt.okx({"enableRateLimit": True})
def hist(sym, pages=30):
    rows, end = [], ex.milliseconds()
    for _ in range(pages):
        c = ex.fetch_ohlcv(sym, "1h", limit=100, params={"after": end})
        if not c: break
        rows = c + rows; end = c[0][0]; time.sleep(0.05)
    return pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

HOURS = 24
out = []
for sym in ("BTC/USDT", "ETH/USDT", "DOGE/USDT", "ADA/USDT"):
    h = hist(sym); h["symbol"] = sym
    f = engineer_features(h).dropna(subset=["target_1h"] + FEATURE_COLS).reset_index(drop=True)
    last, ens = [], []
    for k in range(HOURS, 0, -1):
        win = f.iloc[:len(f)-k]                     # data available at that hour
        if len(win) < 700: continue
        m = XGBoostModel()
        if "error" in m.fit_walk_forward(win, n_splits=5): continue
        row = win.iloc[[-1]]
        X = row[FEATURE_COLS].to_numpy(float)
        # current behaviour: last fold only
        s0, m0 = m.fold_models[-1]
        last.append(float(m0.predict_proba(s0.transform(X))[:, 1][0]))
        # proposed: average the folds
        ens.append(float(m._raw_proba(X)[0]))
    def churn(v):
        v = np.array(v); side = np.sign(v - 0.5)
        return (side[1:] != side[:-1]).mean(), np.abs(np.diff(v)).mean()
    fl, dl = churn(last); fe, de = churn(ens)
    out.append((sym, len(last), fl, fe, dl, de))
    print(f"  {sym:10} n={len(last):>2}  flip {fl:.1%} -> {fe:.1%}   |dp| {dl:.4f} -> {de:.4f}")

d = pd.DataFrame(out, columns=["sym","n","flip_last","flip_ens","dp_last","dp_ens"])
print(f"\n  MEAN flip rate: last-fold {d.flip_last.mean():.1%}  ->  fold-ensemble {d.flip_ens.mean():.1%}")
print(f"  MEAN |dp|/hour: last-fold {d.dp_last.mean():.4f}  ->  fold-ensemble {d.dp_ens.mean():.4f}")
cost = 0.0008
print(f"\n  churn cost per open position per hour @0.08% maker:")
print(f"    before {d.flip_last.mean()*cost*100:.4f}%   after {d.flip_ens.mean()*cost*100:.4f}%")
