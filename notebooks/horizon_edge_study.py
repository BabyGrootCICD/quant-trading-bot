# Reproduces the measurement in docs/07-why-no-trades.md.
# Run: python notebooks/horizon_edge_study.py   (needs network access to OKX)
"""Does the directional edge survive at a longer horizon?

Everything downstream depends on the answer: EV = (2p-1)*E|move| - cost.
Moves grow ~sqrt(t) while cost is fixed, so a longer horizon helps *if* the
edge holds up. Nothing had measured that.
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/dennis_leedennis_lee/Documents/GitHub/quant-trading-bot")
import numpy as np, pandas as pd, ccxt
from src.features.engineer import engineer_features
from src.models.features import FEATURE_COLS
from src.models.logistic import LogisticModel
from src.strategy.economics import per_leg_cost_pct

ex = ccxt.okx({"enableRateLimit": True})
SYMS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT", "DOGE/USDT"]
HORIZONS = [1, 4, 12, 24]
taker, maker = 2*per_leg_cost_pct("taker"), 2*per_leg_cost_pct("maker")

def history(sym, pages=40):
    rows, end = [], ex.milliseconds()
    for _ in range(pages):
        c = ex.fetch_ohlcv(sym, "1h", limit=100, params={"after": end})
        if not c: break
        rows = c + rows
        end = c[0][0]
        time.sleep(0.08)
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

print(f"taker round trip {taker*100:.2f}%   maker {maker*100:.3f}%")
print("edge = balanced accuracy - 0.50 (immune to class skew); EV uses p = 0.5 + edge\n")
print(f"{'sym':10} {'h':>3} {'rows':>5} {'base':>6} {'bal.acc':>8} {'edge':>9} {'E|move|':>9} "
      f"{'EV taker':>9} {'EV maker':>9}")
summary = {}
for sym in SYMS:
    df = history(sym)
    if len(df) < 1000:
        print(f"{sym:10} only {len(df)} bars, skipping"); continue
    df["symbol"] = sym
    f = engineer_features(df)
    close = df["close"].reset_index(drop=True)
    for h in HORIZONS:
        fut = np.log(close.shift(-h) / close)
        g = f.copy()
        g["target_1h"] = np.where(fut > 0, 1.0, np.where(fut < 0, 0.0, np.nan))
        move = float(np.nanmean(np.abs(fut)))
        g = g.dropna(subset=["target_1h"] + FEATURE_COLS)
        if len(g) < 600: continue
        r = LogisticModel().fit_walk_forward(g, n_splits=5)
        if "error" in r: continue
        y, p = np.array(r["oos_y_true"]), np.array(r["oos_preds"])
        up = y.mean(); base = max(up, 1-up)
        rec = [ (p[y==c]==c).mean() for c in (0,1) if (y==c).any() ]
        bal = float(np.mean(rec))
        edge = bal - 0.5
        strength = 0.5 + max(edge, 0.0)
        ev_t = (2*strength-1)*move - taker
        ev_m = (2*strength-1)*move - maker
        summary.setdefault(h, []).append((ev_t, ev_m, edge))
        print(f"{sym:10} {h:>3} {len(g):>5} {base:>6.3f} {bal:>8.4f} {edge*100:>+8.2f}pp "
              f"{move*100:>8.3f}% {ev_t*100:>+8.3f}% {ev_m*100:>+8.3f}%")
    print()

print("=== mean across symbols ===")
print(f"{'h':>3} {'edge':>9} {'EV taker':>10} {'EV maker':>10}  tradeable(maker)")
for h in HORIZONS:
    if h not in summary: continue
    t = np.mean([x[0] for x in summary[h]]); m = np.mean([x[1] for x in summary[h]])
    e = np.mean([x[2] for x in summary[h]]); n = sum(1 for x in summary[h] if x[1] > 0)
    print(f"{h:>3} {e*100:>+8.2f}pp {t*100:>+9.3f}% {m*100:>+9.3f}%  {n}/{len(summary[h])}")
