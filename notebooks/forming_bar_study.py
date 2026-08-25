# Reproduces the measurement in docs/08-forming-bar.md.
# Run: python notebooks/forming_bar_study.py   (needs network access to OKX)
"""How wrong is the feature vector of a still-forming hourly bar?

The trainer predicts on the newest row of the features frame and writes that
as THE prediction. The newest candle the fetcher stores is the bar currently
forming, so the traded prediction is computed from a partial bar.
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/dennis_leedennis_lee/Documents/GitHub/quant-trading-bot")
import numpy as np, pandas as pd, ccxt
from src.features.engineer import engineer_features

ex = ccxt.okx({"enableRateLimit": True})
rows, end = [], ex.milliseconds()
for _ in range(60):
    c = ex.fetch_ohlcv("BTC/USDT", "5m", limit=100, params={"after": end})
    if not c: break
    rows = c + rows; end = c[0][0]; time.sleep(0.05)
m5 = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"]).drop_duplicates("timestamp").sort_values("timestamp")
m5["hour"] = (m5.timestamp // 3_600_000) * 3_600_000
m5["min_into"] = (m5.timestamp % 3_600_000) // 60_000
hours = sorted(m5.hour.unique())
hours = [h for h in hours if (m5.hour == h).sum() == 12]      # complete hours only
m5 = m5[m5.hour.isin(hours)]
print(f"{len(m5)} 5m bars -> {len(hours)} complete hourly bars\n")

def hourly(upto_min):
    out = []
    for i, h in enumerate(hours):
        g = m5[m5.hour == h]
        if i == len(hours) - 1:
            g = g[g.min_into < upto_min]
            if g.empty: return None
        out.append({"timestamp": h, "open": g.open.iloc[0], "high": g.high.max(),
                    "low": g.low.min(), "close": g.close.iloc[-1], "volume": g.volume.sum()})
    df = pd.DataFrame(out); df["symbol"] = "BTC/USDT"
    return df

COLS = ["volume_ratio","volume_ratio_48","log_return_1h","atr_14_pct","rsi_14","bb_width"]
f_full = engineer_features(hourly(60))
full = f_full.iloc[-1]
hist = f_full.iloc[:-1][COLS].dropna()

print("The final row is the one that gets predicted and traded.\n")
print(f"  {'as of':>8} " + " ".join(f"{c:>13}" for c in COLS))
print(f"  {'60 (real)':>8} " + " ".join(f"{full[c]:>13.4f}" for c in COLS))
for m in (5, 10, 20, 30, 45):
    h = hourly(m)
    if h is None: continue
    r = engineer_features(h).iloc[-1]
    print(f"  {m:>8} " + " ".join(f"{r[c]:>13.4f}" for c in COLS))

print("\npercentile of the partial-bar value within the distribution of COMPLETED bars")
print("(a model trained only on completed bars has never seen anything outside ~0-100)")
print(f"  {'as of':>8} " + " ".join(f"{c:>13}" for c in COLS))
for m in (5, 10, 20, 30, 45, 60):
    h = hourly(m)
    if h is None: continue
    r = engineer_features(h).iloc[-1]
    cells = []
    for c in COLS:
        p = (hist[c] < r[c]).mean()*100
        cells.append(f"{p:>11.1f}%" + ("!" if p < 2 or p > 98 else " "))
    print(f"  {m:>8} " + " ".join(cells))
