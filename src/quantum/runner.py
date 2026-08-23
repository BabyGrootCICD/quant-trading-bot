import sys
import os
import json
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import SYMBOLS
from src.data.supabase_client import get_client


def fetch_historical_prices(client, symbols: list[str], limit: int = 720) -> pd.DataFrame:
    all_data = []
    for symbol in symbols:
        resp = (
            client.table("candles")
            .select("timestamp,close")
            .eq("symbol", symbol)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        if resp.data:
            df = pd.DataFrame(resp.data)
            df = df.rename(columns={"close": symbol, "timestamp": "timestamp"})
            df = df.set_index("timestamp")
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, axis=1)
    combined = combined.sort_index()
    return combined


def main():
    print("=" * 60)
    print("Quantum Optimization Pipeline")
    print("=" * 60)

    client = get_client()

    print("\n[1/4] Fetching historical prices...")
    prices = fetch_historical_prices(client, SYMBOLS)
    print(f"  Got {len(prices)} hours of data for {len(prices.columns)} symbols")

    if prices.empty or len(prices) < 100:
        print("  Insufficient data. Need at least 100 hours.")
        return

    results = {"timestamp": datetime.now(timezone.utc).isoformat(), "symbols": SYMBOLS}

    print("\n[2/4] Running quantum portfolio optimization...")
    try:
        from src.quantum.portfolio_optimizer import run_quantum_optimization
        opt_result = run_quantum_optimization(prices, SYMBOLS, use_simulator=True)
        results["portfolio_optimization"] = opt_result
        print(f"  Classical: {opt_result['classical']['num_selected']} assets selected")
        print(f"  Quantum:   {opt_result['quantum']['num_selected']} assets selected")
    except Exception as e:
        print(f"  Portfolio optimization failed: {e}")
        results["portfolio_optimization"] = {"error": str(e)}

    print("\n[3/4] Running Monte Carlo risk analysis...")
    try:
        from src.quantum.monte_carlo import run_risk_analysis
        risk_result = run_risk_analysis(prices, SYMBOLS)
        results["risk_analysis"] = risk_result
        for sym, data in risk_result["results"].items():
            print(f"  {sym}: VaR={data['historical_var_24h']:.4f} CVaR={data['historical_cvar_24h']:.4f}")
    except Exception as e:
        print(f"  Risk analysis failed: {e}")
        results["risk_analysis"] = {"error": str(e)}

    print("\n[4/4] Running quantum kernel comparison...")
    try:
        from src.quantum.quantum_kernel import compare_quantum_classical
        kernel_result = compare_quantum_classical(prices, SYMBOLS)
        results["kernel_comparison"] = kernel_result
        for sym, data in kernel_result.items():
            q_acc = data["quantum"]["test_accuracy"]
            c_acc = data["classical"]["test_accuracy"]
            advantage = data["quantum_advantage"]
            print(f"  {sym}: quantum={q_acc:.2%} classical={c_acc:.2%} advantage={advantage:+.2%}")
    except Exception as e:
        print(f"  Kernel comparison failed: {e}")
        results["kernel_comparison"] = {"error": str(e)}

    with open("quantum_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Results saved to quantum_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
