import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.quantum.config import MONTE_CARLO_SHOTS, IBM_QUANTUM_BACKEND


def compute_var_classical(returns: np.ndarray, confidence: float = 0.95) -> float:
    return float(np.percentile(returns, (1 - confidence) * 100))


def compute_cvar_classical(returns: np.ndarray, confidence: float = 0.95) -> float:
    var = compute_var_classical(returns, confidence)
    return float(np.mean(returns[returns <= var]))


def run_monte_carlo_simulation(
    prices: pd.Series,
    num_paths: int = 1000,
    horizon: int = 24,
    confidence: float = 0.95,
) -> dict:
    returns = prices.pct_change().dropna().values
    mu = np.mean(returns)
    sigma = np.std(returns)

    np.random.seed(42)
    simulated_returns = np.random.normal(mu, sigma, (num_paths, horizon))
    cumulative_returns = np.cumprod(1 + simulated_returns, axis=1)
    terminal_values = cumulative_returns[:, -1]

    var = compute_var_classical(terminal_values - 1, confidence)
    cvar = compute_cvar_classical(terminal_values - 1, confidence)
    expected_return = float(np.mean(terminal_values - 1))
    worst_case = float(np.min(terminal_values - 1))
    best_case = float(np.max(terminal_values - 1))

    return {
        "num_paths": num_paths,
        "horizon_hours": horizon,
        "confidence": confidence,
        "expected_return": round(expected_return, 6),
        "var": round(var, 6),
        "cvar": round(cvar, 6),
        "worst_case": round(worst_case, 6),
        "best_case": round(best_case, 6),
        "mean_terminal": round(float(np.mean(terminal_values)), 6),
        "std_terminal": round(float(np.std(terminal_values)), 6),
    }


def estimate_var_amplitude_estimation(
    prices: pd.Series,
    horizon: int = 24,
    confidence: float = 0.95,
    num_eval_qubits: int = 5,
    use_simulator: bool = True,
) -> dict:
    try:
        from qiskit_finance.circuit.library import LogNormalDistribution
        from qiskit_finance.applications import EuropeanCallPricing
        from qiskit.primitives import Sampler
        from qiskit_algorithms import AmplitudeEstimation
    except ImportError:
        raise ImportError(
            "Install qiskit-finance: pip install qiskit-finance qiskit-algorithms"
        )

    returns = prices.pct_change().dropna().values
    mu = np.mean(returns)
    sigma = np.std(returns)

    num_qubits = 3
    bounds = [(0, mu + 3 * sigma)]
    mvnd = LogNormalDistribution(num_qubits, mu=mu, sigma=sigma, bounds=bounds)

    strike_price = float(prices.iloc[-1])
    european_call = EuropeanCallPricing(
        num_state_qubits=num_qubits,
        Strikes=strike_price * np.ones(1),
        prices=mvnd,
        rescaling_factor=0.1,
        bounds=bounds,
    )

    problem = european_call.to_estimation_problem()

    if use_simulator:
        sampler = Sampler()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

        service = QiskitRuntimeService(channel="ibm_quantum_platform")
        backend = service.backend(IBM_QUANTUM_BACKEND)
        sampler = Sampler(backend=backend)

    ae = AmplitudeEstimation(num_eval_qubits=num_eval_qubits, sampler=sampler)
    result = ae.estimate(problem)

    estimated_value = european_call.interpret(result)

    return {
        "method": "amplitude_estimation",
        "estimated_call_value": round(float(estimated_value), 4),
        "max_probability": round(float(result.max_probability), 4),
        "num_eval_qubits": num_eval_qubits,
        "strike_price": strike_price,
    }


def run_risk_analysis(
    prices_df: pd.DataFrame,
    symbols: list[str],
    confidence: float = 0.95,
) -> dict:
    results = {}
    for symbol in symbols:
        prices = prices_df[symbol].dropna()
        if len(prices) < 100:
            continue

        mc = run_monte_carlo_simulation(prices, confidence=confidence)
        var_24h = compute_var_classical(prices.pct_change().dropna().values, confidence)
        cvar_24h = compute_cvar_classical(prices.pct_change().dropna().values, confidence)

        results[symbol] = {
            "monte_carlo": mc,
            "historical_var_24h": round(var_24h, 6),
            "historical_cvar_24h": round(cvar_24h, 6),
            "current_price": round(float(prices.iloc[-1]), 4),
        }

    return {
        "symbols": list(results.keys()),
        "confidence": confidence,
        "results": results,
    }
