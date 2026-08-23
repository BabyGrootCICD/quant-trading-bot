import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.quantum.config import (
    DEFAULT_RISK_FACTOR,
    DEFAULT_BUDGET_RATIO,
    QAOA_REPS,
    QAOA_MAX_ITER,
    IBM_QUANTUM_BACKEND,
)


def compute_expected_returns(prices: pd.DataFrame) -> np.ndarray:
    returns = prices.pct_change().dropna()
    return returns.mean().values


def compute_covariance_matrix(prices: pd.DataFrame) -> np.ndarray:
    returns = prices.pct_change().dropna()
    return returns.cov().values


def optimize_portfolio_qaoa(
    expected_returns: np.ndarray,
    covariances: np.ndarray,
    risk_factor: float = DEFAULT_RISK_FACTOR,
    budget: int = None,
    use_simulator: bool = True,
) -> dict:
    try:
        from qiskit_finance.applications.optimization import PortfolioOptimization
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
    except ImportError:
        raise ImportError(
            "Install qiskit-finance: pip install qiskit-finance qiskit-optimization"
        )

    num_assets = len(expected_returns)
    if budget is None:
        budget = max(1, int(num_assets * DEFAULT_BUDGET_RATIO))

    portfolio = PortfolioOptimization(
        expected_returns=expected_returns,
        covariances=covariances,
        risk_factor=risk_factor,
        budget=budget,
    )
    qp = portfolio.to_quadratic_program()

    if use_simulator:
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit.primitives import Sampler

        cobyla = COBYLA()
        cobyla.set_options(maxiter=QAOA_MAX_ITER)
        qaoa = QAOA(sampler=Sampler(), optimizer=cobyla, reps=QAOA_REPS)
        optimizer = MinimumEigenOptimizer(qaoa)
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_ibm_runtime import Sampler

        service = QiskitRuntimeService(channel="ibm_quantum_platform")
        backend = service.backend(IBM_QUANTUM_BACKEND)
        cobyla = COBYLA()
        cobyla.set_options(maxiter=QAOA_MAX_ITER)
        sampler = Sampler(backend=backend)
        qaoa = QAOA(sampler=sampler, optimizer=cobyla, reps=QAOA_REPS)
        optimizer = MinimumEigenOptimizer(qaoa)

    result = optimizer.solve(qp)

    selected = [i for i, x in enumerate(result.x) if x == 1]
    portfolio_return = float(np.dot(expected_returns[selected], np.ones(len(selected))))
    portfolio_risk = float(np.dot(np.ones(len(selected)), np.dot(covariances[np.ix_(selected, selected)], np.ones(len(selected)))))

    return {
        "selected_assets": selected,
        "num_selected": len(selected),
        "num_total": num_assets,
        "expected_return": portfolio_return,
        "portfolio_risk": portfolio_risk,
        "objective_value": result.fval,
        "raw_result": result.x.tolist(),
    }


def optimize_portfolio_classical(
    expected_returns: np.ndarray,
    covariances: np.ndarray,
    risk_factor: float = DEFAULT_RISK_FACTOR,
    budget: int = None,
) -> dict:
    from qiskit_finance.applications.optimization import PortfolioOptimization
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms import NumPyMinimumEigensolver

    num_assets = len(expected_returns)
    if budget is None:
        budget = max(1, int(num_assets * DEFAULT_BUDGET_RATIO))

    portfolio = PortfolioOptimization(
        expected_returns=expected_returns,
        covariances=covariances,
        risk_factor=risk_factor,
        budget=budget,
    )
    qp = portfolio.to_quadratic_program()

    exact_solver = NumPyMinimumEigensolver()
    optimizer = MinimumEigenOptimizer(exact_solver)
    result = optimizer.solve(qp)

    selected = [i for i, x in enumerate(result.x) if x == 1]
    portfolio_return = float(np.dot(expected_returns[selected], np.ones(len(selected))))
    portfolio_risk = float(np.dot(np.ones(len(selected)), np.dot(covariances[np.ix_(selected, selected)], np.ones(len(selected)))))

    return {
        "selected_assets": selected,
        "num_selected": len(selected),
        "num_total": num_assets,
        "expected_return": portfolio_return,
        "portfolio_risk": portfolio_risk,
        "objective_value": result.fval,
        "raw_result": result.x.tolist(),
    }


def run_quantum_optimization(prices_df: pd.DataFrame, symbols: list[str], use_simulator: bool = True) -> dict:
    prices = prices_df[symbols]
    mu = compute_expected_returns(prices)
    sigma = compute_covariance_matrix(prices)

    classical = optimize_portfolio_classical(mu, sigma)
    quantum = optimize_portfolio_qaoa(mu, sigma, use_simulator=use_simulator)

    return {
        "classical": classical,
        "quantum": quantum,
        "symbols": symbols,
        "risk_factor": DEFAULT_RISK_FACTOR,
    }
