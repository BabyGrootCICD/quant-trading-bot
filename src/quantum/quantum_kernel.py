import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.quantum.config import IBM_QUANTUM_BACKEND


def prepare_features(prices: pd.DataFrame, window: int = 24) -> tuple[np.ndarray, np.ndarray]:
    returns = prices.pct_change().dropna()

    features = []
    labels = []

    for i in range(window, len(returns) - 1):
        feat = returns.iloc[i - window:i].mean().values
        current_return = returns.iloc[i].values
        next_return = returns.iloc[i + 1].values

        combined = np.concatenate([feat, current_return])
        features.append(combined)

        label = 1 if np.mean(next_return) > 0 else 0
        labels.append(label)

    return np.array(features), np.array(labels)


def train_quantum_svm(
    features: np.ndarray,
    labels: np.ndarray,
    use_simulator: bool = True,
    feature_map_reps: int = 2,
) -> dict:
    try:
        from qiskit_machine_learning.algorithms import QSVC
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit.primitives import Sampler
    except ImportError:
        raise ImportError(
            "Install qiskit-machine-learning: pip install qiskit-machine-learning"
        )

    num_features = features.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=feature_map_reps)

    if use_simulator:
        kernel = FidelityQuantumKernel(feature_map=feature_map, sampler=Sampler())
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(channel="ibm_quantum_platform")
        backend = service.backend(IBM_QUANTUM_BACKEND)
        kernel = FidelityQuantumKernel(feature_map=feature_map, sampler=Sampler(backend=backend))

    qsvc = QSVC(quantum_kernel=kernel)
    qsvc.fit(features, labels)

    train_accuracy = qsvc.score(features, labels)

    return {
        "model": qsvc,
        "train_accuracy": round(train_accuracy, 4),
        "num_features": num_features,
        "feature_map_reps": feature_map_reps,
    }


def train_classical_svm(features: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    svm = SVC(kernel="rbf", probability=True)
    svm.fit(features_scaled, labels)

    train_accuracy = svm.score(features_scaled, labels)

    return {
        "model": svm,
        "scaler": scaler,
        "train_accuracy": round(train_accuracy, 4),
    }


def compare_quantum_classical(
    prices: pd.DataFrame,
    symbols: list[str],
    window: int = 24,
) -> dict:
    results = {}

    for symbol in symbols:
        price_data = prices[[symbol]].dropna()
        if len(price_data) < window + 50:
            continue

        features, labels = prepare_features(price_data, window)
        if len(features) < 10:
            continue

        split = int(len(features) * 0.8)
        X_train, X_test = features[:split], features[split:]
        y_train, y_test = labels[:split], labels[split:]

        try:
            quantum = train_quantum_svm(X_train, y_train, use_simulator=True)
            quantum_test_acc = quantum["model"].score(X_test, y_test)
        except Exception as e:
            quantum = {"error": str(e), "train_accuracy": 0}
            quantum_test_acc = 0

        classical = train_classical_svm(X_train, y_train)
        classical_test_acc = classical["model"].score(
            classical["scaler"].transform(X_test), y_test
        )

        results[symbol] = {
            "quantum": {
                "train_accuracy": quantum.get("train_accuracy", 0),
                "test_accuracy": round(quantum_test_acc, 4),
            },
            "classical": {
                "train_accuracy": classical["train_accuracy"],
                "test_accuracy": round(classical_test_acc, 4),
            },
            "quantum_advantage": round(quantum_test_acc - classical_test_acc, 4),
        }

    return results
