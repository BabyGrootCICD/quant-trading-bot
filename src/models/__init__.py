from src.models.logistic import LogisticModel
from src.models.magnitude import MagnitudeModel
from src.models.neural import NeuralModel
from src.models.xgboost_model import XGBoostModel

# Keyed by each class's own `name`, not by a hand-typed string. The old map
# said "logistic_v1" while LogisticModel.name was "logistic_v2", so
# get_model(LogisticModel.name) raised ValueError.
MODEL_REGISTRY = {
    LogisticModel.name: LogisticModel,
    XGBoostModel.name: XGBoostModel,
    NeuralModel.name: NeuralModel,
}

# The magnitude head is not interchangeable with the directional models -- it
# answers "how far", not "which way" -- so it is kept out of MODEL_REGISTRY
# and exported on its own.
__all__ = ["LogisticModel", "XGBoostModel", "NeuralModel", "MagnitudeModel",
           "MODEL_REGISTRY", "get_model"]


def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()
