from src.models.logistic import LogisticModel
from src.models.xgboost_model import XGBoostModel

MODEL_REGISTRY = {
    "logistic_v1": LogisticModel,
    "xgboost_v1": XGBoostModel,
}


def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()
