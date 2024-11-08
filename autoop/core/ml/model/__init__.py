
from autoop.core.ml.model.model import (
    Model)
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression)
from autoop.core.ml.model.classification.random_forest_classifier import (
    RandomForest)
from autoop.core.ml.model.regression.ridge_regression import (
    RidgeRegression)
from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.classification.logistic_regression import (
    LogisticRegression)
from autoop.core.ml.model.classification.k_nearest_neighbors import (
    KNearestNeighbors)

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "RidgeRegression",
    "lassoRegression"
]  # add your models as str here

CLASSIFICATION_MODELS = [
    "RandomForestClassifier",
    "LogisticRegression",
    "KNN"
]  # add your models as str here


def get_model(model_name: str) -> Model:
    if model_name not in REGRESSION_MODELS and CLASSIFICATION_MODELS:
        print(f"No such model `{model_name}` found.")

    if model_name in REGRESSION_MODELS:
        type_ = "regression"
    else:
        type_ = "classification"

    """Factory function to get a model by name."""
    if model_name in REGRESSION_MODELS:
        if model_name == "MultipleLinearRegression":
            return MultipleLinearRegression(type=type_)
        if model_name == "RidgeRegression":
            return RidgeRegression(type=type_)
        if model_name == "lassoRegression":
            return Lasso(type=type_)
    elif model_name in CLASSIFICATION_MODELS:
        if model_name == "RandomForestClassifier":
            return RandomForest(type=type_)
        if model_name == "LogisticRegression":
            return LogisticRegression(type=type_)
        if model_name == "KNN":
            return KNearestNeighbors(type=type_)
