
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.classification.random_forest_classifier import RandomForest
from autoop.core.ml.model.regression.ridge_regression import RidgeRegression
from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.classification.logistic_regression import LogisticRegression
from autoop.core.ml.model.classification.k_nearest_neighbors import KNearestNeighbors

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "RidgeRegression",
    "lassoRegression"
] # add your models as str here

CLASSIFICATION_MODELS = [
    "RandomForestClassifier",
    "LogisticRegression",
    "KNN"
] # add your models as str here

def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name in REGRESSION_MODELS:
        if model_name == "MultipleLinearRegression":
            return MultipleLinearRegression()
        if model_name == "RidgeRegression":
            return RidgeRegression()
        if model_name == "lassoRegression":
            return Lasso()
    elif model_name in CLASSIFICATION_MODELS:
        if model_name == "RandomForestClassifier":
            return RandomForest()
        if model_name == "LogisticRegression":
            return LogisticRegression()
        if model_name == "KNN":
            return KNearestNeighbors
    else:
        raise ValueError(f"Model {model_name} not supported.")
    raise NotImplementedError("To be implemented.")
