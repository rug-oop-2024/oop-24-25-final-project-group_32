"""The source module contains all the models."""
from autoop.core.ml.model.model import (
    Model)
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression)
from autoop.core.ml.model.classification.random_forest_classifier import (
    RandomForest)
from autoop.core.ml.model.regression.ridge_regression import (
    RidgeRegression)
from autoop.core.ml.model.regression.lasso import (
    Lasso)
from autoop.core.ml.model.classification.linear_SVC import (
    WrapperLinearSVC)
from autoop.core.ml.model.classification.k_nearest_neighbors import (
    KNearestNeighbors)

from typing import Optional

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "RidgeRegression",
    "lassoRegression",
]

CLASSIFICATION_MODELS = [
    "RandomForestClassifier",
    "linearSVC",
    "KNN",
]


def get_model_types(type: str = "both") -> list:
    """
    Function that returns the available models for the
    inputted data

    Args:
        type (str): The type of the input data. Defaults to "both"

    Returns:
        list: A list containing the available models for the inputted data
    """
    if type == "regression":
        return REGRESSION_MODELS
    elif type == "classification":
        return CLASSIFICATION_MODELS
    elif type == "both":
        return REGRESSION_MODELS + CLASSIFICATION_MODELS


def get_model(model_name: str) -> Optional[Model]:
    """
    Factory function to retrieve a model by name.

    Args:
        model_name (str): The name of the model to retrieve.

    Returns:
        Optional[Model]: An instance of the specified model, if found.
                         Returns None if no matching model is found.
    """
    if model_name not in REGRESSION_MODELS + CLASSIFICATION_MODELS:
        print(f"No such model `{model_name}` found.")
        return None

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
        if model_name == "linearSVC":
            return WrapperLinearSVC()
        if model_name == "KNN":
            return KNearestNeighbors()
