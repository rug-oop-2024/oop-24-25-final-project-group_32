
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.core.ml.model.classification.random_forest_classifier import RandomForestClassifier

REGRESSION_MODELS = [
    "MultipleLinearRegression"
] # add your models as str here

CLASSIFICATION_MODELS = [
    "RandomForestClassifier"
] # add your models as str here

def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    if model_name in REGRESSION_MODELS:
        if model_name == "MultipleLinearRegression":
            return MultipleLinearRegression()
    elif model_name in CLASSIFICATION_MODELS:
        if model_name == "RandomForestClassifier":
            return RandomForestClassifier()
    else:
        raise ValueError(f"Model {model_name} not supported.")
    raise NotImplementedError("To be implemented.")