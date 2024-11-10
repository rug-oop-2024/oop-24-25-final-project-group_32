from autoop.core.ml.model.model import Model
from sklearn.linear_model import Lasso
from autoop.core.ml.artifact import Artifact
import numpy as np


class LassoWrapper(Model):
    """
    Lasso Regression model using L1 regularization.
    Lasso regression is a linear model that applies L1 regularization,
    which can shrink some coefficients to zero, effectively performing
    feature selection.
    Attributes:
        lasso (Lasso): An instance of a Lasso regression model.
        _parameters (dict): Dictionary to store the fitted model parameters
        (weights and intercept).
    """
    def __init__(self, *args, alpha=0.01, **kwargs) -> None:
        """
        Initializes the Lasso regression model by creating an instance of
        IMLasso.
        """
        super().__init__()
        self._lasso = Lasso(*args, alpha=alpha, **kwargs)
        new_parameters = self._lasso.get_params()
        self.parameters = new_parameters
        self._type = "regression"
        self._name = "Lasso Regression"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the Lasso regression model to the provided data.
        This method trains the model on the given observations (features)
        and ground truth
        (target) using L1 regularization. The learned parameters
        (weights and intercept) are stored in the _parameters attribute.
        Args:
            observations (np.ndarray): A 2D array of input features
            (observations).
            ground_truth (np.ndarray): A 1D array of target values
            corresponding to the observations.
        """
        self._lasso.fit(observations, ground_truth)
        self.parameters = {
            "coefficients": np.array(self._model.coef_),
            "intercept": np.atleast_1d(self._model.intercept_),
        }

        self._data = observations

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for a given set of observations using
        the trained Lasso model.
        Args:
            observations (np.ndarray): A 2D array of input features
            for which predictions are needed.
        Returns:
            np.ndarray: A 1D array of predicted target values.
        """
        predictions = self._lasso.predict(observations)
        return predictions.reshape(-1, 1)

    def to_artifact(self, name) -> Artifact:
        """
        Converts the model instance into an Artifact for storage or tracking.

        Args:
            name (str): The name to assign to the Artifact.

        Returns:
            Artifact: An Artifact instance representing the model,
            including its asset path, version, encoded data,
            type, parameters, and tags.
        """
        artifact = Artifact(
            name=name,
            asset_path="asset_path",
            version="1.0.0",
            encoded_data=self._data.tobytes(),
            model_type="k nearest",
            parameters=self._parameters,
            tags=["classification"]
        )
        return artifact
