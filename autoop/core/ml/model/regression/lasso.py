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
    def __init__(self) -> None:
        """
        Initializes the Lasso regression model by creating an instance of
        IMLasso.
        """
        super().__init__()
        self.lasso = Lasso()

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
        self.lasso.fit(observations, ground_truth)
        self._parameters = {
            "weights": self.lasso.coef_,
            "intercept": self.lasso.intercept_
        }

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
        predictions = self.lasso.predict(observations)
        return predictions

    def to_artifact(self, name) -> Artifact:
        artifact = Artifact(name,
                            "asset_path",
                            "1.0.0",
                            self._data.encode(),
                            "lasso regression",
                            self._parameters,
                            ["regression"]
                            )
        return artifact
