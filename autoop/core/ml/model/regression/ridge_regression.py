from sklearn.linear_model import Ridge
from autoop.core.ml.model.model import Model
import numpy as np


class RidgeRegression(Model):
    """
    A Ridge Regression model for linear regression with L2 regularization.

    Ridge Regression minimizes the least squares error
    with an added penalty term proportional to the square of
    the magnitude of the coefficients. This helps to prevent overfitting
    by shrinking the coefficients, making it well-suited for problems with
    multicollinearity or when there are more features than samples.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes ridge regression model by creating an
        instance of RidgeRegression.
        """
        super().__init__()
        self._model = Ridge(*args, **kwargs)
        new_parameters = self._model.get_params()
        self.parameters = new_parameters
        self._type = "regression"
        self._name = "Ridge Regression"

    def fit(self, observation: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the Ridge regression model to the provided data.
        Args:
            observation (np.ndarray): A 2D array of input features
            (observations).
            ground_truth (np.ndarray): A 1D array of target values
            corresponding to the observations.
        """
        self._model.fit(observation, ground_truth)
        self.parameters = {
            "coefficients": np.array(self._model.coef_),
            "intercept": np.atleast_1d(self._model.intercept_),
        }
        self._data = observation

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for a given set of observations.
        Args:
            observation (np.ndarray): A 2D array of input features
            (observations).
        Returns:
            np.ndarray: A 1D array of predicted target values.
        """
        return self._model.predict(observation)
