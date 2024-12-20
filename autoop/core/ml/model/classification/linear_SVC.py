from sklearn.svm import LinearSVC
from autoop.core.ml.model.model import Model
import numpy as np


class WrapperLinearSVC(Model):
    """
    LinearSVC Classifier model for predicting a categorical
    target variable.
    This model computes the optimal weights (parameters) for a given
    set of observations
    (features) and ground truth (target variable)
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the wrapper logistic regression model by creating an
        instance of WrapperLogisticRegression.
        """
        super().__init__()
        self._model = LinearSVC(*args, **kwargs)
        new_parameters = self._model.get_params()
        self._parameters = new_parameters
        self._type = "classification"

    def fit(self, observation: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the logistic regression model to the provided data.
        Args:
            observation (np.ndarray): A 2D array of input features
            (observations).
            ground_truth (np.ndarray): A 1D array of target values
            corresponding to the observations.
        """
        if ground_truth.ndim > 1:
            ground_truth = np.argmax(ground_truth, axis=1)
        self._model.fit(observation, ground_truth)
        self._parameters = {
            "coefficients": np.array(self._model.coef_),
            "intercept": np.atleast_1d(self._model.intercept_),
        }

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
