from sklearn.linear_model import Ridge
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model.model import Model
import numpy as np

class RidgeRegression(Model):
    def __init__(self):
        super().__init__()
        self._model = Ridge()

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
        self._parameters = {
            "weights": self._model.get_params()
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

    def to_artifact(self, name) -> Artifact:
        artifact = Artifact(name,
                            "asset_path",
                            "1.0.0",
                            self._data.encode(),
                            "ridge regression",
                            self._parameters,
                            ["regression"]
                            )
        return artifact