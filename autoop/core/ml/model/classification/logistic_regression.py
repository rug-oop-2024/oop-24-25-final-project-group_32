from sklearn.linear_model import LogisticRegression
from autoop.core.ml.model.model import Model
from autoop.core.ml.artifact import Artifact
import numpy as np

class LogisticRegression(Model):
    def __init__(self):
        super().__init__()
        self._model = LogisticRegression()

    def fit(self, observation: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the logistic regression model to the provided data.
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
                            "logistic regression",
                            self._parameters,
                            ["classification"]
                            )
        return artifact