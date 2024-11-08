from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from autoop.core.ml.artifact import Artifact


class RandomForest(Model):
    """
    Random Forest Classifier model for predicting a categorical
    target variable.
    This model computes the optimal weights (parameters) for a given
    set of observations
    (features) and ground truth (target variable) using the
    Normal Equation method.
    """
    def __init__(self):
        super().__init__()
        self._model = RandomForestClassifier()
        self._type = "RandomForestClassifier"

    def fit(self, observation: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the random forest classifier model to the provided data.
        Args:
            observation (np.ndarray): A 2D array of input features
            (observations).
            ground_truth (np.ndarray): A 1D array of target values
            corresponding to the observations.
        """
        self._model.fit(observation, ground_truth)
        self._parameters = {
            "parameters": self._model.get_params()
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
                            "random forest classifier",
                            self._parameters,
                            ["classification"]
                            )
        return artifact
