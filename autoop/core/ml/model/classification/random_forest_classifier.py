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
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the random forest model by creating an
        instance of RandomForest.
        """
        super().__init__()
        self._model = RandomForestClassifier(*args, **kwargs)
        new_parameters = self._model.get_params()
        self.parameters = new_parameters
        self._type = "classification"

    def fit(self, observation: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the random forest classifier model to the provided data.
        Args:
            observation (np.ndarray): A 2D array of input features
            (observations).
            ground_truth (np.ndarray): A 1D array of target values
            corresponding to the observations.
        """
        if ground_truth.ndim > 1:
            ground_truth = np.argmax(ground_truth, axis=1)

        self._model.fit(observation, ground_truth)
        self.parameters = {"estimations": np.array(self._model.estimators_)}
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
