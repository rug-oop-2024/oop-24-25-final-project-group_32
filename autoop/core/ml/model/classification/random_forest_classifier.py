from autoop.core.ml.model.model import Model
import numpy as np
from sklearn.ensemble import RandomForestClassifier


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
        self._name = "Random Forest Classifier"

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
        self.parameters = {"estimations": self._model.estimators_}

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
