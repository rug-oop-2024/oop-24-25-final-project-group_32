import numpy as np
from collections import Counter
from copy import deepcopy
from autoop.core.ml.model.model import Model
from autoop.core.ml.artifact import Artifact


class KNearestNeighbors(Model):
    """
    A k-Nearest Neighbors (KNN) classifier for supervised learning tasks.

    This KNN classifier predicts the label of a test instance
    based on the labels of its nearest neighbors in the training data.
    The distance between instances is calculated,
    and the most frequent label among the k-nearest neighbors is assigned
    as the prediction for a test instance.
    """
    def __init__(self, k: int = 3):
        """
        Initializes the KNearestNeighbors model.

        Args:
            k (int): Presents the amount of neighbours
            that will be used for the model predictions.
        """
        super().__init__()
        self.k: int = k
        self._type = "classification"

    @property
    def k(self) -> int:
        """
        Getter for the current k value

        Returns: The number of neighbors used in the model.
        """
        return deepcopy(self._k)

    @k.setter
    def k(self, k: int) -> None:
        """
        Setter for the value of k.
        Args:
            k (int): The new of neighbors to use.
        """
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        elif k <= 0:
            raise ValueError("k must be atleast 1")
        else:
            self._k = k

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the data by storing the observations and ground truth.

        Args:
            observations (np.ndarray): A numpy array
            containing data points (rows).
            ground_truth (np.ndarray): A numpy array
            containing corresponding labels.

        Raises:
            ValueError: If observations and ground_truth lengths do not match.
        """
        if observations.shape[0] != ground_truth.shape[0]:
            raise ValueError("The number of observations and"
                             " ground_truth labels must match.")

        self._parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }
        self._data = observations

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the label for each observation in the input.

        Args:
            observations (np.ndarray): A numpy array
            that contains one data point in each row.

        Returns: A numpy array of predicted labels
        for each of the input observation.
        """
        if self._parameters is None:
            raise ValueError(
                "Model not fitted. Call 'fit' with appropriated arguments")
        else:
            predictions = [self._predict_single_point(point) for
                           point in observations]
            return np.array(predictions)

    def _predict_single_point(self, data_point: np.ndarray):
        """
        Predicts the label for a single observation
        by looking at labels of the k-nearest neighbours.

        Args:
            data_point (np.ndarray): One data point
            for which the label wil be predicted

        Returns: the most common label among the k-nearest neighbors.
        """
        difference = np.linalg.norm(self._parameters[
            "observations"] - data_point, axis=1)
        k_indices = np.argsort(difference)[:self.k]
        k_nearest_labels = [self._parameters["ground_truth"][index].tolist()
                            for index in k_indices]
        final_k_nearest_labels = [str(label) for label in k_nearest_labels]
        prediction = Counter(final_k_nearest_labels).most_common()
        return prediction[0][0]

    def to_artifact(self, name: str) -> Artifact:
        """
        Converts the model instance into an Artifact for storage or tracking.

        Args:
            name (str): The name to assign to the Artifact.

        Returns:
            Artifact: An Artifact instance representing the model.
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
