from autoop.core.ml.model.model import Model
from autoop.core.ml.artifact import Artifact
import numpy as np


class MultipleLinearRegression(Model):
    """
    Multiple Linear Regression model for predicting a continuous
    target variable.
    This model computes the optimal weights (parameters) for a given
    set of observations
    (features) and ground truth (target variable) using the
    Normal Equation method.
    """

    def __init__(self) -> None:
        super().__init__()

    def _check_inversion(self, matrix) -> bool:
        """
        Checks if a matrix is invertible by calculating its determinant.
        Args:
            matrix (np.ndarray): The matrix to check for invertibility.
        Raises:
            ValueError: If the matrix is singular (determinant is 0) and not
            invertible.
        Returns:
            bool: True if the matrix is invertible.
        """
        det = np.linalg.det(matrix)
        if det == 0:
            raise ValueError("Matrix is singular and not invertible.")
        return True

    def fit(self, observation: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the multiple linear regression model to the provided data.
        This method computes the optimal weights using the Normal Equation:
        (X^T X)⁻¹ X^T y, where X is the observation matrix and y is the
        ground truth.
        Args:
            observation (np.ndarray): A 2D array of input features
            (observations).
            ground_truth (np.ndarray): A 1D array of target values
            corresponding to the observations.
        """
        # Add a column of ones for the intercept term (bias)
        observation_matrix = np.c_[observation, np.ones(observation.shape[0])]

        # Compute the Normal Equation: (X^T X)⁻¹ X^T y
        transposed_observation_matrix = np.transpose(observation_matrix)
        # Compute X^T X and check if it is invertable
        product = np.matmul(transposed_observation_matrix, observation_matrix)
        if self._check_inversion:
            invert = np.linalg.inv(product)
        # Compute (X^T X)⁻¹ X^T under the name next_product
        next_product = np.matmul(invert, transposed_observation_matrix)
        weights = np.matmul(next_product, ground_truth)
        self._parameters["weights"] = weights
        self._data = observation

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for a given set of observations using
        the trained model.
        Args:
            observation (np.ndarray): A 2D array of input features for which
            predictions are needed.
        Returns:
            np.ndarray: A 1D array of predicted target values.
        """
        # Add a column of ones for the intercept term (bias)
        observation_matrix = np.c_[observation, np.ones(observation.shape[0])]
        return observation_matrix.dot(self._parameters["weights"])
    
    def to_artifact(self, name) -> Artifact:
        artifact = Artifact(name,
                            "asset_path",
                            "1.0.0",
                            self._data.encode(),
                            "multiple linear regression",
                            self._parameters,
                            ["regression"]
                            )
        return artifact