from autoop.core.ml.model.model import Model
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
        """
        Initializes the Multiple linear regression model by creating an
        instance of MultipleLinearRegression.
        """
        super().__init__()
        self._type = "regression"
        self._name = "Multiple Linear Regression"

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
        observation_matrix = np.c_[observation, np.ones(observation.shape[0])]

        transposed_observation_matrix = np.transpose(observation_matrix)

        product = np.matmul(transposed_observation_matrix, observation_matrix)
        if self._check_inversion:
            invert = np.linalg.inv(product)

        next_product = np.matmul(invert, transposed_observation_matrix)
        weights = np.matmul(next_product, ground_truth)
        self._parameters["weights"] = weights

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
        observation_matrix = np.c_[observation, np.ones(observation.shape[0])]
        return observation_matrix.dot(self._parameters["weights"])
