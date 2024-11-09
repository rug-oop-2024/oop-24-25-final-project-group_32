from abc import ABC, abstractmethod
import numpy as np

METRICS = [
    "accuracy",
    "mean_squared_error",
    "root_mean_squared_error",
    "logarithmic_loss",
    "precision",
    "r_squared"
]

def get_metrics() -> list:
    return METRICS

class Metric(ABC):
    """
    Base class for all metrics.
    """

    @abstractmethod
    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray
                 ) -> float:
        """
        Evaluate the metric given ground truth and predicted values.

        Args:
            prediction (np.ndarray): The predicted values.
            ground_truth (np.ndarray): The true values (ground truth).

        Returns:
            float: A real number representing the metric score.
        """
        pass

    def __call__(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray
                 ) -> float:
        """
        Allows the metric object to be called like a function.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The evaluated metric score.
        """
        return self.evaluate(prediction, ground_truth)


class Accuracy(Metric):
    """
    Accuracy metric for classification.
    """

    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray
                 ) -> float:
        """
        Calculate the accuracy score between predictions and ground truth.

        Args:
            prediction (np.ndarray): Predicted class labels.
            ground_truth (np.ndarray): True class labels.

        Returns:
            float: Accuracy score as a ratio of correct predictions.
        """
        correct = np.sum(prediction == ground_truth)
        return correct / len(ground_truth)


class MeanSquaredError(Metric):
    """
    Mean Squared Error (MSE) metric for regression.
    """

    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray
                 ) -> float:
        """
        Calculate the mean squared error between predictions and ground truth.

        Args:
            prediction (np.ndarray): Predicted values.
            ground_truth (np.ndarray): True values.

        Returns:
            float: Mean squared error value.
        """
        errors = (prediction - ground_truth) ** 2
        return np.mean(errors)


class RootMeanSquaredError(Metric):
    """
    Root Mean Squared Error (RMSE) metric for regression.
    """

    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray
                 ) -> float:
        """
        Calculate the root mean squared error
        between predictions and ground truth.

        Args:
            prediction (np.ndarray): Predicted values.
            ground_truth (np.ndarray): True values.

        Returns:
            float: Root mean squared error value.
        """
        errors = (prediction - ground_truth) ** 2
        return np.sqrt(np.mean(errors))


class LogarithmicLoss(Metric):
    """
    Logarithmic Loss metric for classification with probabilistic outputs.
    """

    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray
                 ) -> float:
        """
        Calculate the logarithmic loss (log loss)
        between predictions and ground truth.

        Args:
            prediction (np.ndarray): Predicted probabilities
            for the positive class.
            ground_truth (np.ndarray): True binary class labels.

        Returns:
            float: Logarithmic loss value.
        """
        epsilon = 1e-15
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        log_loss = np.sum(
            ground_truth * np.log(prediction) +
            (1 - ground_truth) * np.log(1 - prediction)
        )
        return -log_loss / len(ground_truth)


class Precision(Metric):
    """
    Precision metric for binary classification.
    """

    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray
                 ) -> float:
        """
        Calculate the precision score between predictions and ground truth.

        Args:
            prediction (np.ndarray): Predicted binary class labels.
            ground_truth (np.ndarray): True binary class labels.

        Returns:
            float: Precision score as a ratio of
            true positives to predicted positives
            with a cor
        """
        true_positives = np.sum((prediction == 1) & (ground_truth == 1))
        false_positives = np.sum((prediction == 1) & (ground_truth == 0))
        return true_positives / (true_positives + false_positives + 1e-15)


class RSquared(Metric):
    """
    R-squared (Coefficient of Determination) metric for regression.
    """

    def evaluate(self,
                 prediction: np.ndarray,
                 ground_truth: np.ndarray
                 ) -> float:
        """
        Calculate the R-squared score between predictions and ground truth.

        Args:
            prediction (np.ndarray): Predicted values.
            ground_truth (np.ndarray): True values.

        Returns:
            float: R-squared score, with 1 being perfect fit.
        """
        nominator = np.sum((ground_truth - prediction) ** 2)
        denomiter = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        return 1 - (nominator / denomiter)


def get_metric(name: str) -> Metric:
    """
    Factory function to get a metric by name.

    Args:
        name (str): The name of the metric.

    Returns:
        Metric: An instance of the corresponding Metric subclass.

    Raises:
        ValueError: If the specified metric name is not supported.
    """
    if name == "accuracy":
        return Accuracy()
    elif name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "root_mean_squared_error":
        return RootMeanSquaredError()
    elif name == "logarithmic_loss":
        return LogarithmicLoss()
    elif name == "precision":
        return Precision()
    elif name == "r_squared":
        return RSquared()
    else:
        raise ValueError(f"Metric '{name}' is not supported.")
