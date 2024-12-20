from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy

ClASSIFICATION_METRICS = [
    "accuracy",
    "logarithmic_loss",
    "macro_average"
]

REGRESSION_METRICS = [
    "mean_squared_error",
    "root_mean_squared_error",
    "r_squared"
]


def get_metrics(type: str) -> list:
    """
    Getter for the metrics.

    Args:
        type (str): A string that decides which
        metrics are available for the inputted feature

    Returns:
        list: The list of available metrics.
    """

    if type == "classification":
        return ClASSIFICATION_METRICS
    elif type == "regression":
        return REGRESSION_METRICS


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

    def _check_arrays(
        self, predictions: np.ndarray, ground_truth: np.ndarray
    ) -> None:
        """
        Checks if the predictions and ground_truth have the same array length,
        dimensions And if they are empty or not.

        Args:
            predictions (np.ndarray): an array of predictions
            ground_truth (np.ndarray): an array with the ground_truth
                (Must match number of predictions.)

        Raises:
            ValueError: If the number of predictions and
            ground_truth are not the same.
            ValueError: If either predictions or ground_truths is empty.
        """

        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"The number of predictions ({len(predictions)}) must equal ",
                f"the number of ground truth labels ({len(ground_truth)}).",
            )
        if (len(predictions) == 0) or (len(ground_truth) == 0):
            raise ValueError(
                "Predictions and ground truth arrays cannot be empty."
            )

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
    name = "accuracy"

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
        self._check_arrays(predictions=prediction, ground_truth=ground_truth)
        if prediction.ndim > 1:
            prediction = np.argmax(prediction, axis=1)
        if ground_truth.ndim > 1:
            ground_truth = np.argmax(ground_truth, axis=1)
        correct = np.sum(prediction == ground_truth)
        return float(correct / len(ground_truth))


class LogarithmicLoss(Metric):
    """
    Logarithmic Loss metric for classification with probabilistic outputs.
    """
    name = "logarithmic_loss"

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
        self._check_arrays(predictions=prediction, ground_truth=ground_truth)

        prediction = np.clip(prediction, 0, 1)
        if prediction.ndim > 1:
            prediction = np.argmax(prediction, axis=1)
        if ground_truth.ndim > 1:
            ground_truth = np.argmax(ground_truth, axis=1)

        log_loss = np.sum(
            ground_truth * np.log(
                prediction) + (1 - ground_truth) * np.log(1 - prediction)
        )

        try:
            return float(-log_loss / len(ground_truth))
        except ZeroDivisionError:
            print("You cannot divide by zero")


class MacroAveragePrecision(Metric):
    """
    A class to compute the macro-average precision for multi-class
    classification tasks.
    Macro-average precision calculates the precision for each class
    individually and
    then averages these scores, treating all classes equally.
    """
    name = "macro_average"
    def evaluate(self, prediction: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Computes the macro-average precision score
        Args:
            y_true (np.ndarray): The ground truth values
            y_pred (np.ndarray): The predicted values
        Returns:
            float: The computed metric score
        """
        unique_labels = np.unique(deepcopy(ground_truth))
        if prediction.ndim > 1:
            prediction = np.argmax(prediction, axis=1)
        if ground_truth.ndim > 1:
            ground_truth = np.argmax(ground_truth, axis=1)

        precision_per_class = []
        for label in unique_labels:
            correct = np.sum((ground_truth == label) & (prediction == label))
            incorrect = np.sum((ground_truth != label) & (prediction == label))
            try:
                precision = correct / (correct + incorrect)
            except ZeroDivisionError:
                precision = 0
            precision_per_class.append(precision)
        return float(np.mean(precision_per_class))


class MeanSquaredError(Metric):
    """
    Mean Squared Error (MSE) metric for regression.
    """
    name = "mean_squared_error"

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
        self._check_arrays(predictions=prediction, ground_truth=ground_truth)
        errors = np.mean((prediction - ground_truth) ** 2)
        return float(errors)


class RootMeanSquaredError(Metric):
    """
    Root Mean Squared Error (RMSE) metric for regression.
    """
    name = "root_mean_squared_error"

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
        self._check_arrays(predictions=prediction, ground_truth=ground_truth)
        errors = np.mean((prediction - ground_truth) ** 2)
        return float(np.sqrt(errors))


class RSquared(Metric):
    """
    R-squared (Coefficient of Determination) metric for regression.
    """
    name = "r_squared"

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
        self._check_arrays(predictions=prediction, ground_truth=ground_truth)
        nominator = np.sum((np.mean(ground_truth) - prediction) ** 2)
        denomiter = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        if denomiter == 0:
            return print("cannot divide by zero")
        else:
            return float(1 - (nominator / denomiter))


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
    elif name == "r_squared":
        return RSquared()
    elif name == "macro_average":
        return MacroAveragePrecision()
    else:
        raise ValueError(f"Metric '{name}' is not supported.")
