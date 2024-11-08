from typing import List, Dict, Any
import pickle
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline:
    """
    A pipeline for executing a machine learning workflow, including
    data preprocessing, model training, and evaluation.

    Attributes:
        _dataset (Dataset): The dataset to be used for training and evaluation.
        _model (Model): The machine learning model to be trained.
        _input_features (List[Feature]): List of input features for the model.
        _target_feature (Feature): The target feature for prediction.
        _metrics (List[Metric]): List of metrics for evaluating the model.
        _artifacts (Dict[str, Any]): Dictionary of artifacts generated during
        preprocessing.
        _split (float): Proportion of data to use for training.
    """

    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8) -> None:
        """
        Initializes the pipeline with dataset, model,
        and other configuration details.

        Args:
            metrics (List[Metric]): Metrics to evaluate the model.
            dataset (Dataset): The dataset used for training and testing.
            model (Model): The model to train and evaluate.
            input_features (List[Feature]): Features used as model inputs.
            target_feature (Feature): The feature to predict.
            split (float, optional): Proportion of data to use for training
            with Defaults to 0.8.

        Raises:
            ValueError: If model type doesn't match target feature type
            (classification vs regression).
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if (target_feature.type == "categorical"
                and model.type != "classification"):
            raise ValueError(
                "Model type must be classification for categorical feature")
        if (target_feature.type == "continuous"
                and model.type != "regression"):
            raise ValueError(
                "Model type must be regression for continuous feature")

    def __str__(self) -> str:
        return f"""
        Pipeline(
            model={self._model.type},
            input_features={list(map(str, self._input_features))},
            target_feature={str(self._target_feature)},
            split={self._split},
            metrics={list(map(str, self._metrics))},
        )
        """

    @property
    def model(self) -> Model:
        """
        Returns the model used in the pipeline.

        Returns:
            Model: The machine learning model instance.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Retrieves artifacts generated during preprocessing and model saving.

        Returns:
            List[Artifact]: List of serialized artifacts.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = pickle.dumps(artifact["encoder"])
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = pickle.dumps(artifact["scaler"])
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(
            name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact: Any) -> None:
        """
        Registers an artifact in the pipeline's artifacts dictionary.

        Args:
            name (str): The name of the artifact.
            artifact (Any): The artifact data.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Preprocesses input and target features and registers artifacts."""
        target_feature_name, target_data, artifact = preprocess_features(
            [self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        self._output_vector = target_data
        self._input_vectors = [data for _, data, _ in input_results]

    def _split_data(self) -> None:
        """
        Splits the data into training and
        testing sets based on the defined split ratio.
        """
        split = self._split
        self._train_X = [vector[:int(split * len(vector))]
                         for vector in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):]
                        for vector in self._input_vectors]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Compacts multiple arrays into a
        single array by concatenating along the second axis.

        Args:
            vectors (List[np.array]): List of numpy arrays to concatenate.

        Returns:
            np.array: Concatenated numpy array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Trains the model using the training dataset.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        Evaluates the model using the testing dataset and computes metrics.
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def _evaluate_training(self) -> None:
        """
        Evaluates the model using the training dataset and computes metrics.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._metrics_training_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_training_results.append((metric, result))
        self._predictions_training = predictions

    def execute(self) -> Dict[str, Any]:
        """
        Executes the pipeline, including preprocessing,
        training, and evaluation.

        Returns:
            Dict[str, Any]: Results including metrics and predictions for
            both training and testing datasets.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        self._evaluate_training()
        return {
            "metrics training set prediction": self._metrics_training_results,
            "predictions training set": self._predictions_training,
            "metrics testing set prediction": self._metrics_results,
            "predictions testing set": self._predictions,
        }
