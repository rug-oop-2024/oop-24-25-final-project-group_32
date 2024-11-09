from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Dict, Literal


class Model(ABC):
    """
    Base class for the models.
    """
    def __init__(self) -> None:
        """
        Initialize the Model base class.

        Args:
            type (str): The type of model.
        """
        self._parameters: Dict = {}
        self._type: Literal["regression", "classification"] = None

    @property
    def parameters(self) -> Dict:
        """
        Getter of the model parameters.
        Returns:
            Dict: a deepcopy of the dictionary of the parameters of the model
        """
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, new_parameters: dict[str, Any]) -> None:
        """
        Use the setter to check and update parameters.

        Args:
            new_parameters (dict[str, np.ndarray]):
                The new parameter dictionary.

        Raises:
            ValueError:
                If the new input is not a dictionary.
        """
        if not isinstance(new_parameters, dict):
            raise TypeError("Parameters must be a dictionary.")

        for key in new_parameters.keys():
            self._check_key(key)

        self._parameters.update(new_parameters)

    @property
    def type(self) -> str:
        """
        Getter of the type of the model.

        Returns:
            str: The type of the model
        """
        try:
            return self._type
        except None:
            print("The type cannot be None")

    def _check_key(self, key: str) -> None:
        """
        Checks keys for the parameters.

        Args:
            key (str):
                The parameter name.

        Raises:
            TypeError:
                If the key is not a string.
        """
        if not isinstance(key, str):
            raise TypeError("Keys for the parameters must be strings.")

    @abstractmethod
    def fit(self, observation: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Abstract method to fit the model to provided data.

        Args:
            observation (np.ndarray): Input features for the model training.
            ground_truth (np.ndarray): Ground truth labels corresponding to
            the observations.

        Raises:
            NotImplementedError: To be implemented in subclasses.
        """
        pass

    @abstractmethod
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Abstract method to generate predictions from the model.

        Args:
            observation (np.ndarray): Input features for
            generating predictions.

        Returns:
            np.ndarray: Predicted outputs from the model.

        Raises:
            NotImplementedError: To be implemented in subclasses.
        """
        pass

    def to_artifact(self, name: str, version: str) -> Artifact:
        """Converts the model to an Artifact object for storage.

        Args:
            name (str): The name of the artifact.
            version (str): The version of the artifact.

        Returns:
            Artifact: An artifact instance containing
            model metadata and parameters.
        """
        model = name
        path = f"assets\\objects{name}"
        params = self._parameters
        artifact = Artifact(
            name=model,
            type=self._type,
            version=version,
            asset_path=path,
            parameters=params,
            data=None
        )
        return artifact
