from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal

class Model(ABC):
    def __init__(self, name: str) -> None:
        self._parameters: dict = {}

    @abstractmethod
    def fit(self, observation: np.ndarray, ground_truth: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, observation: np.ndarray) -> np.ndarray:
        pass

    def to_artifact(self) -> Artifact:
        return Artifact(
            name=self.__class__.__name__,
            asset_path="",
            version="1.0.0",
            data=deepcopy(self._parameters.encode()),
            type="Model",
        )