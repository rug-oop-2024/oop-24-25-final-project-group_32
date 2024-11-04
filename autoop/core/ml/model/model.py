from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal

class Model(ABC):
    def __init__(self) -> None:
        self._parameters: dict = {}
        self._data: np.ndarray

    @abstractmethod
    def fit(self, observation: np.ndarray, ground_truth: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, observation: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def to_artifact(self) -> Artifact:
        pass