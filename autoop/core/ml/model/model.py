from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal

class Model(ABC):
    def __init__(self) -> None:
        self._parameters: dict = {}
        self._data: np.ndarray
        self._type

    @property
    def parameters(self) -> dict:
        return deepcopy(self._parameters)
    
    @property
    def type(self) -> str:
        return self._type

    @abstractmethod
    def fit(self, observation: np.ndarray, ground_truth: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, observation: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def to_artifact(self, name: str) -> Artifact:
        artifact = Artifact(name,
                            "asset_path",
                            "1.0.0",
                            self._data.encode(),
                            self._type,
                            self._parameters,
                            ["regression"] if self._type in REGRESSION_MODELS else ["classification"]
                            # I just didn't quite finish that but we can fix it later
                            )
        return artifact
