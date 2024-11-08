from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy


class Model(ABC):
    def __init__(self, type) -> None:
        self._parameters: dict = {}
        self._type = type

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

    def to_artifact(self, name: str, version: str) -> Artifact:
        model = name
        path = "assets\\objects" + name
        params = self._parameters
        artifact = Artifact(name=model,
                            type=self._type,
                            version=version,
                            asset_path=path,
                            parameters=params,
                            data=None
                            )
        return artifact
