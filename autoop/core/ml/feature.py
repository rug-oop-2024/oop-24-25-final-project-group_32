
from typing import Literal
import numpy as np
from copy import deepcopy
import pandas as pd

from autoop.core.ml.dataset import Dataset


class Feature():
    # attributes here
    def __init__(self,
                 dataset: pd.DataFrame,
                 name: str,
                 type: Literal["categorical", "numerical"]):
        self._name = name
        self._dataset = dataset.to_csv(index=False).encode()
        self._type = type

    @property
    def name(self):
        return deepcopy(self._name)

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def dataset(self):
        return deepcopy(self._dataset)

    @property
    def type(self):
        return deepcopy(self._type)

    @type.setter
    def type(self, value: Literal["categorical", "numerical"]):
        self._type = value

    def __str__(self):
        raise NotImplementedError("To be implemented.")
