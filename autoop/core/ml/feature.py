from typing import Literal
from copy import deepcopy


class Feature:
    """
    Represents a feature (or column) in a dataset, which can be of either
    categorical or numerical type.
    """

    def __init__(self,
                 name: str,
                 type: Literal["categorical", "numerical"]
                 ) -> None:
        """
        Initializes a Feature instance with a name and type.

        Args:
            name (str): The name of the feature.
            type (Literal["categorical", "numerical"]): The data type
            of the feature, either "categorical" or "numerical".
        """
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        """
        Gets the name of the feature.

        Returns:
            str: the name of the feature
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def type(self) -> Literal["categorical", "numerical"]:
        """
        Gets the type of the feature.
        Returns:
            Literal["categorical", "numerical"]: a deepcopy
            of the type
        """
        return deepcopy(self._type)

    @type.setter
    def type(self, value: Literal["categorical", "numerical"]) -> None:
        """
        Setter of the type of the feature.
        """
        self._type = value

    def to_artifact(self) -> dict:
        """
        Converts the feature to an artifact dictionary.

        Returns:
            dict: The feature as an artifact dictionary.
        """
        return {
            "name": self._name,
            "type": self._type
        }

    def __str__(self) -> str:
        """
        Provides a string representation of the feature.

        Returns:
            str: A description of the feature's name and type.
        """
        return f"{self.name}: {self.type}"
