import base64
import pandas as pd

class Artifact:
    def __init__(
            self,
            name: str,
            asset_path: str,
            version: str,
            data: bytes,
            type: str,
            metadata: dict = None,
            tags: list = None,
            ) -> None:
        """
        Initialize the artifact class.

        Experiment_id and run_id together make the metadata
        dictionary.

        type is called artifact_type to avoid poor naming convention

        Args:
            name (str): The name of the artifact
            asset_path (str): The asset path of the artifact
            version (str): The version of the artifact
            data (byte): The data of the artifact
            artifact_type (str): The type of the artifact
            meta_data (dict): The meta_data of the artifact
            artifact_type (str): The type of the , default = None
            tags (list): The tags of the artifact, default = None
        """

        self._name = name
        self._asset_path = asset_path
        self._version = version
        self._data = data
        self._metadata = metadata if metadata is not None else {}
        self._type = type
        self._tags = tags if tags is not None else []

    @property
    def id(self):
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def read(self) -> bytes:
        return self._data

    def save(self, new_data) -> None:
        self._data = new_data
        return self.read

    def get(self):
        # Look into pipeline what this functions does
        # Looks like it should get the type
        # Can be that it should get anything when specified
        pass
