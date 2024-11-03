import base64


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
        self._id = f"{base64.base64_encode(self.asset_path)}-{self.version}"

    @property
    def id(self) -> str:
        return self._id

    def read(self) -> bytes:
        return self._data

    def save(self, new_data) -> None:
        self._data = new_data
        return self.read
