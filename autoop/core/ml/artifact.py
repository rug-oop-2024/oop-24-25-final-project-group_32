import base64

class Artifact():
    def __init__(self, path: str, data: bytes, version: str, type: str):
        self._path = path
        self._version = version
        self._data = data
        self._type = type