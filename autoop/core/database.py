import json
from typing import Tuple, List, Union
import os
from autoop.core.storage import Storage


class Database:
    """
    Database class for managing collections of data using a storage backend.
    """

    def __init__(self, storage: Storage) -> None:
        """
        Initializes the Database with a storage backend and
        loads existing data.

        Args:
            storage (Storage): The storage backend to use for data persistence.
        """
        self._storage = storage
        self._data = {}
        self._load()

    @property
    def storage(self) -> Storage:
        """
        Gets the storage backend used by the Database.

        Returns:
            Storage: The storage backend instance.
        """
        return self._storage

    def set(self, collection: str, id: str, entry: dict) -> dict:
        """
        Sets a key-value pair in the database.

        Args:
            collection (str): The collection in which to store the data.
            id (str): The identifier for the data entry.
            entry (dict): The data to store.

        Returns:
            dict: The data that was stored.
        """
        assert isinstance(entry, dict), "Data must be a dictionary"
        assert isinstance(collection, str), "Collection must be a string"
        assert isinstance(id, str), "ID must be a string"

        if not self._data.get(collection):
            self._data[collection] = {}
        self._data[collection][id] = entry
        self._persist()
        return entry

    def get(self, collection: str, id: str) -> Union[dict, None]:
        """
        Gets a value from the database.

        Args:
            collection (str): The collection to retrieve data from.
            id (str): The identifier of the data entry.

        Returns:
            Union[dict, None]: The retrieved data, or None if not found.
        """
        if not self._data.get(collection):
            return None
        return self._data[collection].get(id)

    def delete(self, collection: str, id: str) -> None:
        """
        Deletes a key-value pair from the database.

        Args:
            collection (str): The collection to delete data from.
            id (str): The identifier of the data entry to delete.
        """
        if not self._data.get(collection):
            return
        if self._data[collection].get(id):
            del self._data[collection][id]
        self._persist()

    def list(self, collection: str) -> List[Tuple[str, dict]]:
        """
        Lists all data entries in a collection.

        Args:
            collection (str): The collection to list data from.

        Returns:
            List[Tuple[str, dict]]: A list of tuples with
            each item's id and data.
        """
        if not self._data.get(collection):
            return []
        return [(id, data) for id, data in self._data[collection].items()]

    def refresh(self) -> None:
        """
        Refreshes the database by reloading data from the storage backend.
        """
        self._load()

    def _persist(self) -> None:
        """
        Persists the current state of the database to the storage backend.
        """
        for collection, data in self._data.items():
            if not data:
                continue
            for id, item in data.items():
                self._storage.save(json.dumps(item).encode(),
                                   f"{collection}{os.sep}{id}")
        keys = self._storage.list("")
        for key in keys:
            collection, id = key.split(os.sep)[-2:]
            if not self._data.get(collection, {}).get(id):
                self._storage.delete(f"{collection}{os.sep}{id}")

    def _load(self) -> None:
        """
        Loads data from the storage backend into the in-memory database.
        """
        self._data = {}
        for key in self._storage.list(""):
            collection, id = key.split(os.sep)[-2:]
            data = self._storage.load(f"{collection}{os.sep}{id}")
            if collection not in self._data:
                self._data[collection] = {}
            self._data[collection][id] = json.loads(data.decode())
