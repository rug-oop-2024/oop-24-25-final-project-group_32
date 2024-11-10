from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """
    Class that raises an error when a path is not found.
    """

    def __init__(self, path: str) -> None:
        """
        Initializes the NotFoundError with a message
        indicating the missing path.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Abstract base class for storage systems.
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path.

        Args:
            data (bytes): Data to save.
            path (str): Path to save the data.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path.

        Args:
            path (str): Path to load the data.

        Returns:
            bytes: Loaded data.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path.

        Args:
            path (str): Path to delete the data.
        """
        pass

    @abstractmethod
    def list(self, path: str) -> List[str]:
        """
        List all paths under a given path.

        Args:
            path (str): Path to list.

        Returns:
            List[str]: List of paths.
        """
        pass


class LocalStorage(Storage):
    """
    Class for Local storage implementation using the file system.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """Initializes the LocalStorage instance with a base path.

        Args:
            base_path (str): The base directory
            for storing data (default is "./assets").
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a file at the specified path.

        Args:
            data (bytes): Data to be saved.
            key (str): The path (key) to save the data.
        """
        path = self._join_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a file at the specified path.

        Args:
            key (str): The path (key) to load the data from.

        Returns:
            bytes: The loaded data.

        Raises:
            NotFoundError: If the path does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete the file at the specified path.

        Args:
            key (str): The path (key) of the file to delete.

        Raises:
            NotFoundError: If the path does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        List all files under a given path (prefix).

        Args:
            prefix (str): The directory to list the files from.

        Returns:
            List[str]: List of relative file paths.

        Raises:
            NotFoundError: If the directory does not exist.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path)
                for p in keys if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        """
        Check if a given path exists.

        Args:
            path (str): The path to check.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Join the base path with a given relative path.

        Args:
            path (str): The relative path to join with the base path.

        Returns:
            str: The joined absolute path.
        """

        return os.path.normpath(os.path.join(self._base_path, path))
