from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from autoop.functional.feature import detect_feature_types
import streamlit as st
from typing import List, Optional
import os


class ArtifactRegistry:
    """
    Handles the registration, retrieval, listing,
    and deletion of artifacts in the system.
    """

    def __init__(self, database: Database, storage: Storage):
        """
        Initializes the ArtifactRegistry with a database and storage.

        Args:
            database (Database): The database instance for
            storing artifact metadata.
            storage (Storage): The storage instance for
            saving artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Registers an artifact by saving it to storage and
        recording metadata in the database.

        Args:
            artifact (Artifact): The artifact to be registered.
        """
        st.write(item.name for item in detect_feature_types(artifact))

        self._storage.save(artifact.data, artifact.asset_path)

        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: Optional[str] = None) -> List[Artifact]:
        """
        Lists all artifacts, optionally filtered by type.

        Args:
            type (Optional[str]): The type of artifacts to list.
            Defaults to None, which lists all types.

        Returns:
            List[Artifact]: A list of artifacts matching the specified type.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieves an artifact by its ID.

        Args:
            artifact_id (str): The unique identifier of the artifact.

        Returns:
            Artifact: The artifact corresponding to the provided ID.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Deletes an artifact by its ID, removing its
        data from storage and metadata from the database.

        Args:
            artifact_id (str): The unique identifier of the artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    Represents the core system for managing artifacts within an automated
    machine learning environment.
    """

    _instance: Optional['AutoMLSystem'] = None

    def __init__(self, storage: LocalStorage, database: Database):
        """
        Initializes the AutoMLSystem with storage and database instances.

        Args:
            storage (LocalStorage): The local storage instance for
            saving artifact data.
            database (Database): The database instance for
            storing artifact metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> 'AutoMLSystem':
        """
        Retrieves the singleton instance of the AutoMLSystem,
        creating it if necessary.

        Returns:
            AutoMLSystem: The singleton instance of the AutoMLSystem.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage(os.path.join(".", "assets", "objects")),
                Database(LocalStorage(os.path.join(".", "assets", "dbo")))
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Provides access to the ArtifactRegistry for managing artifacts.

        Returns:
            ArtifactRegistry: The registry instance for managing artifacts.
        """
        return self._registry
