from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """
    Represents a dataset artifact with functionalities to load data from
    a DataFrame or CSV file and save it back.

    Inherits:
        Artifact: Base class for managing various artifacts in the system.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the Dataset artifact with a default type of 'dataset'.

        Args:
            *args: Unkown amount of non-keyword arguments
            **kwargs: Unkown amount of keyword arguments
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame,
        name: str,
        asset_path: str,
        version: str = "1.0.0"
    ) -> "Dataset":
        """
        Creates a Dataset artifact from a pandas DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to be saved as a dataset.
            name (str): The name of the dataset.
            asset_path (str): The path where the dataset will be stored.
            version (str, optional): The version of the dataset whichs
            Defaults to "1.0.0".

        Returns:
            Dataset: An instance of Dataset with data encoded as CSV.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset as a pandas DataFrame.

        Returns:
            pd.DataFrame: The data stored in the dataset artifact,
            loaded as a DataFrame.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Saves a pandas DataFrame as CSV bytes in the dataset artifact.

        Args:
            data (pd.DataFrame): The DataFrame to be saved.

        Returns:
            bytes: The CSV-encoded data as bytes.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
