from typing import List, Tuple
from autoop.core.ml.feature import Feature
from autoop.core.ml.dataset import Dataset
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_features(
    features: List[Feature],
    dataset: Dataset
) -> List[Tuple[str, np.ndarray, dict]]:
    """
    Preprocesses features in the dataset.

    Args:
        features (List[Feature]): A list of features to preprocess.
        Each feature specifies its name and type (categorical or numerical).
        dataset (Dataset): The dataset object containing
        the raw data to be processed.

    Returns:
        List[Tuple[str, np.ndarray, dict]]: A list of tuples, each containing:
            - str: The name of the feature.
            - np.ndarray: The preprocessed feature data, with shape (N, ...).
            - dict: Metadata about the preprocessing
            method used for each feature.
    """
    results = []
    raw = dataset.read()
    for feature in features:
        if feature.type == "categorical":
            encoder = OneHotEncoder()
            data = encoder.fit_transform(
                raw[feature.name].values.reshape(-1, 1)).toarray()
            artifact = {"type": "OneHotEncoder",
                        "encoder": encoder.get_params()}
            results.append((feature.name, data, artifact))
        if feature.type == "numerical":
            scaler = StandardScaler()
            data = scaler.fit_transform(
                raw[feature.name].values.reshape(-1, 1))
            artifact = {"type": "StandardScaler",
                        "scaler": scaler.get_params()}
            results.append((feature.name, data, artifact))

    results = list(sorted(results, key=lambda x: x[0]))
    return results
