from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    feature_list = []
    data = dataset.read()
    for column in data.columns:
        if data[column].dtype == 'object':
            feature = Feature(data[column], column, 'categorical')
            feature_list.append(feature)
        else:
            feature = Feature(data[column], column, 'numerical')
            feature_list.append(feature)
    return feature_list
