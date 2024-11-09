from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Identifies and returns features with their types.

    Assumption:
        Only categorical and numerical features are present in the dataset,
        and there are no NaN values.

    Args:
        dataset (Dataset): The dataset containing raw data.

    Returns:
        List[Feature]: A list of features, each with its type specified
        (categorical or numerical).
    """
    feature_list = []
    data = dataset.read()
    print("This is the type", type(data))
    for column in data.columns:
        if data[column].dtype == 'object':
            feature = Feature(name=column, type='categorical')
            feature_list.append(feature)
        else:
            feature = Feature(name=column, type='numerical')
            feature_list.append(feature)
    return feature_list
