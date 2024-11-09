import streamlit as st
import pandas as pd
import io
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import get_metrics, get_metric
from autoop.core.ml.model import get_model_types, get_model
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline
from typing import Optional, List


class CreatePipeline:
    """
    Manages the creation of a machine learning pipeline by
    guiding the user through data and model selections.

    """

    _instance: Optional["CreatePipeline"] = None

    def __init__(self) -> None:
        """
        Initializes the pipeline creation instance,
        retrieving an AutoML system instance and setting default values for
        data, features, model, metrics, and split ratio.
        """
        self._automl = AutoMLSystem.get_instance()
        self._data: Optional[Dataset] = None
        self._features: Optional[List[Feature]] = None
        self._target_feature: Optional[Feature] = None
        self._input_features: Optional[List[Feature]] = None
        self._model: Optional[str] = None
        self._metric: Optional[List[str]] = None
        self._split: Optional[float] = None

    @staticmethod
    def get_instance() -> "CreatePipeline":
        """
        Retrieves the singleton instance of CreatePipeline,
        creating it if it doesn't exist.

        Returns:
            CreatePipeline: Singleton instance of CreatePipeline.
        """
        if CreatePipeline._instance is None:
            CreatePipeline._instance = CreatePipeline()
        return CreatePipeline._instance

    @property
    def data(self) -> Optional[Dataset]:
        """
        The selected dataset.

        Returns:
            Optional[Dataset]: The dataset selected for the pipeline.
        """
        return self._data

    @property
    def target_feature(self) -> Optional[Feature]:
        """
        The target feature for model training.

        Returns:
            Optional[Feature]: The feature chosen as the target.
        """
        return self._target_feature

    @property
    def input_features(self) -> Optional[List[Feature]]:
        """
        The list of input features for model training.

        Returns:
            Optional[List[Feature]]: The features selected as inputs.
        """
        return self._input_features

    @property
    def model(self) -> Optional[str]:
        return self._model

    def choose_data(self) -> None:
        """
        Prompts the user to select a dataset from
        available datasets in the registry.
        Displays the chosen dataset or allows for
        data change if already selected.
        """
        if self._data is None:
            datasets = self._automl.registry.list(type="dataset")
            selected = st.selectbox("Select your dataset", datasets,
                                    format_func=lambda data: data.name)
            if st.button("Choose"):
                self._data = Dataset.from_dataframe(pd.read_csv(io.StringIO(
                    selected.data.decode())),
                    selected.name,
                    selected.asset_path,
                    selected.version)
                self._features = detect_feature_types(self._data)
        else:
            st.write(f"Chosen dataset: {self._data.name}")
            if st.button("Change data"):
                self._data = None
                self._features = None
                self._target_feature = None
                self._input_features = None
                self._metric = None
                self._split = None

    def choose_target_feature(self) -> None:
        """
        Prompts the user to select a target feature
        from the list of detected features.
        """
        self._target_feature = st.selectbox("Select a target feature",
                                            self._features)

    def choose_input_features(self) -> None:
        """
        Prompts the user to select input features for
        model training from the list of detected features.
        """
        self._input_features = st.multiselect("Select input features",
                                              self._features)

    def choose_model(self) -> None:
        """
        Prompts the user to select a model based on
        the type of the target feature.
        """
        if self._target_feature.type == "categorical":
            model_types = get_model_types("classification")
        else:
            model_types = get_model_types("regression")
        self._model = st.selectbox("Select a model to train your data on",
                                   model_types)

    def choose_metrics(self) -> None:
        """
        Prompts the user to select evaluation metrics for the chosen model.
        """
        if self._target_feature.type == "categorical":
            self._metric = st.multiselect("Select metrics",
                                          get_metrics("classification"))
        else:
            self._metric = st.multiselect("Select metrics",
                                          get_metric("regression"))

    def choose_split(self) -> None:
        """
        Prompts the user to select the train-test
        split ratio for model evaluation.
        """
        self._split = st.slider("Select the split ratio", 0.1, 0.9, 0.8)

    @st.dialog("Summary", width="large")
    def summary(self) -> None:
        """
        Displays a summary of the chosen dataset, features, model, metrics,
        and split ratio.
        Initiates pipeline creation if confirmed by the user.
        """
        st.write(f"**Dataset**: {self._data.name}")
        st.write(f"**Target feature**: {self._target_feature.name}")
        input_features_str = ', '.join(
            [feature.name for feature in self._input_features])
        st.write(f"**Input features**: {input_features_str}")
        st.write(f"**Model**: {self._model}")
        st.write(f"**Metrics**: {', '.join(self._metric)}")
        st.write(f"**Split ratio**: {self._split}")
        if st.button("Create"):
            self.create_pipeline()

    def create_pipeline(self) -> None:
        """
        Creates a pipeline with the selected data, features, model, metrics,
        and split ratio.
        Executes the pipeline and displays the results.
        """
        pipeline = Pipeline(
            metrics=[get_metric(metric) for metric in self._metric],
            dataset=self._data,
            model=get_model(self._model),
            input_features=self._input_features,
            target_feature=self._target_feature,
            split=self._split
        )
        st.write("Pipeline created and saved to artifact registry")
        results = pipeline.execute()
        for result in results:
            st.write(results[result])
