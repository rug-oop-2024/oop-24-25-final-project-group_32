import streamlit as st
import pandas as pd
import io

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric, get_metrics, get_metric
# import autoop.core.ml.model as model
from autoop.core.ml.model import get_model_types, get_model, Model
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline
from copy import deepcopy


class CreatePipeline:
    _instance = None

    def __init__(self) -> None:
        self._automl = AutoMLSystem.get_instance()
        self._data: Dataset = None
        self._features: list[Feature] = None
        self._target_feature: Feature = None
        self._input_features: list[Feature] = None
        self._model: str = None
        self._metric: list[str] = None
        self._split: float = None

    @staticmethod
    def get_instance() -> "CreatePipeline":
        if CreatePipeline._instance is None:
            CreatePipeline._instance = CreatePipeline()
        return CreatePipeline._instance
    
    @property
    def data(self) -> Dataset:
        return self._data
    
    @property
    def target_feature(self) -> Feature:
        return self._target_feature
    
    @property
    def input_features(self) -> list[Feature]:
        return self._input_features

    def choose_data(self) -> None:
        if self._data is None:
            datasets = self._automl.registry.list(type="dataset")
            selected = st.selectbox("select your dataset", datasets,
                                    format_func=lambda data: data.name)
            if st.button("choose"):
                self._data = Dataset.from_dataframe(pd.read_csv(io.StringIO(selected.data.decode())),
                    selected.name,
                    selected.asset_path, 
                    selected.version)
                self._features = detect_feature_types(self._data)
        else:
            st.write(f"Chosen dataset: {self._data.name}")
            if st.button("change data"):
                self._data = None
                self._features = None
                self._target_feature = None
                self._input_features = None
                self._metric = None
                self._split = None

    def choose_target_feature(self) -> None:
        self._target_feature = st.selectbox("Select a target feature",
                            self._features)

    def choose_input_features(self) -> None:
        features = deepcopy(self._features)
        for ind, feature in enumerate(features):
            if feature.name == self._target_feature.name:
                features.pop(ind)
        self._input_features = st.multiselect("Select input features",
                                        features)

    def choose_model(self) -> None:
        if self._target_feature.type == "categorical":
            model_types = get_model_types("classification")
        else:
            model_types = get_model_types("regression")
        self._model = st.selectbox("Select a model to train your data on", model_types)

    def choose_metrics(self) -> None:
        if self._target_feature.type == "categorical":
            self._metric = st.multiselect("Select metrics", get_metrics("classification"))
        else:
            self._metric = st.multiselect("Select metrics", get_metrics("regression"))

    def choose_split(self) -> None:
        self._split = st.slider("Select the split ratio", 0.1, 0.9, 0.8)



    @st.dialog("Summary", width = "large")
    def summary(self) -> None:
        st.write(f"**Dataset**: {self._data.name}")
        st.write(f"**Target feature**: {self._target_feature.name}")
        st.write(f"**Input features**: {', '.join([feature.name for feature in self._input_features])}")
        st.write(f"**Model**: {self._model}")
        st.write(f"**Metrics**: {', '.join([metric for metric in self._metric])}")
        st.write(f"**Split ratio**: {self._split}")
        if st.button("Create"):
            self.create_pipeline()

    def create_pipeline(self) -> None:
        pipeline = Pipeline(metrics=[get_metric(metric) for metric in self._metric],
                            dataset=self._data,
                            model=get_model(self._model),
                            input_features=self._input_features,
                            target_feature=self._target_feature,
                            split=self._split)
        st.write("Pipeline created and saved to artifact registry")
        results = pipeline.execute()
        for result in results:
            st.write(results[result])