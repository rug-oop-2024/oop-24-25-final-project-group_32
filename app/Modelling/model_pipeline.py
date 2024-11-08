import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline

class CreatePipeline:
    def __init__(self) -> None:
        self._automl = AutoMLSystem.get_instance()
        self._data: Dataset
        self._features: list[Feature]
        self._target_feature: Feature
        self._input_features: list[Feature]

    def choose_data(self) -> None:
        datasets = self._automl.registry.list(type="dataset")
        selected = st.selectbox("select your dataset", datasets, format_func=lambda data:data.name)
        if st.button("choose"):
            self._features = detect_feature_types(self._data)
            self.choose_features
            
    def choose_features(self) -> None:
        self._target_feature = st.radio("Select a target feature", self._features)
        # This is not working yet cos self._data is a artifact and we need it to be a dataset or find another way around this
        # We also need to add more functions to this class for all the selection stuff and just put the functions in modelling
        # And then we can put a make pipeline function once all the selection is done.
