import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

current_dataset = Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")


st.title("Dataset Management")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    refractored_data = Dataset.from_dataframe(df, "test", "", "1.0.0")
    automl.registry.register(refractored_data)
    st.write("Dataset saved to artifact registry")

st.write("Check")
