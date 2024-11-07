import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
 
def create() -> Dataset:
    '''Prompts the user to upload a file using streamlit and returns a dataset from that file.

    Args:
        None

    Returns: 
        Dataset: Dataset created from the uploaded csv file
    '''
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        refractored_data = Dataset.from_dataframe(df, uploaded_file.name, uploaded_file.name, "1.0.0")
        return refractored_data

def save(artifact: Dataset|None) -> None:
    '''Saves file that is uploaded as an artifact to the artifact registry.
    
    Args:
        Artifact (Dataset): Dataset to be saved
        
    Returns:
        None
    '''
    automl = AutoMLSystem.get_instance()
    if st.button("save"):
        if artifact == None:
            st.write("Need a file to be saved")
            return
        automl.registry.register(artifact)
        st.write("Dataset saved to artifact registry")
