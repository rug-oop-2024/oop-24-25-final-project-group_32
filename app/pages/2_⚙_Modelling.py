import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from app.Modelling.model_pipeline import CreatePipeline

print("it gets here")
create_pipeline = CreatePipeline().get_instance()

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design \
                  a machine learning pipeline to train a model on a dataset.")
create_pipeline.choose_data()
if create_pipeline.data is not None:
    create_pipeline.choose_target_feature()
    create_pipeline.choose_input_features()
    create_pipeline.choose_model()
    create_pipeline.choose_metrics()
    create_pipeline.choose_split()
    if st.button("Create Pipeline"):
        create_pipeline.summary()
