import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from app.datasets.management import create, save

st.title("Dataset Management")
data = create()
save(data)