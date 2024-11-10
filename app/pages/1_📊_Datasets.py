import streamlit as st
from app.datasets.management import create, save


def display_title() -> None:
    """
    Displays the title for the Streamlit page.

    This function sets the Streamlit page title to "Dataset Management.
    """
    st.title("Dataset Management")


def manage_dataset() -> None:
    """
    Creates and saves a dataset.

    This function utilizes the `create` function to generate a dataset
    and then saves it using the `save` function.

    Raises:
        Exception: If either `create` or `save` functions throw an error.
    """
    data = create()
    if data:
        st.write(data.read())
    save(data)
    

display_title()
manage_dataset()
