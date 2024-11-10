import streamlit as st
from app.core.system import AutoMLSystem

def configure_page() -> None:
    """
    Configures the Streamlit page with a title and
    icon specific to the modelling section.
    """
    st.set_page_config(page_title="Deployment", page_icon="🦾")

def write_helper_text(text: str) -> None:
    """
    Displays helper text in a specific color on the Streamlit page.

    Args:
        text (str): The text to display on the page,s
        tyled with a specific color.
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

def main() -> None:
    """
    Main function to run the Streamlit app
    for model pipeline creation.

    This function initializes the page, displays a title and helper text,
    and guides the user through the pipeline creation steps,
    including selecting data, target features, input features, model type,
    metrics, and split configuration.
    """
    autoML = AutoMLSystem.get_instance()
    configure_page()
    st.write("# 🦾 Deployment")
    write_helper_text("In this section, you can manage and use "
                      "your saved machine learning pipelines.")
    saved_pipelines = autoML.registry.list("pipeline")
    if len(saved_pipelines) == 0:
        st.write("No pipelines saved yet.")
    else:
        selected_pipeline = st.selectbox("Select a pipeline", saved_pipelines)
        pipeline = autoML.registry.get(selected_pipeline)
        st.write(pipeline)
        if st.button("Deploy"):
            autoML.deploy(pipeline)
            st.write("Pipeline deployed successfully.")
        if st.button("Delete"):
            autoML.registry.delete(selected_pipeline)
            st.write("Pipeline deleted successfully.")
        if st.button("Download"):
            st.write("Downloaded pipeline.")
            st.write("Downloaded pipeline configuration.")
            st.write("Downloaded pipeline model.")
            st.write("Downloaded pipeline artifacts.")