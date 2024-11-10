import streamlit as st
from app.core.system import AutoMLSystem
from app.Modelling.model_pipeline import CreatePipeline


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


def initialize_pipeline() -> CreatePipeline:
    """
    Initializes the machine learning pipeline creation instance.

    Returns:
        CreatePipeline: An instance of the `CreatePipeline` class
        from `model_pipeline`.
    """
    return CreatePipeline().get_instance()


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
    load_pipeline = initialize_pipeline()
    configure_page()
    st.write("# 🦾 Deployment")
    write_helper_text("In this section, you can manage and use "
                      "your saved machine learning pipelines.")
    saved_pipelines = autoML.registry.list("pipeline")
    if len(saved_pipelines) == 0:
        st.write("No pipelines saved yet.")
    else:
        selected_pipeline = st.selectbox("Select a pipeline",
                                         saved_pipelines,
                                         format_func=lambda x: x.name)
        if st.button("load"):
            load_pipeline.load(selected_pipeline)
        if st.button("delete"):
            autoML.registry.delete(selected_pipeline.id)
            st.write("Pipeline deleted.")

if __name__ == "__main__":
    main()
