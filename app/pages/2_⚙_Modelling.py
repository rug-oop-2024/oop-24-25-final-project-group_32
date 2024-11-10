import streamlit as st
from app.Modelling.model_pipeline import CreatePipeline


def initialize_pipeline() -> CreatePipeline:
    """
    Initializes the machine learning pipeline creation instance.

    Returns:
        CreatePipeline: An instance of the `CreatePipeline` class
        from `model_pipeline`.
    """
    return CreatePipeline().get_instance()


def configure_page() -> None:
    """
    Configures the Streamlit page with a title and
    icon specific to the modelling section.
    """
    st.set_page_config(page_title="Modelling", page_icon="📈")


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
    configure_page()
    st.write("# ⚙ Modelling")
    write_helper_text("In this section, you can design "
                      "a machine learning pipeline"
                      "to train a model on a dataset.")

    create_pipeline = initialize_pipeline()

    create_pipeline.choose_data()
    if create_pipeline.data:
        create_pipeline.choose_target_feature()
        if create_pipeline.target_feature:
            create_pipeline.choose_model()
            if create_pipeline.model:
                create_pipeline.choose_input_features()
                if create_pipeline.input_features:
                    pipeline = create_pipeline.model
                    if (pipeline != "MultipleLinearRegression") or (len(
                            create_pipeline.input_features) > 1):
                        create_pipeline.choose_metrics()
                        if create_pipeline.metrics:
                            create_pipeline.choose_split()
                            if create_pipeline.split:
                                if st.button("Create Pipeline"):
                                    create_pipeline.summary()


if __name__ == "__main__":
    main()
