import streamlit as st


def set_page_configuration() -> None:
    """
    Configures the Streamlit page settings with a title and icon.

    This function sets up the Streamlit page with a title of "Instructions"
    and a waving hand icon.
    """
    st.set_page_config(
        page_title="Instructions",
        page_icon="👋",
    )


def display_instructions(file_path: str) -> None:
    """
    Reads the content of a Markdown file and displays it on the Streamlit page.

    Args:
        file_path (str): The path to the Markdown file
        containing the instructions.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If there is an error reading the file.
    """
    st.markdown(open(file_path).read())


set_page_configuration()
display_instructions("INSTRUCTIONS.md")
