import streamlit as st
import os

DATA_FOLDER = "data"
VECTOR_FOLDER = "vector_store"

def setup_app():
    st.set_page_config(
        page_title="Cognivault AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(VECTOR_FOLDER, exist_ok=True)