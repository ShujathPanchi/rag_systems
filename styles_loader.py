import streamlit as st

def load_css():
    with open("styles/main.css","r",encoding="utf-8") as f:
        css = f.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True
    )