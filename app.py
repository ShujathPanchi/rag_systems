import streamlit as st

from utils.config import setup_app
from styles_loader import load_css

from components.sidebar import render_sidebar

from views.chat import render_chat
from views.governance import render_governance

from services.session_store import load_chat


setup_app()
load_css()

# --------------------------
# SESSION INIT
# --------------------------
if "page" not in st.session_state:
    st.session_state.page = "Chat"

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "messages" not in st.session_state:
    st.session_state.messages = load_chat()

# --------------------------
# SIDEBAR
# --------------------------
render_sidebar()

# --------------------------
# ROUTING
# --------------------------
if st.session_state.page == "Chat":
    render_chat()

elif st.session_state.page == "Governance":
    render_governance()