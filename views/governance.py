import streamlit as st
import os
from utils.config import DATA_FOLDER
from services.vector_store import (
    rebuild_index,
    get_latest_chunk_count,
    load_vector_store
)

def render_governance():

    st.markdown("""
    <div class="card">
        <div style="font-size:1.8rem;font-weight:800;">
            📊 Governance Center
        </div>
        <div style="color:#64748b;">
            Manage enterprise files, indexing and trust operations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    files = os.listdir(DATA_FOLDER)

    c1,c2 = st.columns(2)

    with c1:
        st.metric("Stored Files", len(files))

    with c2:
        st.metric("Indexed Chunks", get_latest_chunk_count())

    st.markdown("---")

    for file in files:

        path = os.path.join(DATA_FOLDER,file)

        with st.expander(file):

            st.write(
                "Size:",
                round(os.path.getsize(path)/1024,2),
                "KB"
            )

            if st.button("Delete", key=file):

                os.remove(path)

                progress = st.progress(0)
                status = st.empty()

                rebuild_index(progress,status)
                load_vector_store.clear()

                st.success("Deleted successfully.")
                st.rerun()

    st.markdown("---")

    uploaded = st.file_uploader(
        "Select files",
        type=["txt","pdf"],
        accept_multiple_files=True,
        key=f"upload_{st.session_state.uploader_key}"
    )

    if uploaded:

        if st.button("Submit Upload"):

            for file in uploaded:

                path = os.path.join(DATA_FOLDER,file.name)

                with open(path,"wb") as f:
                    f.write(file.getbuffer())

            progress = st.progress(0)
            status = st.empty()

            rebuild_index(progress,status)
            load_vector_store.clear()

            st.session_state.uploader_key += 1

            st.success("Uploaded successfully.")
            st.rerun()

    if st.button("🔄 Rebuild Full Index"):

        progress = st.progress(0)
        status = st.empty()

        rebuild_index(progress,status)
        load_vector_store.clear()

        st.success("Rebuilt successfully.")
        st.rerun()