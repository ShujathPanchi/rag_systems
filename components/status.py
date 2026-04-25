import streamlit as st


def render_status():

    st.markdown(
        """
        <div class="card">

            <b>System Status</b>

            <br><br>

            <span style="color:#22c55e;">●</span>
            Ollama Running

            <br>

            <span style="color:#22c55e;">●</span>
            Vector DB Ready

            <br>

            <span style="color:#22c55e;">●</span>
            Model Loaded

        </div>
        """,
        unsafe_allow_html=True
    )