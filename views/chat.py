import streamlit as st
import os

from components.hero import render_hero

from services.model_service import load_model
from services.vector_store import (
    load_vector_store,
    get_latest_chunk_count
)
from services.rag_service import ask_question
from services.session_store import (
    save_chat,
    clear_chat
)


def render_chat():

    # ---------------------------------
    # MODE DEFAULT
    # ---------------------------------
    if "mode" not in st.session_state:
        st.session_state.mode = "Onboarding"

    mode = st.session_state.mode

    prompts = {
        "Onboarding":
            "Explain simply for beginners.",

        "Support":
            "Focus on troubleshooting and support.",

        "Architect":
            "Focus on architecture and systems.",

        "Leadership":
            "Summarize business insights."
    }

    placeholders = {
        "Onboarding":
            "Ask onboarding steps, tools, access...",

        "Support":
            "Ask leave policy, support issue, troubleshooting...",

        "Architect":
            "Explain SAP Commerce architecture, APIs...",

        "Leadership":
            "Summarize business impact, risks, opportunities..."
    }

    # ---------------------------------
    # STICKY HEADER
    # ---------------------------------
    st.markdown(
        '<div class="hero-sticky">',
        unsafe_allow_html=True
    )

    render_hero()

    st.write("")

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        if st.button(
            "📘 Ask Policy",
            use_container_width=True
        ):
            st.session_state.mode = "Support"
            st.rerun()

    with q2:
        if st.button(
            "🏗 Architecture",
            use_container_width=True
        ):
            st.session_state.mode = "Architect"
            st.rerun()

    with q3:
        if st.button(
            "📄 Summarize File",
            use_container_width=True
        ):
            st.session_state.mode = "Onboarding"
            st.rerun()

    with q4:
        if st.button(
            "📈 Executive View",
            use_container_width=True
        ):
            st.session_state.mode = "Leadership"
            st.rerun()

    st.write("")

    c1, c2 = st.columns([5, 1])

    with c1:
        st.info(
            f"Current Mode: {mode}"
        )

    with c2:
        if st.button(
            "🗑 Clear Chat",
            use_container_width=True
        ):
            st.session_state.messages = []
            clear_chat()
            st.rerun()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # ---------------------------------
    # RIGHT PANEL FIRST
    # ---------------------------------
    left, right = st.columns(
        [3.2, 1],
        gap="large"
    )

    with right:

        files = len(
            os.listdir("data")
        )

        chunks = get_latest_chunk_count()

        st.markdown(
            f"""
            <div class="kpi">
                <h3>{files}</h3>
                <div>Stored Files</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="kpi">
                <h3>{chunks}</h3>
                <div>Indexed Chunks</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="card">
                <b>System Status</b><br><br>
                🟢 Ollama Running<br>
                🟢 Vector DB Ready<br>
                🟢 Model Loaded
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------------------------
    # LEFT CHAT AREA
    # ---------------------------------
    with left:

        st.markdown(
            '<div class="chat-region">',
            unsafe_allow_html=True
        )

        if len(
            st.session_state.messages
        ) == 0:

            st.markdown(
                """
                <div class="card"
                style="text-align:center;
                padding:2.5rem 2rem;">

                <h2>
                Welcome to Cognivault AI
                </h2>

                <p style="opacity:.75;">
                Ask policies, architecture,
                onboarding or executive
                insights.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            for msg in st.session_state.messages:

                with st.chat_message(
                    msg["role"]
                ):
                    st.write(
                        msg["content"]
                    )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        # ---------------------------------
        # INPUT
        # ---------------------------------
        question = st.chat_input(
            placeholders[mode]
        )

        if question:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            save_chat(
                st.session_state.messages
            )

            with st.spinner(
                "Thinking..."
            ):

                model = load_model()

                index, texts = load_vector_store()

                answer, chunks, t0, t1, t2 = ask_question(
                    model,
                    index,
                    texts,
                    question,
                    prompts[mode]
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            save_chat(
                st.session_state.messages
            )

            st.rerun()