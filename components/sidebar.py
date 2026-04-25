import streamlit as st

def render_sidebar():

    with st.sidebar:

        st.markdown("""
        <div style="padding:1rem 0 1rem 0;">
            <div style="font-size:1.8rem;font-weight:800;">
                🧠 Cognivault AI
            </div>
            <div style="opacity:0.85;font-size:0.95rem;">
                Intelligent Knowledge. Trusted Decisions.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "💬 Chat Workspace",
            use_container_width=True
        ):
            st.session_state.page = "Chat"

        if st.button(
            "📊 Governance Center",
            use_container_width=True
        ):
            st.session_state.page = "Governance"

        st.markdown("### Assistant Mode")

        current_mode = st.session_state.get(
            "mode",
            "Onboarding"
        )

        mode_list = [
            "Onboarding",
            "Support",
            "Architect",
            "Leadership"
        ]

        mode = st.selectbox(
            "Mode",
            mode_list,
            index=mode_list.index(current_mode),
            label_visibility="collapsed"
        )

        st.session_state.mode = mode

        with st.expander(
            "Quick Tips",
            expanded=False
        ):
            st.write("• Ask policy questions")
            st.write("• Explain architecture")
            st.write("• Summarize files")
            st.write("• Executive insights")

        st.caption("Cognivault V8.1")