import streamlit as st

def render_hero():

    st.markdown("""
    <div class="hero">
        <div style="font-size:2.4rem;font-weight:800;">
            🧠 Cognivault AI
        </div>
        <div style="font-size:1.1rem;margin-top:0.3rem;">
            Your Enterprise Intelligence Platform
        </div>
        <div style="margin-top:0.8rem;opacity:0.9;">
            Ask documents • Retrieve trusted answers • Accelerate decisions
        </div>
    </div>
    """, unsafe_allow_html=True)