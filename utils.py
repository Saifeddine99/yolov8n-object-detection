import streamlit as st

def activation_callback():
    # Button was clicked!
    st.session_state.detection_button = True

def deactivation_callback():
    st.session_state.detection_button = False