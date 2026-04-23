import os
import streamlit as st

def get_api_key():
    
    if os.getenv("OPENROUTER_API_KEY"):
        return os.getenv("OPENROUTER_API_KEY")
    return st.secrets["OPENROUTER_API_KEY"]