import streamlit as st
from rag_pipeline import get_rag_answer

st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="",
    layout="centered"
)

st.markdown("""
<style>
    .main { max-width: 800px; margin: 0 auto; }
    .chat-message { padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
    .user-message { background-color: #2d2d2d; text-align: right; }
    .assistant-message { background-color: #1a1a2e; }
    .stTextInput input { border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("Medical AI Assistant")
st.caption("Ask about any disease, symptom, or medical condition.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask about any disease or medical condition...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching medical knowledge base..."):
            answer, sources = get_rag_answer(query)
        st.markdown(answer)

        with st.expander("View Retrieved Sources"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**Source {i}:** {src}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})