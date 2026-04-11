import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from data_loader import load_medical_docs

FAISS_INDEX_PATH = "faiss_index"

def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def build_faiss_index():
    
    print("Loading documents...")
    docs = load_medical_docs()

    print("Building embeddings ..")
    embeddings = get_embeddings()

    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved to '{FAISS_INDEX_PATH}/' folder!")
    return vectorstore

def load_faiss_index():
    
    embeddings = get_embeddings()

    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("No index found — building a new one...")
        return build_faiss_index()


if __name__ == "__main__":
    build_faiss_index()