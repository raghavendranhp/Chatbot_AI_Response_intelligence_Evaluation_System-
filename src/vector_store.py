import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DB_PATH = os.path.join(os.getcwd(), "vector_db")

def get_embedding_function():
    """
    Initialize the embedding model.
    Using the full path ensures HuggingFace finds it correctly.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_db(chunks):
    """Creates and saves the FAISS index to disk."""
    embeddings = get_embedding_function()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_PATH)
    print(f" Vector DB saved to {DB_PATH}")

def load_vector_db():
    """Loads the FAISS index from disk."""
    embeddings = get_embedding_function()
    if not os.path.exists(DB_PATH):
        print(f" Vector DB not found at {DB_PATH}")
        return None
    
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)