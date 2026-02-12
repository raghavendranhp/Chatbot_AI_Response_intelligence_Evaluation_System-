from src.vector_store import load_vector_db

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class UnstructuredSearchEngine:
    """
    Handles semantic search (RAG) on the text documents (Brochures, FAQs).
    """

    def __init__(self, vector_db_path="vector_db"):
        self.vector_db_path = vector_db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        
        self._load_db()

    def _load_db(self):
        """Loads the FAISS index from disk."""
        if os.path.exists(self.vector_db_path):
            try:
                #Allow dangerous deserialization because we trust our own local file
                self.vector_store = FAISS.load_local(
                    self.vector_db_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"[ERROR] Failed to load Vector DB: {e}")
        else:
            print(f"[WARNING] Vector DB not found at {self.vector_db_path}. RAG will fail.")

    def search(self, query: str, k: int = 3) -> list:
        """
        Retrieves top k relevant text chunks for the query.
        Returns a list of strings.
        """
        if not self.vector_store:
            return ["(Vector Database is not loaded. Cannot retrieve documents.)"]

        try:
            #Perform similarity search
            docs = self.vector_store.similarity_search(query, k=k)
            #Return just the content text
            return [doc.page_content for doc in docs]
        except Exception as e:
            print(f"[ERROR] RAG Search failed: {e}")
            return ["(Error occurred during document retrieval)"]