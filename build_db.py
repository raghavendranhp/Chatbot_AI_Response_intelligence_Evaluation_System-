import os
import sys


sys.path.append(os.getcwd())

from src.ingestion import load_documents, chunk_documents
from src.vector_store import create_vector_db

def main():
    print("Starting Knowledge Base Build...")
    
    #load text files
    
    print(" Loading documents...")
    docs = load_documents()
    
    if not docs:
        print(" No documents found in 'data/unstructured/'.")
        print("   Please add .txt files (Brochures, Itineraries, FAQs) there.")
        return

    #Chunk Documents
    
    chunks = chunk_documents(docs)
    
    if not chunks:
        print(" Error: No chunks were created.")
        return

    #create Save Vector DB
    print(f" Saving {len(chunks)} chunks to Vector Database...")
    create_vector_db(chunks)
    
    print("\n  KBase Built Successfully!")
    print("   You can now run 'streamlit run app.py'")

if __name__ == "__main__":
    main()