import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents():
    """
    Load ONLY unstructured text files (.txt).
    We explicitly point to 'data/unstructured' to avoid reading the CSV here.
    """
    #define the path to the unstructured folder
    data_path = os.path.join("data", "unstructured")
    
    print(f"Loading text documents from {data_path}...")
    
    if not os.path.exists(data_path):
        print(f" Warning: Directory {data_path} not found.")
        return []

    #load only txt files
    loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} text documents.")
    return documents

def chunk_documents(documents):
    """
    Split text into chunks.
    """
    print("Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    docs = load_documents()
    if docs:
        chunks = chunk_documents(docs)