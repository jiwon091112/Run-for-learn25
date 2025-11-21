import json
import re
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings # Uncomment for OpenAI

# --- Configuration ---
FILE_PATH = "data.json"
DB_FAISS_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # A fast, popular local model

load_dotenv()

def load_articles_from_json(file_path="data.json"):
    """
    Loads articles from the specified JSON file.
    Assumes the file contains a valid JSON list.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                print(f"Successfully loaded {len(data)} articles from {file_path}.")
                return data
            else:
                print(f"Error: JSON file '{file_path}' does not contain a list.")
                return []
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}.")
        print("Please ensure 'data.txt' contains only the valid JSON list: [ { ... }, { ... } ]")
        return []
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading the file: {e}")
        return []

def clean_text(text):
    """
    Cleans the article text by removing tags.
    """
    if not text:
        return ""
    # Remove tags
    cleaned_text = re.sub(r"\\", '', text)
    # Remove excessive newlines/tabs and strip whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def create_langchain_documents(articles):
    """
    Converts the list of article dictionaries into LangChain Document objects.
    """
    docs = []
    for article in articles:
        # Combine title, desc, and main content for a comprehensive document
        content = f"Title: {clean_text(article.get('title'))}\n\n"
        #content += f"Description: {clean_text(article.get('desc'))}\n\n"
        #content += f"Content: {clean_text(article.get('dic_area'))}"
        
        # Metadata will be stored with the vector
        metadata = {
            "url": article.get("url"),
            "press": article.get("press"),
            "time": article.get("time"),
            "title": article.get("title"),
            "dic_area": article.get("dic_area") # Keep original title in metadata
        }
        
        # Ensure metadata values are strings or basic types
        for key, value in metadata.items():
            if value is None:
                metadata[key] = "N/A"
        
        docs.append(Document(page_content=content, metadata=metadata))
    
    print(f"Created {len(docs)} LangChain Document objects.")
    return docs

def split_documents(docs):
    """
    Splits the documents into smaller chunks for better embedding and retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Max size of each chunk
        chunk_overlap=200,  # Overlap between chunks to maintain context
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} text chunks.")
    return chunks

def get_embedding_model():
    """
    Initializes and returns the embedding model.
    """
    print(f"Initializing embedding model: {EMBEDDING_MODEL}")
    # Uses HuggingFace (local) embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'} # Use 'cuda' if you have a GPU
    )
    
    # --- Alternative: Use OpenAI (requires API key) ---
    #os.environ["OPENAI_API_KEY"] = 
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    #print("Initializing OpenAI embedding model.")
    
    return embeddings

def create_and_save_vector_store(chunks, embeddings, db_path):
    """
    Creates the FAISS vector store from chunks and saves it locally.
    """
    print("Creating vector store in memory...")
    # This creates the vector store from the document chunks
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print(f"Saving vector store to local disk at '{db_path}'...")
    # Save the vector store to disk
    vector_store.save_local(db_path)
    print("Vector store saved successfully.")
    return vector_store

def test_vector_store(db_path, embeddings):
    """
    Loads the saved vector store and performs a test similarity search.
    """
    print("\n--- Testing Vector DB ---")
    if not os.path.exists(db_path):
        print(f"Error: Database path '{db_path}' not found.")
        return

    # Load the local vector store
    # allow_dangerous_deserialization is needed for FAISS (which uses pickle)
    try:
        db = FAISS.load_local(
            db_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded from disk.")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        print("Did you forget to install 'faiss-cpu'?")
        return

    # Perform a similarity search
    """   query = "중국인 건강보험 적자"
    print(f"Performing similarity search for: '{query}'")
    results = db.similarity_search(query, k=2) # Get top 2 results
    
    if not results:
        print("No results found.")
        return

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source (URL): {doc.metadata.get('url')}")
        print(f"Title: {doc.metadata.get('title')}")
        print(f"Match Score (Distance): {doc.metadata.get('_distance', 'N/A')}") # If using search_with_score
        print(f"Snippet: {doc.page_content[:250]}...")
        print("---------------------")
    """
def main():
    # 1. Load Data
    articles = load_articles_from_json(FILE_PATH)
    if not articles:
        return

    # 2. Create Documents
    docs = create_langchain_documents(articles)

    # 3. Split Documents
    chunks = split_documents(docs)

    # 4. Initialize Embeddings
    embeddings = get_embedding_model()

    # 5. Create and Save Vector DB
    create_and_save_vector_store(chunks, embeddings, DB_FAISS_PATH)
    
    # 6. Test the DB
    test_vector_store(DB_FAISS_PATH, embeddings)


if __name__ == "__main__":
    main()