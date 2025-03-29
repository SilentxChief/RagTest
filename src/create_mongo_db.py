import os
import glob
from typing import List
from dotenv import load_dotenv
import pymongo
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch

# Load environment variables from .env file 
load_dotenv()
connection_string = os.getenv("MONGODB_CONNECTION_STRING")

def load_documents(directory_path: str) -> List[Document]:
    """Load documents from the specified directory"""
    documents = []
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist!")
        return []
    
    # Get all text files in the directory
    file_paths = glob.glob(os.path.join(directory_path, "**/*.txt"), recursive=True)
    
    if not file_paths:
        print(f"No text files found in {directory_path}!")
        return []
    
    print(f"Found {len(file_paths)} text files to process.")
    
    # Read each file and create documents
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                # Create a Document with metadata including source
                doc = Document(
                    page_content=text,
                    metadata={"source": file_path}
                )
                documents.append(doc)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    return documents

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks"""
    # Create text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Split documents into chunks
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Created {len(chunked_documents)} document chunks.")
    
    return chunked_documents

def create_vector_index(collection):
    """Create vector index for the MongoDB collection"""
    try:
        # Define index model for vector search
        index_model = {
            "name": "vector_index",
            "definition": {
                "mappings": {
                    "dynamic": True,
                    "fields": {
                        "embedding": {
                            "dimensions": 384,  # MiniLM-L6-v2 has 384 dimensions
                            "similarity": "cosine",
                            "type": "knnVector"
                        }
                    }
                }
            }
        }
        
        # Create the index
        collection.create_search_index(index_model)
        print("Vector search index created successfully")
        return True
    except Exception as e:
        print(f"Error creating vector index: {e}")
        return False

def create_mongodb_vectorstore(documents: List[Document]) -> None:
    """Create MongoDB vector store with document embeddings"""
    if not connection_string:
        print("Error: connection_string not found in .env file")
        return None
    
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
        )
        
        # Alternative approach with explicit index creation
        from pymongo import MongoClient
        from langchain_community.vectorstores import MongoDBAtlasVectorSearch

        # Connect to MongoDB
        client = MongoClient(connection_string)
        db_name = "RAG"
        collection_name = "sample_collection"
        collection = client[db_name][collection_name]

        # Drop existing collection if needed
        if collection_name in client[db_name].list_collection_names():
            collection.drop()
            print(f"Dropped existing collection '{collection_name}'")
            
        # Create vector store
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents,
            embeddings,
            collection=collection,
            index_name="vector_index"  # Specify the index name
        )
        
        # Create vector index for the collection
        create_vector_index(collection)

        return vector_store
    except Exception as e:
        print(f"Error creating MongoDB vector store: {e}")
        return None

def main():
    """Main function to create MongoDB vector database"""
    print("Starting MongoDB vector database creation...")
    
    # Define the directory containing documents
    docs_directory = "./data/"
    
    # Load documents
    print("Loading documents...")
    documents = load_documents(docs_directory)
    if not documents:
        print("No documents loaded. Exiting.")
        return
    
    # Chunk documents
    print("Chunking documents...")
    chunked_documents = chunk_documents(documents)
    
    # Create MongoDB vector store
    print("Creating MongoDB vector store...")
    vector_store = create_mongodb_vectorstore(chunked_documents)
    
    if vector_store:
        print("MongoDB vector database creation complete!")
    else:
        print("Failed to create MongoDB vector database.")

if __name__ == "__main__":
    main()