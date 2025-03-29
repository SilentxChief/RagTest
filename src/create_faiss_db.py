import os
import glob
import argparse
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_documents(directory_path):
    """Load text documents from a directory."""
    try:
        logger.info(f"Loading documents from {directory_path}")
        loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []

def create_vector_db(documents, output_dir):
    """Create a vector database from documents."""
    try:
        if not documents:
            logger.warning("No documents to process")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(text_chunks)} text chunks")
        
        # Create embeddings with MiniLM-L6-v2 model
        logger.info("Creating embeddings using MiniLM-L6-v2 model")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        
        # Create FAISS vector store
        logger.info("Creating FAISS vector database")
        vector_db = FAISS.from_documents(
            documents=text_chunks,
            embedding=embeddings
        )
        
        # Save FAISS index to disk
        faiss_path = os.path.join(output_dir, "faiss_index")
        vector_db.save_local(faiss_path)
        logger.info(f"FAISS vector database created and saved to {faiss_path}")
        
        return vector_db
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Create a vector database from text documents")
    parser.add_argument("--input_dir", type=str, default="./data", help="Directory containing text files")
    parser.add_argument("--output_dir", type=str, default="./db", help="Directory to save the vector database")
    args = parser.parse_args()
    
    # Load documents
    documents = load_documents(args.input_dir)
    
    # Create vector database
    create_vector_db(documents, args.output_dir)

if __name__ == "__main__":
    main()
