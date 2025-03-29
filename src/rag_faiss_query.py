import os
import glob
from typing import List
from dotenv import load_dotenv
import faiss
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.schema import SystemMessage

# Load environment variables from .env file 
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model options
MODEL_OPTIONS = {
    "1": "llama3-8b-8192",
    "2": "llama3-70b-8192"
}

def load_vector_db():
    """Load an existing FAISS vector database from the db folder"""
    db_path = os.path.join(os.getcwd(), "./db/faiss_index")
    
    if not os.path.exists(db_path):
        print(f"Database directory {db_path} does not exist!")
        return None
    
    try:
        # Initialize embeddings with MiniLLM (same as used for creating the DB)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
        
        # Load the FAISS vector store
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print(f"Successfully loaded vector database from {db_path}")
        return vector_store
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return None

def get_llm_response(query: str, vector_store, model_choice: str):
    """Get LLM response using RAG with the selected model"""
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in .env file")
        return "API key missing. Please set GROQ_API_KEY in your .env file.", []
    
    # Get the model name from the choice
    model_name = MODEL_OPTIONS.get(model_choice)
    if not model_name:
        print(f"Invalid model choice: {model_choice}. Using default model.")
        model_name = MODEL_OPTIONS["1"]
    
    # System message to instruct the LLM
    system_message = """You are a helpful assistant providing information based on the retrieved documents.
    Please follow these guidelines:
    1. Answer questions concisely and directly
    2. Only use information found in the provided documents
    3. If you're unsure or the documents don't contain the answer, admit this clearly
    4. Use a friendly, conversational tone
    5. Do not hallucinate or make up information
    6. Focus on facts from the documents, not opinions"""
    
    # Initialize the LLM with Groq
    llm = ChatGroq(
        api_key=GROQ_API_KEY, 
        model_name=model_name,
        temperature=0.2
    )
    
    # Set the system message through the model's configuration
    system_message_obj = SystemMessage(content=system_message)
    messages = [system_message_obj]
    chat = llm.bind(messages=messages)
    
    # Create a retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Simple method to stuff all documents into the prompt
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    # Get the response
    result = qa_chain({"query": query})
    
    return result["result"], [doc.metadata["source"] for doc in result["source_documents"]]

def main():
    """Main function to run the RAG pipeline"""
    print("Welcome to the RAG system!")
    
    # Try to load existing vector database first
    vector_store = load_vector_db()
    if not vector_store:
        print("Exiting program.")
        return
    
    # Model selection
    print("\nAvailable models:")
    for key, model in MODEL_OPTIONS.items():
        print(f"{key}: {model}")
    
    model_choice = input("Choose a model (1 or 2): ")
    
    # Main interaction loop
    while True:
        # Get user query
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        print("Retrieving information and generating response...")
        response, sources = get_llm_response(query, vector_store, model_choice)
        
        print("\nResponse:")
        print(response)
        print("\nSources:")
        for source in set(sources):
            print(f"- {source}")
    
    print("Thank you for using the RAG system!")

if __name__ == "__main__":
    main()

