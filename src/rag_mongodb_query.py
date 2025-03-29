import os
from typing import List, Dict
from dotenv import load_dotenv
import pymongo
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import time
import json

# Load environment variables
load_dotenv()
connection_string = os.getenv("MONGODB_CONNECTION_STRING")
groq_api_key = os.getenv("GROQ_API_KEY")

# Available LLM models with Groq
AVAILABLE_MODELS = {
    "1": {"name": "llama3-70b-8192", "display": "Llama 3 70B (recommended)"},
    "2": {"name": "llama3-8b-8192", "display": "Llama 3 8B"}
}


def connect_to_mongodb_vectorstore():
    """Connect to the MongoDB vector store created by create_mongodb_db.py"""
    if not connection_string:
        print("Error: MongoDB connection string not found in .env file")
        return None
    
    try:
        print(f"Connecting to MongoDB with connection string: {connection_string[:20]}...")
        
        # Initialize embeddings - use the same model as in create_mongodb_db.py
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
        )
        
        # Connect to MongoDB using the same database and collection names
        client = pymongo.MongoClient(connection_string)
        db_name = "RAG"
        collection_name = "sample_collection"
    
    
        collection = client[db_name][collection_name]
        
        # Initialize the vector store
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index"
        )
        
        print(f"Successfully connected to MongoDB vector store in database '{db_name}', collection '{collection_name}'")
        return vector_store
    except Exception as e:
        print(f"Error connecting to MongoDB vector store: {e}")
        import traceback
        traceback.print_exc()
        return None

def select_model() -> Dict:
    """Allow user to select an LLM model"""
    print("\nAvailable LLM models:")
    for key, model in AVAILABLE_MODELS.items():
        print(f"{key}. {model['display']}")
    
    while True:
        choice = input("\nSelect a model (1-2) [default: 1]: ").strip()
        if choice == "":
            choice = "1"  # Default to Llama 3 70B
        
        if choice in AVAILABLE_MODELS:
            selected_model = AVAILABLE_MODELS[choice]
            print(f"Selected model: {selected_model['display']}")
            return selected_model
        else:
            print("Invalid choice. Please try again.")

def setup_qa_chain(vector_store):
    """Set up a question-answering chain using the MongoDB vector store"""
    if not groq_api_key:
        print("Error: Groq API key not found in .env file")
        return None
    
    try:
        # Let user select an LLM model
        selected_model = select_model()
        
        # Initialize language model using Groq with selected model
        llm = ChatGroq(
            model_name=selected_model["name"],
            temperature=0,
            api_key=groq_api_key
        )
        
        # Create retriever from vector store with better parameters
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Removed score_threshold as it may filter out all results
        )
        
        # Test the retriever with a simple query to verify it works
        print("Testing retriever with a sample query...")
        test_docs = retriever.get_relevant_documents("test")
        if test_docs:
            print(f"Retriever test successful: found {len(test_docs)} documents")
        else:
            print("Warning: Retriever test returned no documents")
            print("This suggests there might be an issue with the vector store or index.")
        
        # Import the required classes for creating a custom prompt
        from langchain.prompts import PromptTemplate
        from langchain.chains import RetrievalQA
        
        # Create a custom prompt template
        template = """Answer the question based ONLY on the following context:
        {context}
        
        Question: {question}
        
        If the context doesn't contain the information needed, say "I don't have enough information to answer this question."
        Provide a detailed answer and cite the sources from the context."""
        
        # Create the prompt with the template
        custom_prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain with the custom prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # This combines all documents into one context
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": custom_prompt
            }
        )
        
        return qa_chain
    except Exception as e:
        print(f"Error setting up QA chain: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_query(qa_chain, query: str):
    """Process a query using the QA chain and return the result"""
    if not qa_chain:
        print("Error: QA chain not initialized")
        return None
    
    try:
        print("\nProcessing query...")
        
        # Debug: First check what documents would be retrieved
        docs = qa_chain.retriever.get_relevant_documents(query)
        print(f"Retrieved {len(docs)} documents from MongoDB")
        if len(docs) == 0:
            print("Warning: No relevant documents found for this query!")
            
        start_time = time.time()
        result = qa_chain({"query": query})
        end_time = time.time()
        
        answer = result.get("result")
        source_docs = result.get("source_documents", [])
        
        print("\nAnswer:", answer)
        print(f"\nTime taken: {end_time - start_time:.2f} seconds")
        
        print("\nSources:")
        if not source_docs:
            print("No source documents were returned")
        for i, doc in enumerate(source_docs):
            print(f"Source {i+1}: {doc.metadata.get('source', 'Unknown source')}")
         
        
        return answer
    except Exception as e:
        print(f"Error processing query: {e}")
        return None

def main():
    """Main function for MongoDB RAG demo"""
    print("Initializing MongoDB RAG system...")
    
    # Connect to MongoDB vector store
    vector_store = connect_to_mongodb_vectorstore()
    if not vector_store:
        print("Failed to connect to MongoDB vector store. Exiting.")
        return
    
    # Set up QA chain
    qa_chain = setup_qa_chain(vector_store)
    if not qa_chain:
        print("Failed to set up QA chain. Exiting.")
        return
    
    print("\nMongoDB RAG system initialized! Enter your questions (type 'exit' to quit):")
    
    # Interactive query loop
    while True:
        query = input("\nQuestion: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting...")
            break
        
        process_query(qa_chain, query)

if __name__ == "__main__":
    main()