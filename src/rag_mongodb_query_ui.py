"""
RAG System with Tkinter UI

IMPORTANT USAGE INSTRUCTIONS:
This will not work in vscode. this needs to be run in the terminal outside VSCode
1. Activate the virtual environment first:
   $ source .venv/bin/activate  # On macOS/Linux
   $ .\.venv\Scripts\activate   # On Windows

2. Run this script:
   $ python src/rag_mongodb_query_ui.py

"""

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
import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter import messagebox
import threading

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
        return None, "Error: MongoDB connection string not found in .env file"
    
    try:
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
        
        return vector_store, f"Successfully connected to MongoDB vector store in database '{db_name}', collection '{collection_name}'"
    except Exception as e:
        import traceback
        error_msg = f"Error connecting to MongoDB vector store: {e}"
        traceback.print_exc()
        return None, error_msg

def setup_qa_chain(vector_store, model_name):
    """Set up a question-answering chain using the MongoDB vector store"""
    if not groq_api_key:
        return None, "Error: Groq API key not found in .env file"
    
    try:
        # Initialize language model using Groq with selected model
        llm = ChatGroq(
            model_name=model_name,
            temperature=0,
            api_key=groq_api_key
        )
        
        # Create retriever from vector store with better parameters
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Test the retriever with a simple query to verify it works
        test_docs = retriever.get_relevant_documents("test")
        retriever_status = ""
        if test_docs:
            retriever_status = f"Retriever test successful: found {len(test_docs)} documents"
        else:
            retriever_status = "Warning: Retriever test returned no documents"
            
        # Import the required classes for creating a custom prompt
        from langchain.prompts import PromptTemplate
        
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
        
        return qa_chain, retriever_status
    except Exception as e:
        import traceback
        error_msg = f"Error setting up QA chain: {e}"
        traceback.print_exc()
        return None, error_msg

def process_query(qa_chain, query: str):
    """Process a query using the QA chain and return the result"""
    if not qa_chain:
        return "Error: QA chain not initialized", []
    
    try:
        # Check what documents would be retrieved
        docs = qa_chain.retriever.get_relevant_documents(query)
        
        if len(docs) == 0:
            return "Warning: No relevant documents found for this query!", []
            
        start_time = time.time()
        result = qa_chain({"query": query})
        end_time = time.time()
        
        answer = result.get("result")
        source_docs = result.get("source_documents", [])
        
        time_taken = f"\n\nTime taken: {end_time - start_time:.2f} seconds"
        
        return answer + time_taken, source_docs
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing query: {e}", []

class RAGApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("MongoDB RAG System")
        self.root.geometry("1200x800")
        self.vector_store = None
        self.qa_chain = None
        
        self.setup_ui()
        
        # Start initialization in a separate thread to keep UI responsive
        threading.Thread(target=self.initialize_system, daemon=True).start()
    
    def setup_ui(self):
        # Create frames
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)
        
        middle_frame = ttk.Frame(self.root, padding=10)
        middle_frame.pack(fill=tk.BOTH, expand=True)
        
        bottom_frame = ttk.Frame(self.root, padding=10)
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model selection dropdown
        ttk.Label(top_frame, text="Select Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        model_values = [model["display"] for model in AVAILABLE_MODELS.values()]
        self.model_dropdown = ttk.Combobox(top_frame, textvariable=self.model_var, values=model_values)
        self.model_dropdown.current(0)  # Set default to first model
        self.model_dropdown.pack(side=tk.LEFT, padx=10)
        
        # Button to set model
        self.set_model_button = ttk.Button(top_frame, text="Set Model", command=self.change_model)
        self.set_model_button.pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing system...")
        ttk.Label(top_frame, textvariable=self.status_var).pack(side=tk.RIGHT)
        
        # Question and answer boxes (side by side)
        left_panel = ttk.Frame(middle_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_panel = ttk.Frame(middle_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Question area
        ttk.Label(left_panel, text="Your Question:").pack(anchor=tk.W)
        self.question_box = scrolledtext.ScrolledText(left_panel, wrap=tk.WORD, height=10)
        self.question_box.pack(fill=tk.BOTH, expand=True)
        self.ask_button = ttk.Button(left_panel, text="Ask Question", command=self.ask_question)
        self.ask_button.pack(pady=10)
        
        # Answer area
        ttk.Label(right_panel, text="Answer:").pack(anchor=tk.W)
        self.answer_box = scrolledtext.ScrolledText(right_panel, wrap=tk.WORD, height=10)
        self.answer_box.pack(fill=tk.BOTH, expand=True)
        
        # Sources area
        ttk.Label(bottom_frame, text="Sources:").pack(anchor=tk.W)
        self.sources_box = scrolledtext.ScrolledText(bottom_frame, wrap=tk.WORD, height=10)
        self.sources_box.pack(fill=tk.BOTH, expand=True)
        
        # End chat button
        self.end_button = ttk.Button(self.root, text="End Chat", command=self.root.destroy)
        self.end_button.pack(pady=10)
    
    def initialize_system(self):
        """Initialize the RAG system"""
        self.update_status("Connecting to MongoDB vector store...")
        self.vector_store, msg = connect_to_mongodb_vectorstore()
        self.update_status(msg)
        
        if self.vector_store:
            self.update_status("Setting up question answering system...")
            self.change_model()
        else:
            self.update_status("Failed to initialize. Check console for details.")
    
    def update_status(self, message):
        """Update status message in the UI"""
        def _update():
            self.status_var.set(message)
        self.root.after(0, _update)  # Schedule UI update from any thread
    
    def change_model(self):
        """Change the LLM model based on dropdown selection"""
        if not self.vector_store:
            messagebox.showerror("Error", "Vector store not initialized")
            return
        
        # Get selected model
        selected_display = self.model_var.get()
        selected_model = None
        for model_id, model_info in AVAILABLE_MODELS.items():
            if model_info["display"] == selected_display:
                selected_model = model_info
                break
        
        if not selected_model:
            selected_model = AVAILABLE_MODELS["1"]  # Default to first model
        
        self.update_status(f"Setting up {selected_model['display']} model...")
        
        # Setup QA chain in a separate thread
        def setup_model():
            self.qa_chain, status = setup_qa_chain(self.vector_store, selected_model["name"])
            if self.qa_chain:
                self.update_status(f"Ready with {selected_model['display']} model")
            else:
                self.update_status(f"Error setting up model: {status}")
        
        threading.Thread(target=setup_model, daemon=True).start()
    
    def ask_question(self):
        """Process the question from the user"""
        question = self.question_box.get("1.0", tk.END).strip()
        if not question:
            messagebox.showinfo("Info", "Please enter a question.")
            return
            
        if not self.qa_chain:
            messagebox.showerror("Error", "QA system not initialized")
            return
        
        # Clear previous answers and sources
        self.answer_box.delete("1.0", tk.END)
        self.sources_box.delete("1.0", tk.END)
        
        self.update_status("Processing question...")
        
        # Process query in a separate thread
        def process():
            answer, sources = process_query(self.qa_chain, question)
            
            # Update UI with results
            def update_ui():
                self.answer_box.delete("1.0", tk.END)
                self.answer_box.insert(tk.END, answer)
                
                self.sources_box.delete("1.0", tk.END)
                if not sources:
                    self.sources_box.insert(tk.END, "No source documents were returned")
                else:
                    for i, doc in enumerate(sources):
                        source = doc.metadata.get('source', 'Unknown source')
                        self.sources_box.insert(tk.END, f"Source {i+1}: {source}\n")
                
                self.update_status("Ready")
            
            self.root.after(0, update_ui)
            
        threading.Thread(target=process, daemon=True).start()

def main():
    """Main function to start the Tkinter UI"""
    root = tk.Tk()
    app = RAGApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()