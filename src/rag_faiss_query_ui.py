"""
RAG System with Tkinter UI

IMPORTANT USAGE INSTRUCTIONS:
This will not work in vscode. this needs to be run in the terminal outside VSCode
1. Activate the virtual environment first:
   $ source .venv/bin/activate  # On macOS/Linux
   $ .\.venv\Scripts\activate   # On Windows

2. Run this script:
   $ python src/rag_faiss_query_ui.py

3. Ensure your .env file contains GROQ_API_KEY
"""

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
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading

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

class RagUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG System UI")
        self.root.geometry("1000x600")
        self.vector_store = None
        
        # Load vector database
        self.vector_store = load_vector_db()
        if not self.vector_store:
            error_label = tk.Label(root, text="Error: Could not load vector database. Closing in 5 seconds...", fg="red")
            error_label.pack(pady=20)
            root.after(5000, root.destroy)
            return
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Model selection
        model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding="10")
        model_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(model_frame, text="Choose Model:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for model selection
        self.model_var = tk.StringVar()
        model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly", width=20)
        model_dropdown['values'] = [f"{key}: {value}" for key, value in MODEL_OPTIONS.items()]
        model_dropdown.current(0)  # Set default to first model
        model_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Create chat frame (holds input and output side by side)
        chat_frame = ttk.Frame(main_frame)
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Input frame (left side)
        input_frame = ttk.LabelFrame(chat_frame, text="Enter Your Question", padding="10")
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.input_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=10)
        self.input_text.pack(fill=tk.BOTH, expand=True)
        
        # Submit button
        submit_button = ttk.Button(input_frame, text="Submit Question", command=self.process_query)
        submit_button.pack(fill=tk.X, pady=10)
        
        # Output frame (right side)
        output_frame = ttk.LabelFrame(chat_frame, text="Response", padding="10")
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.config(state=tk.DISABLED)
        
        # Sources frame
        sources_frame = ttk.LabelFrame(main_frame, text="Sources", padding="10")
        sources_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.sources_text = scrolledtext.ScrolledText(sources_frame, wrap=tk.WORD, height=5)
        self.sources_text.pack(fill=tk.BOTH, expand=True)
        self.sources_text.config(state=tk.DISABLED)
        
        # End chat button
        end_button = ttk.Button(main_frame, text="End Chat", command=self.root.destroy)
        end_button.pack(pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def process_query(self):
        # Get user input
        query = self.input_text.get("1.0", tk.END).strip()
        if not query:
            return
        
        # Clear previous output
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
        
        self.sources_text.config(state=tk.NORMAL)
        self.sources_text.delete("1.0", tk.END)
        self.sources_text.config(state=tk.DISABLED)
        
        # Update status
        self.status_var.set("Processing query...")
        
        # Get model choice (extracting the key part)
        model_choice = self.model_var.get().split(":")[0].strip()
        
        # Process in a separate thread to keep UI responsive
        threading.Thread(target=self._get_response, args=(query, model_choice), daemon=True).start()
    
    def _get_response(self, query, model_choice):
        try:
            # Get response
            response, sources = get_llm_response(query, self.vector_store, model_choice)
            
            # Update UI with response
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, response)
            self.output_text.config(state=tk.DISABLED)
            
            # Update sources
            self.sources_text.config(state=tk.NORMAL)
            for source in set(sources):
                self.sources_text.insert(tk.END, f"- {source}\n")
            self.sources_text.config(state=tk.DISABLED)
            
            # Update status
            self.status_var.set("Ready")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, f"An error occurred: {str(e)}")
            self.output_text.config(state=tk.DISABLED)

def main():
    """Main function to run the RAG UI"""
    root = tk.Tk()
    app = RagUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()