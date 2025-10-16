**PDF Question Answering System**
Ask questions about PDF documents using AI. Built with LlamaIndex, ChromaDB, and Groq.

Quick Start
1. Install
bashpip install llama-index llama-index-llms-groq llama-index-embeddings-huggingface chromadb llama-index-vector-stores-chroma

2. Setup
pythonimport os
os.environ["GROQ_API_KEY"] = "your-api-key-here"  # Get from https://console.groq.com

3. Index Your Documents
pythonfrom llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

**# Load PDFs**
documents = SimpleDirectoryReader(input_dir="path/to/pdfs").load_data()

# Create embeddings and store

embed_model = HuggingFaceEmbedding()
db = chromadb.PersistentClient(path="./vector_db")
collection = db.get_or_create_collection("my_docs")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build index

index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context,
    embed_model=embed_model
)
**4. Ask Questions**

pythonfrom llama_index.llms.groq import Groq
from llama_index.core import Settings

Settings.llm = Groq(model="llama-3.1-8b-instant")

# Load index

db = chromadb.PersistentClient(path="./vector_db")
collection = db.get_collection("my_docs")
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# Query

query_engine = index.as_query_engine()
response = query_engine.query("Your question here?")
print(response.response)

Example Questions

For BERT and Transformer papers:

"What are the key differences between BERT and Transformer?"
"How is BERT pre-trained?"
"What is multi-head attention?"
"Explain the attention mechanism"


Project Structure
.
├── doc_dir/              # Your PDF files
├── vector_db/            # Database (auto-created)
└── notebook.ipynb        # Your code

How It Works

Load - Reads your PDF files
Chunk - Splits into smaller pieces (1024 chars)
Embed - Converts text to vectors using HuggingFace
Store - Saves in ChromaDB
Query - Finds relevant chunks for your question
Answer - Groq LLM generates response


Troubleshooting
No files found?
python# Check your files
import os
for file in os.listdir("path/to/pdfs"):
    if file.endswith('.pdf'):
        print(file)
API key error?
python# Verify it's set
print(os.environ.get("GROQ_API_KEY"))

Technologies

LlamaIndex - RAG framework
ChromaDB - Vector database
HuggingFace - Embeddings (BAAI/bge-small-en-v1.5)
Groq - LLM (llama-3.1-8b-instant)


Documents Used

Attention Is All You Need (Transformer paper)
BERT paper


Note: Run indexing code first, then querying code. Replace API key before running.
