# Context Engineering RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built for BCS28a Engineering Teamwork I — Summer 2026.
Topic: Context Engineering in RAG Systems


## What This Project Does

This chatbot demonstrates the difference between an LLM answering from memory versus answering using retrieved context from real documents. It shows how RAG reduces hallucination and improves answer accuracy.

## Data Sources

The system uses 4 different data formats as required:

- PDF — Context Engineering 2.0 research paper
- Text File — Custom notes on context engineering
- Website — Prompting Guide context engineering article
- YouTube — Context engineering video transcript

## How It Works

1. `vector.py` loads all 4 data sources, chunks them, generates embeddings using Ollama llama3.2, and stores everything in a ChromaDB vector database
2. `app.py` takes a user question, converts it to an embedding, searches ChromaDB for the most relevant chunks, builds a prompt with the retrieved context, and sends it to Ollama llama3.2 to generate an answer
3. The interface shows both the RAG answer and the non-RAG answer side by side so you can directly compare them

## Tech Stack

- Python
- ChromaDB — vector database with HNSW indexing
- Ollama llama3.2 — local LLM and embedding model
- Gradio — chat interface
- pdfplumber / pypdf — PDF loading
- BeautifulSoup — website scraping
- youtube-transcript-api — YouTube transcript extraction

## Setup

Make sure you have Ollama installed and llama3.2 downloaded:
ollama pull llama3.2

Install dependencies:
pip install chromadb ollama gradio pdfplumber requests beautifulsoup4 youtube-transcript-api pypdf

Build the vector database (run once):
python vector.py

Start the chatbot:
python app.py

## Project Structure
rag-project/
├── data/
│   ├── context engineering paper.pdf
│   ├── context_engineering.txt
│   └── links.txt
├── vector.py        # Data loading, chunking, embeddings, ChromaDB
├── app.py           # RAG chatbot interface
├── requirements.txt
└── README.md

## Key Concepts Demonstrated

- Text chunking with overlap strategy
- Embedding generation using a local LLM
- Vector similarity search with cosine similarity
- Context engineering — how the retrieved chunks are structured in the prompt
- RAG vs no RAG comparison showing hallucination reduction