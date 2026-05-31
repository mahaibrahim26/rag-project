import chromadb
import ollama
import gradio as gr

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("context_engineering")

# Retrieval
def retrieve(query):
    response = ollama.embeddings(model="llama3.2", prompt=query)
    query_embedding = response["embedding"]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    context = ""
    for doc, meta in zip(documents, metadatas):
        context += f"[{meta['source']}]\n{doc}\n\n"
    return context

# Without RAG
def no_rag(question):
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": question}]
    )
    return response["message"]["content"]

# With RAG
def rag_chat(question):
    context = retrieve(question)
    prompt = f"""You are an expert assistant on context engineering in AI and RAG systems.
Use the context below to give a detailed and complete answer to the question.
Base your answer on the context provided. Be thorough and explain clearly.

Context:
{context}

Question:
{question}

Provide a detailed answer based on the context above:"""
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# UI
with gr.Blocks(title="Context Engineering RAG Chatbot") as demo:
    gr.Markdown("# 🧠 Context Engineering RAG Chatbot")
    gr.Markdown("Compare answers with and without RAG.")
    question = gr.Textbox(label="Your Question", placeholder="e.g. What is context engineering?")
    button = gr.Button("Ask", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("###  With RAG")
            rag_output = gr.Textbox(label="Answer with RAG", lines=10)
        with gr.Column():
            gr.Markdown("###  Without RAG")
            no_rag_output = gr.Textbox(label="Answer without RAG", lines=10)
    button.click(
        fn=lambda q: (rag_chat(q), no_rag(q)),
        inputs=question,
        outputs=[rag_output, no_rag_output]
    )

demo.launch(share=True)