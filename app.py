import streamlit as st
import fitz  # PyMuPDF
import chromadb
from google.generativeai import GenerativeModel, configure
from google.generativeai.types import Content
import os

# Configure Gemini
configure(api_key="YOUR_GEMINI_API_KEY")
model = GenerativeModel('gemini-pro')

# Set up Chroma
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("earthquake_docs")

# ------------------- Utility Functions -------------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_text(text):
    embed_model = GenerativeModel('models/text-embedding-004')
    response = embed_model.embed_content(
        contents=text,
        task_type='RETRIEVAL_DOCUMENT'
    )
    return response.embedding

def store_chunks(chunks):
    for i, chunk in enumerate(chunks):
        emb = embed_text(chunk)
        collection.add(documents=[chunk], embeddings=[emb], ids=[f"chunk_{i}"])

def retrieve_similar_chunks(query, k=5):
    emb = embed_text(query)
    results = collection.query(query_embeddings=[emb], n_results=k)
    return results['documents'][0]  # list of top-k docs

def generate_response_rag(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)
    prompt = f"""
    Context: {context}
    
    Task: Generate an earthquake response plan based on the documents above. 
    Please structure the output in JSON with keys: 'ResponseObjectives', 'PriorityAreas', 'ResourcesNeeded'.
    
    User query: {query}
    """
    response = model.generate_content(prompt)
    return response.text

# ------------------- Streamlit UI -------------------
st.title("Earthquake Response Plan Generator")

uploaded_file = st.file_uploader(" Upload a disaster report (PDF)", type="pdf")

query = st.text_input("Ask your question (e.g., 'What areas were most affected?')")

if uploaded_file and query:
    with st.spinner("Processing..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        store_chunks(chunks)
        retrieved = retrieve_similar_chunks(query)
        response = generate_response_rag(query, retrieved)
    
    st.subheader("GenAI RAG Response")
    st.code(response, language="json")

elif not uploaded_file:
    st.info("Please upload a disaster report to get started.")
elif not query:
    st.info("Enter a question to generate the response.")
