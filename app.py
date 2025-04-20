import streamlit as st
import fitz  # PyMuPDF
import chromadb
import os
from google import genai
from google.genai import types

from IPython.display import Markdown
# genai.__version__

import config.config as cfg

#with open(pdf_path, "rb") as f:
#    pdf_bytes = f.read()

# Configure Gemini
GOOGLE_API_KEY = ""
client = genai.Client(api_key=GOOGLE_API_KEY)
for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)

# Download documents 
'''
!wget -nv -O gemini.pdf https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf
document_eval_file = client.files.upload(file='gemini.pdf')
!wget -nv -O Myanmar.pdf https://prddsgofilestorage.blob.core.windows.net/api/event-featured-documents/file/2025_IFRC_MYANMAR_EQ_DISASTER_BRIEF_24H_Low_Res.pdf
document_sim_file = client.files.upload(file='Myanmar.pdf')
!wget -nv -O Myanmar_SitRep.pdf https://www.unocha.org/attachments/2c9df378-1961-4841-bfea-2a37c31a5c9d/Myanmar%20Earthquake%20Situation%20Report%201_final.pdf 
document_sitrep_file = client.files.upload(file='Myanmar_SitRep.pdf')
!wget -nv -O EQ_Guidelines.pdf https://www.urban-response.org/system/files/content/resource/files/main/26164-earthquakeguidelinesenweb.pdf
document_guide_file = client.files.upload(file='EQ_Guidelines.pdf')
'''

# Enter EQ details
# Example
location = 'Los Angeles'
magnitude = 6.8
population_density = 'high'
time = '03:45 AM'

# Simulate EQ scenarios based on given variables and documents
def sim_scenario():
    request = f"""
    You're a crisis simulator. Create a realistic earthquake scenario based on:
    Location: {location}
    Magnitude: {magnitude}
    Population Density: {population_density}
    Time: {time}
    """
    model_config = types.GenerateContentConfig(temperature=0.1, top_p=0.95)

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        config=model_config,
        contents=[request, cfg.document_sim_file]) #contents=[request] uncomment this for runnning without document

    final_resp = response.text
    # print(final_resp)
    Markdown(final_resp)
    # Set up Chroma
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection("earthquake_docs")

# ------------------- RAG Functions -------------------
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
