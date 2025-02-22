import streamlit as st
import os
import faiss
import requests
from bs4 import BeautifulSoup
import numpy as np
from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer, AutoModel

def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def load_text(text):
    return text

def load_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = ' '.join(map(lambda p: p.get_text(), soup.find_all('p')))
    return text

def create_embeddings(text, tokenizer, model):
    chunk_size = 512
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())

    embeddings = np.vstack(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return chunks, index

def retrieve_and_generate_answer(query, chunks, index, tokenizer, model, qa_pipeline):
    query_inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    query_embedding = model(**query_inputs).last_hidden_state.mean(dim=1).detach().numpy()

    _, indices = index.search(query_embedding, k=3)
    relevant_chunks = [chunks[i] for i in indices[0]]

    context = " ".join(relevant_chunks)
    response = qa_pipeline(question=query, context=context)

    return response['answer']

# Streamlit UI
st.title("Document QA System")

st.sidebar.title("Input Options")
input_option = st.sidebar.selectbox("Select input format:", ("PDF", "Text", "URL"))

document_text = ""

if input_option == "PDF":
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        document_text = load_pdf(uploaded_file)
elif input_option == "Text":
    document_text = st.sidebar.text_area("Enter text:")
elif input_option == "URL":
    url = st.sidebar.text_input("Enter URL:")
    if url:
        document_text = load_url(url)

if document_text:
    st.write("Document loaded successfully!")

    # Load models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    chunks, index = create_embeddings(document_text, tokenizer, model)

    query = st.text_input("Enter your query:")
    if query:
        answer = retrieve_and_generate_answer(query, chunks, index, tokenizer, model, qa_pipeline)
        st.write("Answer:", answer)
