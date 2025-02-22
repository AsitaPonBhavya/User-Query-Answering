# 📄 RAG-Based LLM Document QA System

This application leverages **Retrieval-Augmented Generation (RAG)** to answer user queries based on an uploaded document. It supports **PDFs, URLs, and text inputs**, extracts relevant information using **FAISS**, and generates answers using **transformer-based NLP models**.

## 🚀 Features
- 📄 **Multi-format Input**: Supports **PDF, raw text, and URLs** as input sources.
- 🔍 **Embedding-based Retrieval**: Uses **FAISS** for efficient document chunk retrieval.
- 🤖 **LLM for Answer Generation**: Utilizes **DistilBERT** for **question-answering**.
- 🖥 **Streamlit UI**: User-friendly interface for input and querying.

## 🛠 Technologies Used
- **Python** 🐍  
- **FAISS** 🔎 (for fast vector search)  
- **Hugging Face Transformers** 🤗 (SentenceTransformer & DistilBERT)  
- **PyPDF2** 📄 (for PDF text extraction)  
- **BeautifulSoup** 🌐 (for scraping webpage text)  
- **NumPy** 🔢 (for array operations)  
- **Streamlit** 🖥 (for UI)  

## 📝 Usage
Select an input type (PDF, Text, or URL) from the sidebar.
Upload a file, enter text, or provide a URL.
Enter a question based on the document.
The app retrieves relevant information and generates an answer.
