# ⚓ Naval RAG Assistant

**Naval RAG Assistant** is a local **Retrieval-Augmented Generation (RAG)** chatbot built with **Flask**, **SentenceTransformers**, and **FLAN-T5**.  
It answers questions from a naval recognition training PDF — including topics such as ship, aircraft, and submarine identification.

---

## 🧠 Overview

This project demonstrates how to build a **local question-answering system** using:

- 🗂️ **PDF Knowledge Base** – extracts and chunks content from `14243_ch13.pdf`
- 🔍 **Semantic Retrieval** – encodes text using `all-MiniLM-L6-v2` embeddings
- 🧮 **Similarity Search** – retrieves the most relevant chunks for a user’s query
- 🧠 **Local Generation** – uses `google/flan-t5-base` to produce human-readable answers
- 🌐 **Web Interface** – lightweight **Flask + Bootstrap** front-end for user interaction

All processing happens **locally** — no external APIs or internet access required.

---

## ⚙️ Tech Stack

| Component | Purpose |
|------------|----------|
| **Python 3.10+** | Core language |
| **Flask** | Web framework |
| **SentenceTransformers** | Text embeddings for retrieval |
| **Transformers (Hugging Face)** | FLAN-T5 local LLM |
| **PyPDF2** | PDF text extraction |
| **Bootstrap 5** | Front-end styling |

---

## 🚀 Setup & Usage

### 1. Create and activate environment
- conda create -n rag python=3.10 -y
- conda activate rag

### 2. Install dependencies
- pip install flask sentence-transformers transformers PyPDF2
- conda install -c pytorch pytorch cpuonly -y   # optional if torch not installed

### 3. Run the Flask app
python app.py

When initialized, open browser 

### Example Questions
- What are the recognition features of submarines?
- How are ships identified by their silhouettes?
- What are the different categories of naval vessels?
- What are the distinguishing characteristics of aircraft?

### Project Structure
naval_rag_assistant/
 - app.py               # Flask web interface
 - rag_chatbot.py       # Core RAG logic (PDF → embeddings → answers)
 - templates/
     - index.html       # Front-end UI (Bootstrap)
 - 14243_ch13.pdf       # Knowledge base document
