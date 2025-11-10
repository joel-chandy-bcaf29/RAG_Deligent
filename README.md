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

