from flask import Flask, render_template, request, redirect, url_for
import os
import time
import re  # added for cleaning text
import textwrap

# import your rag functions (assumes rag_chatbot.py is in same folder)
import rag_chatbot as rag

app = Flask(__name__)

# --- Initialization: load PDF, chunk, and build embeddings once ---
PDF_PATH = rag.PDF_PATH  # uses path from your rag script

print("Flask app: initializing RAG backend...")
start = time.time()

# extract, chunk, build embeddings
_doc_text = rag.extract_text(PDF_PATH)
_chunks = rag.chunk_text(_doc_text)
_embeddings, _embed_model = rag.build_embeddings(_chunks)

print(f"✅ Initialized RAG backend: {len(_chunks)} chunks (took {time.time()-start:.1f}s)")

# --- helper to run a query ---
def answer_query(query, top_k=rag.TOP_K):
    """
    Returns a dict with:
      - query
      - direct: extracted direct answer (string) or None
      - generated: generated answer (string) if used
    """
    retrieved_texts, scores, idxs = rag.retrieve_top_k(query, _chunks, _embeddings, _embed_model, k=top_k)

    # Try deterministic extraction first
    direct = rag.extract_direct_answer_v2(
        query,
        retrieved_texts,
        retrieved_ids=idxs,
        retrieved_scores=scores,
        max_sentences=6
    )

    generated = None

    # ✅ Clean out "[source: chunk ...]" text if present
    if direct:
        direct = re.sub(r'\[source:[^\]]+\]\s*', '', direct).strip()

    if not direct:
        # fallback to generator
        generated = rag.generate_answer(query, retrieved_texts, retrieved_ids=idxs)
        if generated:
            generated = re.sub(r'\[source:[^\]]+\]\s*', '', generated).strip()

    return {
        "query": query,
        "direct": direct,
        "generated": generated,
    }

# --- Flask routes ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)

@app.route("/ask", methods=["POST"])
def ask():
    q = request.form.get("question", "").strip()
    if not q:
        return redirect(url_for("index"))

    start = time.time()
    res = answer_query(q)
    elapsed = time.time() - start
    return render_template("index.html", result=res, elapsed=f"{elapsed:.2f}s")

# Run with: python app.py
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
