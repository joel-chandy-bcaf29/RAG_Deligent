"""
rag_chatbot.py — Final refined version
--------------------------------------
Local Retrieval-Augmented Generation chatbot using:
 • PyPDF2 to extract text
 • SentenceTransformers for embeddings
 • FLAN-T5 for local text generation (no API key)
 • Improved deterministic extractor with naval keywords
"""

import os
import re
import textwrap
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ---------------- CONFIG ----------------
PDF_PATH = r"C:\Users\jjcha\Downloads\naval_rag_assistant\14243_ch13.pdf"  # update if needed
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 8

GEN_MODEL = "google/flan-t5-base"  # change to flan-t5-small for faster CPU
GEN_MAX_LENGTH = 256
GEN_NUM_BEAMS = 4

# ---------------- PDF / TEXT UTILS ----------------
def extract_text(pdf_path):
    """Extract text from all pages of a PDF."""
    texts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            t = page.extract_text()
            if t:
                texts.append(f"[page {i+1}]\n" + t)
    return "\n\n".join(texts)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split long text into overlapping chunks."""
    chunks, start, idx = [], 0, 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append({"id": idx, "text": piece})
            idx += 1
        start += chunk_size - overlap
    return chunks

# ---------------- EMBEDDINGS ----------------
def build_embeddings(chunks, embed_model_name=EMBED_MODEL):
    model = SentenceTransformer(embed_model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    return embeddings, model

def embed_query(query, embed_model):
    v = embed_model.encode([query], convert_to_numpy=True)
    return v[0] / (np.linalg.norm(v) + 1e-10)

def retrieve_top_k(query, chunks, embeddings, embed_model, k=TOP_K):
    qvec = embed_query(query, embed_model)
    sims = np.dot(embeddings, qvec)
    idxs = np.argsort(-sims)[:k]
    results = [chunks[i]["text"] for i in idxs]
    scores = [float(sims[i]) for i in idxs]
    return results, scores, idxs

# ---------------- IMPROVED DETERMINISTIC EXTRACTOR ----------------
def extract_direct_answer_v2(question, retrieved_chunks, retrieved_ids=None, retrieved_scores=None, max_sentences=6):
    """
    Domain-aware extractor:
      - prioritizes chunks by similarity
      - adds naval keyword hints (sail, bow, silhouette, recognition)
      - deduplicates sentences
    """
    q = question.lower().strip()
    domain_terms = [
        "sail", "sail shape", "sail placement", "bow", "bow profile",
        "silhouette", "recognition", "appearance", "profile"
    ]
    phrase_candidates = [q, q.rstrip("s")] + domain_terms
    keywords = list({w for w in re.findall(r"\w+", q) if len(w) > 2} | set(domain_terms))

    order = list(range(len(retrieved_chunks)))
    if retrieved_scores is not None:
        order = sorted(order, key=lambda i: -retrieved_scores[i])

    seen, out_parts = set(), []
    for i in order:
        text = retrieved_chunks[i]
        text_low = text.lower()
        if not any(ph in text_low for ph in phrase_candidates) and not any(kw in text_low for kw in keywords):
            continue

        sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_low = s_clean.lower()
            if any(ph in s_low for ph in phrase_candidates) or any(kw in s_low for kw in keywords):
                fp = re.sub(r"\s+", " ", s_low)[:200]
                if fp in seen:
                    continue
                seen.add(fp)
                cid = int(retrieved_ids[i]) if retrieved_ids is not None else None
                tag = f"[source: chunk {cid}]" if cid is not None else "[source]"
                out_parts.append(f"{tag} {s_clean}")
                if len(out_parts) >= max_sentences:
                    break
        if len(out_parts) >= max_sentences:
            break

    if not out_parts:
        return None
    return "\n\n".join(out_parts)

# ---------------- FLAN-T5 GENERATOR ----------------
print("Loading FLAN-T5 model (first run may take several minutes)...")
_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)

device = -1
try:
    import torch
    if torch.cuda.is_available():
        device = 0
except Exception:
    device = -1

_gen_pipeline = pipeline("text2text-generation", model=_model, tokenizer=_tokenizer, device=device)

def generate_answer(question, retrieved_chunks, retrieved_ids=None):
    """Use FLAN-T5 to generate an answer from retrieved context."""
    trimmed = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        txt = ch.strip()
        if len(txt) > 1000:
            txt = txt[:1000].rsplit(" ", 1)[0] + " ...[truncated]"
        header = f"[chunk {i}" + (f", id={int(retrieved_ids[i-1])}]" if retrieved_ids is not None else "]")
        trimmed.append(f"{header}\n{txt}")
    context = "\n\n---\n\n".join(trimmed)

    prompt = (
        "You are an assistant on aircraft, ship, and submarine identification.\n"
        "Answer using ONLY the CONTEXT. If the CONTEXT contains enough info, answer directly and cite chunk numbers. "
        "If not, say 'I don't know — not in the provided document.'\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer clearly."
    )

    out = _gen_pipeline(prompt, max_length=GEN_MAX_LENGTH, num_beams=GEN_NUM_BEAMS, do_sample=False, truncation=True)
    return out[0]["generated_text"].strip()

# ---------------- MAIN ----------------
def main():
    if not os.path.exists(PDF_PATH):
        print("ERROR: PDF not found at:", PDF_PATH)
        return

    print("Extracting PDF text...")
    text = extract_text(PDF_PATH)
    print(f"Extracted {len(text)} characters.")
    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")
    print("Building embeddings (please wait)...")
    embeddings, embed_model = build_embeddings(chunks)

    print("\nSystem ready. Type a question (or 'exit' to quit).")
    while True:
        q = input("\nQ: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        retrieved, scores, idxs = retrieve_top_k(q, chunks, embeddings, embed_model, k=TOP_K)
        print("\n[Retrieved chunk IDs]:", [int(i) for i in idxs])
        print("[Similarity scores]:", ["{:.3f}".format(s) for s in scores])

        # (optional) preview first 2 chunks
        for rank, (cid, txt) in enumerate(zip(idxs, retrieved), start=1):
            if rank > 2:  # show just first 2
                break
            print(f"\n--- Chunk {rank} (id={int(cid)}) preview ---")
            print(txt[:500].replace("\n", " ") + "\n")

        # 1️⃣ deterministic extraction first
        direct = extract_direct_answer_v2(q, retrieved, retrieved_ids=idxs, retrieved_scores=scores, max_sentences=6)
        if direct:
            print("\nA (extracted from document):")
            print(textwrap.fill(direct, 100))
        else:
            # 2️⃣ fallback to generator
            answer = generate_answer(q, retrieved, retrieved_ids=idxs)
            print("\nA:", textwrap.fill(answer, 100))

if __name__ == "__main__":
    main()
