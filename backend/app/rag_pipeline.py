# backend/app/rag_pipeline.py

import os
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from transformers import pipeline
import uuid
from typing import List, Dict, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "300"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
client = OpenAI() if (USE_OPENAI and OpenAI is not None) else None

# Reduce semaphore warnings on macOS with torch
os.environ["PYTORCH_NO_SHARING"] = "1"

client = OpenAI()  # uses OPENAI_API_KEY from env

# 1) Models
# Fast embedding model (CPU-friendly)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
# Lightweight re-ranker to improve precision
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# NEW: small instruction-tuned generator for clear, merged answers
generator = pipeline("text2text-generation", model="google/flan-t5-base")  # CPU-friendly



# 2) Qdrant (in-memory; Option B autoload from /data on startup)
qdrant = QdrantClient(":memory:")
collection_name = "powerai_docs"

def build_context(passages, max_chars_per_passage=300, max_passages=6):
    """Make a short, labeled context the LLM can digest easily."""
    out = []
    for i, p in enumerate(passages[:max_passages], start=1):
        txt = (p.get("text") or "").strip()
        if not txt:
            continue
        if len(txt) > max_chars_per_passage:
            txt = txt[:max_chars_per_passage] + " ..."
        out.append(f"[{i}] {txt}")
    return "\n\n".join(out)



def ensure_collection():
    """Create the collection if it doesn't exist."""
    try:
        qdrant.get_collection(collection_name)
    except Exception:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )

ensure_collection()

# ---------- Ingestion helpers ----------

def chunk_text(text: str, size: int = 800, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    size: approx characters per chunk
    overlap: characters overlapping between consecutive chunks
    """
    chunks: List[str] = []
    i, n = 0, len(text)
    step = max(1, size - overlap)
    while i < n:
        end = min(i + size, n)
        chunk = text[i:end].strip()
        if chunk:
            # collapse odd whitespace
            chunks.append(" ".join(chunk.split()))
        i += step
    return chunks

def embed(text: str) -> List[float]:
    return embedder.encode(text).tolist()

def add_document(text: str, doc_id: int, meta: Optional[Dict] = None):
    """Store a single chunk in Qdrant with optional metadata."""
    ensure_collection()
    payload = {"text": text}
    if meta:
        payload.update(meta)
    qdrant.upsert(
        collection_name=collection_name,
        points=[models.PointStruct(id=doc_id, vector=embed(text), payload=payload)],
    )

def add_pdf(path: str) -> int:
    """
    Extract text from a PDF, chunk it with overlap, and store in Qdrant.
    Returns the number of chunks added.
    """
    ensure_collection()

    pdf_path = Path(path)
    filename = pdf_path.name

    # 1) Extract text
    reader = PdfReader(str(pdf_path))
    text = "".join((page.extract_text() or "") for page in reader.pages)

    # 2) Chunk
    chunks = chunk_text(text, size=800, overlap=200)
    if not chunks:
        return 0

    # 3) Unique base_id to avoid collisions (filename + timestamp)
    digest = hashlib.md5(f"{filename}-{time.time()}".encode("utf-8")).hexdigest()
    base_id = int(digest[:8], 16) * 1000

    # 4) Upsert
    added = 0
    for i, chunk in enumerate(chunks):
        add_document(
            chunk,
            doc_id=base_id + i,
            meta={"source": filename, "chunk_index": i},
        )
        added += 1

    return added

def build_context(passages: List[Dict], max_chars_per_passage=300, max_passages=6) -> str:
    """Make a short, labeled context that stays within model limits."""
    out = []
    for i, p in enumerate(passages[:max_passages], start=1):
        txt = (p.get("text") or "").strip()
        if not txt:
            continue
        if len(txt) > max_chars_per_passage:
            txt = txt[:max_chars_per_passage] + " ..."
        out.append(f"[{i}] {txt}")
    return "\n\n".join(out)

def stitch_adjacent(ranked_sources: List[Dict], window: int = 1) -> List[Dict]:
    """
    Merge passages that are adjacent in the same source (e.g., chunk_index neighbors).
    Keeps the first item’s metadata; concatenates text.
    """
    if not ranked_sources:
        return []

    merged = []
    group = [ranked_sources[0]]
    for cur in ranked_sources[1:]:
        prev = group[-1]
        same_doc = cur.get("source") == prev.get("source")
        close = abs(cur.get("chunk_index", 0) - prev.get("chunk_index", 0)) <= window
        if same_doc and close:
            # extend the text of the last item in the group
            group[-1]["text"] = (group[-1]["text"] + " " + cur.get("text", "")).strip()
        else:
            merged.append(group[-1])
            group = [cur]
    merged.append(group[-1])
    return merged


def rewrite_query(user_query: str, history: List[Dict[str, str]]) -> str:
    # heuristic fallback if OpenAI is off or no history
    if not (USE_OPENAI and client is not None) or not history:
        prev_q = ""
        for m in reversed(history):
            if m["role"] == "user":
                prev_q = m["content"]
                break
        return (prev_q + " " + user_query).strip() if prev_q else user_query

    # LLM-based rewrite
    convo = []
    for m in history[-8:]:
        role = "user" if m["role"] == "user" else "assistant"
        convo.append({"role": role, "content": m["content"]})
    convo.append({
        "role": "system",
        "content": "Rewrite the user's latest question as a standalone search query. Output ONLY the rewritten query."
    })
    convo.append({"role": "user", "content": user_query})

    try:
        chat = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=convo,
            temperature=0.0,
            max_tokens=100,
        )
        return chat.choices[0].message.content.strip()
    except Exception:
        return user_query


def local_concise_summary(user_query: str, sources: List[Dict]) -> str:
    if not sources:
        return "No matches found."
    bullets = []
    for s in sources[:5]:
        txt = (s.get("text") or "").strip()
        if txt:
            bullets.append("• " + (txt[:220] + " ..." if len(txt) > 220 else txt))
    return f"Summary for: {user_query}\n" + "\n".join(bullets)


# ---------- Query (vector search + re-ranking) ----------

def answer_query(
    user_query: str,
    top_n: int = 3,
    strict: bool = True,
    vec_min_score: float = 0.25,
    source_filter: str | None = None,
):
    """
    Dual-LLM Chain:
    1) Local extraction (cheap, ensures completeness).
    2) OpenAI refinement (polish).
    Returns: Clean numbered list + unique doc citations.
    """
    ensure_collection()

    # --- Step 1: Retrieve chunks ---
    q_filter = None
    if source_filter:
        q_filter = models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=source_filter))]
        )

    pool_size = 120
    candidates = qdrant.search(
        collection_name=collection_name,
        query_vector=embed(user_query),
        limit=pool_size,
        query_filter=q_filter,
    )
    if not candidates:
        return {"answer": "I couldn’t find that in your uploaded documents.", "sources": []}

    strong = [c for c in candidates if float(getattr(c, "score", 0.0)) >= vec_min_score]
    if strict and not strong:
        return {"answer": "I couldn’t find that in your uploaded documents.", "sources": []}
    pool = strong if (strict and strong) else candidates

    try:
        texts = [c.payload.get("text", "") for c in pool]
        pairs = [(user_query, t) for t in texts]
        scores = reranker.predict(pairs)
        ranked_pairs = list(zip(pool, scores))
        ranked_pairs.sort(key=lambda x: float(x[1]), reverse=True)
    except Exception:
        ranked_pairs = [(c, 0.0) for c in pool]

    prelim = []
    for (hit, score) in ranked_pairs[:60]:
        p = hit.payload or {}
        prelim.append({
            "text": p.get("text", ""),
            "source": p.get("source", "unknown"),
        })

    stitched = stitch_adjacent(prelim, window=2) or prelim
    stitched = stitched[:30]
    if strict and not stitched:
        return {"answer": "I couldn’t find that in your uploaded documents.", "sources": []}

    broader = [{"text": s["text"]} for s in stitched]
    context = build_context(broader, max_chars_per_passage=420, max_passages=18)

    # --- Step 2: Local extractor (ensures all bullets captured) ---
    import re
    joined = "\n".join(s["text"] for s in stitched if s.get("text"))
    lines = [ln.strip() for ln in joined.splitlines() if ln.strip()]
    bullet_like = []
    for ln in lines:
        if re.match(r"^(\d+[\.\)]|[\-\u2022•●])\s", ln, re.U) or any(
            k in ln.lower() for k in ["incentive", "subsid", "grant", "rebate", "support"]
        ):
            bullet_like.append(ln)
    seen, raw_list = set(), []
    for ln in bullet_like:
        key = re.sub(r"\s+", " ", ln.lower())
        if key not in seen:
            seen.add(key)
            raw_list.append(ln)

    if not raw_list:
        return {"answer": "I couldn’t extract a complete list from the uploaded documents.", "sources": stitched}

    raw_text = "\n".join(f"- {c}" for c in raw_list)

    # --- Step 3: OpenAI refinement (if available) ---
    if USE_OPENAI and client is not None:
        try:
            sys_msg = (
                "You are a regulatory assistant. Refine the provided raw extracted list. "
                "Keep ALL items, merge duplicates, and return a clean numbered list (1., 2., 3., …). "
                "Preserve exact figures, percentages, currency, limits, and conditions. Do not omit."
            )
            user_msg = f"Question: {user_query}\n\nRaw extracted list:\n{raw_text}\n\nRefine into a clean numbered list."
            chat = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": user_msg}],
                temperature=0.0,
                max_tokens=max(OPENAI_MAX_TOKENS, 600),
            )
            answer_text = chat.choices[0].message.content.strip()
        except Exception as e:
            print("[warn] OpenAI refinement failed:", e)
            answer_text = raw_text
    else:
        answer_text = raw_text

    # --- Step 4: Deduplicate citations by doc name ---
    seen_sources = set()
    unique_sources = []
    for s in stitched:
        fname = s.get("source", "unknown")
        if fname not in seen_sources:
            seen_sources.add(fname)
            unique_sources.append({"source": fname})

    return {"answer": answer_text, "sources": unique_sources}


def extract_all_items(
    user_query: str,
    vec_min_score: float = 0.25,     # slightly lower to widen recall
    source_filter: str | None = None,
):
    """
    Pull a large pool, filter by similarity, stitch neighbors, and extract ALL items.
    Uses OpenAI when enabled; otherwise regex-based local fallback.
    Returns {"answer": str, "sources": List[Dict]} with a numbered list if possible.
    """
    ensure_collection()

    # Optional single-source filter
    q_filter = None
    if source_filter:
        q_filter = models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=source_filter))]
        )

    # 1) Widen recall aggressively
    pool_size = 120
    candidates = qdrant.search(
        collection_name=collection_name,
        query_vector=embed(user_query),
        limit=pool_size,
        query_filter=q_filter,
    )
    if not candidates:
        return {"answer": "I couldn’t find that in your uploaded documents.", "sources": []}

    # 2) Score cutoff to keep only relevant text
    strong = [c for c in candidates if float(getattr(c, "score", 0.0)) >= vec_min_score]
    if not strong:
        return {"answer": "I couldn’t find that in your uploaded documents.", "sources": []}

    # 3) Re-rank and take a large slice for stitching
    try:
        texts = [c.payload.get("text", "") for c in strong]
        pairs = [(user_query, t) for t in texts]
        scores = reranker.predict(pairs)
        ranked_pairs = list(zip(strong, scores))
        ranked_pairs.sort(key=lambda x: float(x[1]), reverse=True)
    except Exception:
        ranked_pairs = [(c, 0.0) for c in strong]

    prelim = []
    for (hit, score) in ranked_pairs[:60]:
        p = hit.payload or {}
        prelim.append({
            "text": p.get("text", ""),
            "source": p.get("source", "unknown"),
            "chunk_index": p.get("chunk_index", -1),
            "score": float(getattr(hit, "score", 0.0)),
            "rerank": float(score),
        })

    stitched = stitch_adjacent(prelim, window=2)  # merge across wider neighborhood
    stitched = stitched[:30]  # cap to keep prompt manageable

    if not stitched:
        return {"answer": "I couldn’t find that in your uploaded documents.", "sources": []}

    # Build a bigger context (more passages, slightly shorter each)
    broader = [{"text": s["text"]} for s in stitched]
    context = build_context(broader, max_chars_per_passage=420, max_passages=18)

    # 4) Extraction prompt
    sys_msg = (
        "You extract COMPLETE itemized lists ONLY from the provided context. "
        "Preserve exact numbers, limits, dates, thresholds, and conditions. Do NOT omit items. "
        "If the context seems incomplete, say so explicitly."
    )
    user_msg = (
        f"Task: Extract ALL incentives relevant to the question.\n"
        f"Question: {user_query}\n\n"
        f"Context (numbered snippets):\n{context}\n\n"
        "Return a numbered list (1., 2., 3., …). Do not invent items. "
        "If some items appear repeated across snippets, merge them once."
    )

    # 5) OpenAI path or local fallback
    if USE_OPENAI and client is not None:
        try:
            chat = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                temperature=0.0,
                max_tokens=max(OPENAI_MAX_TOKENS, 600),  # allow longer answers
            )
            ans = chat.choices[0].message.content.strip()
            return {"answer": ans, "sources": stitched}
        except Exception:
            pass  # fall through to local fallback

    # Local fallback: regex-pull likely bullet/numbered lines
    import re
    joined = "\n".join(s["text"] for s in stitched)
    lines = [ln.strip() for ln in joined.splitlines() if ln.strip()]
    bullet_like = []
    for ln in lines:
        if re.match(r"^(\d+[\.\)]|[\-\u2022•●])\s", ln, re.U) or ("incentive" in ln.lower()) or ("subsid" in ln.lower()):
            bullet_like.append(ln)
    # de-dup loosely
    seen = set()
    cleaned = []
    for ln in bullet_like:
        key = re.sub(r"\s+", " ", ln.lower())
        if key not in seen:
            seen.add(key)
            cleaned.append(ln)
    if not cleaned:
        return {"answer": "I couldn’t extract a complete list from the uploaded documents.", "sources": stitched}
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(cleaned))
    return {"answer": numbered, "sources": stitched}



class ChatStore:
    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def get(self, sid: str) -> List[Dict[str, str]]:
        return self.sessions.setdefault(sid, [])

    def append(self, sid: str, role: str, content: str):
        self.get(sid).append({"role": role, "content": content})

    def clear(self, sid: str):
        self.sessions.pop(sid, None)

chat_store = ChatStore()


def chat_reply(
    session_id: str | None,
    user_message: str,
    top_n: int = 3,
    strict: bool = True,
    vec_min_score: float = 0.25,
    source_filter: str | None = None,
):
    sid = session_id or uuid.uuid4().hex[:12]
    history = chat_store.get(sid)

    chat_store.append(sid, "user", user_message)

    standalone_query = rewrite_query(user_message, history)

    # just call answer_query without mode
    result = answer_query(
        standalone_query,
        top_n=top_n,
        strict=strict,
        vec_min_score=vec_min_score,
        source_filter=source_filter,
    )

    chat_store.append(sid, "assistant", result["answer"])

    return {"session_id": sid, "answer": result["answer"], "sources": result["sources"]}
