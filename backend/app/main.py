from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import shutil, os

from .rag_pipeline import (
    add_document, add_pdf, answer_query,
    chat_reply, chat_store
)

app = FastAPI()

# CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Option B: persistent PDFs on disk; auto-reingest on startup
DATA_DIR = Path.home() / "Desktop" / "powerai" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

@app.on_event("startup")
def load_docs():
    for pdf_path in DATA_DIR.glob("*.pdf"):
        try:
            add_pdf(str(pdf_path))
        except Exception as e:
            print(f"[startup] Failed ingest {pdf_path.name}: {e}")

@app.get("/")
def root():
    return {"message": "PowerAI is running"}

@app.get("/config")
def get_config():
    return {
        "use_openai": os.getenv("USE_OPENAI", "false").lower() == "true",
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }

@app.get("/sources")
def list_sources():
    files = [p.name for p in DATA_DIR.glob("*.pdf")]
    return {"sources": files}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    dest = DATA_DIR / file.filename
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    chunks_added = add_pdf(str(dest))
    return {"ok": True, "file": file.filename, "chunks_added": chunks_added}

# ---------- Query & Chat ----------

@app.post("/query")
def query_api(
    user_query: str,
    top_n: int = Query(3, ge=1, le=50),        # kept for compatibility
    strict: bool = Query(True),
    vec_min_score: float = Query(0.25, ge=0.0, le=1.0),
    source: str | None = Query(None, description="Restrict to a single PDF filename"),
):
    res = answer_query(
        user_query=user_query,
        top_n=top_n,
        strict=strict,
        vec_min_score=vec_min_score,
        source_filter=source,
    )
    return {"answer": res["answer"], "sources": res["sources"]}

class ChatIn(BaseModel):
    message: str
    session_id: str | None = None
    top_n: int | None = Field(3, ge=1, le=50)
    strict: bool = True
    vec_min_score: float = Field(0.25, ge=0.0, le=1.0)
    source: str | None = None

@app.post("/chat")
def chat_api(payload: ChatIn):
    return chat_reply(
        payload.session_id,
        payload.message,
        top_n=payload.top_n or 3,
        strict=payload.strict,
        vec_min_score=payload.vec_min_score,
        source_filter=payload.source,
    )

@app.post("/chat/reset")
def chat_reset(session_id: str):
    chat_store.clear(session_id)
    return {"ok": True}
