import faulthandler
import json
import logging
import os
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware

faulthandler.enable()
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("documind")

BASE = Path(__file__).resolve().parent
UPLOAD_DIR, PERSIST_DIR, META_FILE = BASE / "uploads", BASE / "chroma_db", BASE / "kb_meta.json"
STATIC_DIR = BASE / "static"
COLLECTION = "documind"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None
_emb_lock = threading.Lock()
_build_lock = threading.Lock()

def get_embeddings():
    global _embeddings
    with _emb_lock:
        if _embeddings is None:
            log.info("Loading embedding model...")
            _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            log.info("Embedding model ready.")
    return _embeddings

def get_store() -> Chroma:
    return Chroma(collection_name=COLLECTION, persist_directory=str(PERSIST_DIR),
                  embedding_function=get_embeddings())

def drop_collection():
    try:
        get_store().delete_collection()
    except Exception as e:
        log.info("No previous collection to delete (%s)", e)

def read_files() -> List[str]:
    return json.loads(META_FILE.read_text()) if META_FILE.exists() else []

@asynccontextmanager
async def lifespan(_):
    def warm():
        try:
            get_embeddings()
        except Exception:
            log.exception("Could not load the embedding model")
    threading.Thread(target=warm, daemon=True).start()
    yield

app = FastAPI(title="DocuMind AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(StarletteHTTPException)
async def http_error(_, exc):
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

@app.exception_handler(RequestValidationError)
async def validation_error(_, exc):
    return JSONResponse({"error": "Invalid request."}, status_code=422)

@app.exception_handler(Exception)
async def unhandled(_, exc):
    log.exception("Unhandled error")
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class Settings(BaseModel):
    k: int = Field(4, ge=1, le=10)
    lambda_mult: float = Field(0.5, ge=0, le=1)
    temperature: float = Field(0.2, ge=0, le=1)
    memory: int = Field(3, ge=0, le=10)
    model: str = "openai/gpt-oss-20b"

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: List[Message] = []
    settings: Settings = Settings()

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Given chat history and follow-up question, rephrase into a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "Follow-up question: {question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer ONLY using the provided context."),
    MessagesPlaceholder("chat_history"),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
])

@app.get("/api/health")
def health():
    return {"ok": True, "model_loaded": _embeddings is not None}

@app.get("/api/status")
def status():
    files = read_files()
    return {"ready": bool(files), "files": files}

@app.post("/api/upload")
def upload(files: List[UploadFile] = File(...), chunk_size: int = Form(1000), chunk_overlap: int = Form(200)):
    if chunk_overlap >= chunk_size:
        raise HTTPException(400, "Chunk overlap must be smaller than chunk size.")
    if not _build_lock.acquire(blocking=False):
        raise HTTPException(409, "A build is already running.")
    try:
        UPLOAD_DIR.mkdir(exist_ok=True)
        docs, names = [], []
        for f in files:
            name = Path(f.filename or "document.pdf").name
            path = UPLOAD_DIR / name
            with open(path, "wb") as out:
                shutil.copyfileobj(f.file, out)
            pages = PyPDFLoader(str(path)).load()
            for p in pages:
                p.metadata["source"] = name
            docs += pages
            names.append(name)

        chunks = [c for c in RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)
            if c.page_content.strip()]
        if not chunks:
            raise HTTPException(400, "No text found.")

        emb = get_embeddings()
        drop_collection()
        Chroma.from_documents(chunks, emb, collection_name=COLLECTION, persist_directory=str(PERSIST_DIR))
        META_FILE.write_text(json.dumps(names))
        return {"files": names, "pages": len(docs), "chunks": len(chunks)}
    finally:
        _build_lock.release()

@app.post("/api/chat")
def chat(req: ChatRequest):
    if not read_files():
        raise HTTPException(400, "Build the knowledge base first.")
    s = req.settings
    turns = req.history[-(s.memory * 2):] if s.memory else []
    history = [HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content) for m in turns]

    llm = ChatGroq(model=s.model, temperature=s.temperature)
    retriever = get_store().as_retriever(
        search_type="mmr",
        search_kwargs={"k": s.k, "fetch_k": max(10, s.k), "lambda_mult": s.lambda_mult})

    standalone = req.question
    if history:
        standalone = llm.invoke(CONDENSE_PROMPT.invoke({"chat_history": history, "question": req.question})).content.strip()
    docs = retriever.invoke(standalone)
    answer = llm.invoke(ANSWER_PROMPT.invoke({
        "context": "\n\n".join(d.page_content for d in docs),
        "question": standalone, "chat_history": history})).content

    return {"answer": answer, "standalone": standalone, "sources": [
        {"source": d.metadata.get("source", "document"),
         "page": int(d.metadata.get("page", 0)) + 1, "text": d.page_content[:400]} for d in docs]}

@app.post("/api/reset")
def reset():
    drop_collection()
    META_FILE.unlink(missing_ok=True)
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    return {"ok": True}

if not STATIC_DIR.is_dir():
    raise RuntimeError(f"Missing frontend folder: {STATIC_DIR}")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)