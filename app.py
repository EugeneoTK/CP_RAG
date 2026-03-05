import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

from rag import build_chain, ingest, load_vectorstore, CHROMA_DIR

rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    if Path(CHROMA_DIR).exists():
        vectorstore = load_vectorstore()
        rag_chain = build_chain(vectorstore)
    yield


app = FastAPI(title="CP RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    url: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not ready. POST /api/ingest first.",
        )
    result = rag_chain.invoke(req.question)
    sources = list({doc.metadata.get("source", "") for doc in result["context"]})
    return ChatResponse(answer=result["answer"], sources=sources)


@app.post("/api/ingest")
async def run_ingest():
    global rag_chain
    chunks = ingest()
    vectorstore = load_vectorstore()
    rag_chain = build_chain(vectorstore)
    return {"status": "ok", "chunks": chunks}


@app.get("/api/status")
async def status():
    return {"ready": rag_chain is not None}
