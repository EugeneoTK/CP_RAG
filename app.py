import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

load_dotenv()

from rag import build_chain, ingest, load_vectorstore, CHROMA_DIR
from context.config import POSTCODE_DISTRICTS, TEST_CLINIC
from context.snapshot import build_snapshot

rag_chain = None

# Phase 1: in-memory snapshot cache. Cold builds take ~15-40 s (anonymous
# data.gov.sg rate limits), so serve repeats within the TTL from memory,
# keyed on coordinates rounded to ~3 dp (~111 m).
CONTEXT_CACHE_TTL_SECONDS = 15 * 60
_context_cache = {}  # "lat,lon" -> (built_at_monotonic, snapshot)


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


def _resolve_clinic_point(lat, lon, postcode, name):
    """(lat, lon) -> postcode district centroid -> TEST_CLINIC (mirrors `python -m context`)."""
    if (lat is None) != (lon is None):
        raise HTTPException(status_code=400, detail="lat and lon must be provided together")
    if lat is not None:
        if not (1.0 <= lat <= 1.6 and 103.0 <= lon <= 104.5):
            raise HTTPException(status_code=400, detail="coordinates outside Singapore")
        return lat, lon, name or ("Clinic @ %.4f,%.4f" % (lat, lon))
    if postcode:
        pc = postcode.strip().zfill(6)
        d = POSTCODE_DISTRICTS.get(pc[0])
        if not d:
            raise HTTPException(
                status_code=400,
                detail="unknown postcode first digit %r (use 1-8 or lat/lon)" % pc[0],
            )
        return d["lat"], d["lon"], name or ("Postcode %s (~%s)" % (pc, d["label"]))
    t = TEST_CLINIC
    return t["lat"], t["lon"], name or t["name"]


def _context_cache_lookup(key):
    for k in list(_context_cache):
        if time.monotonic() - _context_cache[k][0] >= CONTEXT_CACHE_TTL_SECONDS:
            del _context_cache[k]
    hit = _context_cache.get(key)
    if hit is None:
        return None, None
    return hit[1], int(time.monotonic() - hit[0])


@app.get("/api/context")
async def get_context(lat: float = None, lon: float = None,
                      postcode: str = None, name: str = None):
    """Live population-level context snapshot for a clinic point (key-free).

    Cold builds take ~15-40 s; repeats within CONTEXT_CACHE_TTL_SECONDS are
    served from the in-memory cache (see X-Snapshot-Cache header).
    """
    clat, clon, cname = _resolve_clinic_point(lat, lon, postcode, name)
    key = "%.3f,%.3f" % (clat, clon)
    snap, age = _context_cache_lookup(key)
    if snap is not None:
        return JSONResponse(snap, headers={
            "X-Snapshot-Cache": "hit", "X-Snapshot-Age": str(age)})
    # build_snapshot is blocking (urllib + polite sleeps) — run it off the
    # event loop so the rest of the API stays responsive.
    snap = await asyncio.get_running_loop().run_in_executor(
        None, build_snapshot, clat, clon, cname)
    _context_cache[key] = (time.monotonic(), snap)
    return JSONResponse(snap, headers={
        "X-Snapshot-Cache": "miss", "X-Snapshot-Age": "0"})
