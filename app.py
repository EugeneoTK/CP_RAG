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

from rag import build_chain, ingest, ingest_append, load_vectorstore, CHROMA_DIR
from context.config import POSTCODE_DISTRICTS, TEST_CLINIC
from context.snapshot import build_snapshot

rag_chain = None

# Phase 1: in-memory snapshot cache. Cold builds take ~15-40 s (anonymous
# data.gov.sg rate limits), so serve repeats within the TTL from memory,
# keyed on coordinates rounded to ~3 dp (~111 m).
CONTEXT_CACHE_TTL_SECONDS = 15 * 60
_context_cache = {}  # "lat,lon" -> (built_at_monotonic, snapshot)

# Phase 2: the RAG chain reads the SAME 15-min snapshot cache as GET
# /api/context, for a fixed clinic point (env-configurable, defaults to
# TEST_CLINIC). Cold builds take ~15-40 s, so the chat path NEVER blocks on
# a build: on a cache miss the prompt degrades to "context unavailable" and
# a background build refreshes the cache for the next chat (within the TTL).
CONTEXT_LAT = float(os.getenv("CONTEXT_LAT", TEST_CLINIC["lat"]))
CONTEXT_LON = float(os.getenv("CONTEXT_LON", TEST_CLINIC["lon"]))
CONTEXT_NAME = os.getenv("CONTEXT_NAME") or TEST_CLINIC["name"]
CONTEXT_KEY = "%.3f,%.3f" % (CONTEXT_LAT, CONTEXT_LON)
_context_builds = {}  # "lat,lon" -> in-flight background-build future
_app_loop = None  # captured in lifespan; get_context_snapshot may run in a worker thread


def _schedule_context_build(key, clat, clon, cname):
    async def _build():
        try:
            snap = await asyncio.get_running_loop().run_in_executor(
                None, build_snapshot, clat, clon, cname)
            _context_cache[key] = (time.monotonic(), snap)
        except Exception:
            pass  # a failed snapshot must never take the app down
        finally:
            _context_builds.pop(key, None)

    # run_coroutine_threadsafe: safe whether called from the event loop
    # (lifespan) or from a worker thread (the chat path).
    _context_builds[key] = asyncio.run_coroutine_threadsafe(_build(), _app_loop)


def get_context_snapshot():
    """Cached snapshot for the RAG clinic point, or None (never blocks)."""
    snap, _age = _context_cache_lookup(CONTEXT_KEY)
    if snap is not None:
        return snap
    if CONTEXT_KEY not in _context_builds:
        _schedule_context_build(CONTEXT_KEY, CONTEXT_LAT, CONTEXT_LON, CONTEXT_NAME)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, _app_loop
    _app_loop = asyncio.get_running_loop()
    if Path(CHROMA_DIR).exists():
        vectorstore = load_vectorstore()
        rag_chain = build_chain(vectorstore, context_provider=get_context_snapshot)
        # warm the context cache so the first chat has live context when it lands
        _schedule_context_build(CONTEXT_KEY, CONTEXT_LAT, CONTEXT_LON, CONTEXT_NAME)
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
    # rag_chain.invoke() is blocking (retrieval + synchronous LLM HTTP,
    # minutes on the free tier). Run it off the event loop so /,
    # /api/status and /api/context keep answering while a chat is in
    # flight — calling it directly froze the whole server (every other
    # request queued behind the LLM call with no response).
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, rag_chain.invoke, req.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail="LLM request failed: %s" % e)
    sources = list({doc.metadata.get("source", "") for doc in result["context"]})
    return ChatResponse(answer=result["answer"], sources=sources)


@app.post("/api/ingest")
async def run_ingest():
    global rag_chain
    # Full crawl + embed is long and blocking — keep the loop responsive.
    chunks = await asyncio.get_running_loop().run_in_executor(None, ingest)
    vectorstore = load_vectorstore()
    rag_chain = build_chain(vectorstore, context_provider=get_context_snapshot)
    return {"status": "ok", "chunks": chunks}


@app.post("/api/ingest/append")
async def run_ingest_append():
    """Append-only ingest: embed only chunks whose source URL is not in the
    store yet (never re-embeds the existing corpus). Rebuilds the chain so
    the retriever picks up the new chunks."""
    global rag_chain
    new_chunks, sources_added, sources_skipped = (
        await asyncio.get_running_loop().run_in_executor(None, ingest_append))
    vectorstore = load_vectorstore()
    rag_chain = build_chain(vectorstore, context_provider=get_context_snapshot)
    return {"status": "ok", "new_chunks": new_chunks,
            "sources_added": sources_added, "sources_skipped": sources_skipped}


@app.get("/api/status")
async def status():
    snap, age = _context_cache_lookup(CONTEXT_KEY)
    if snap is not None:
        ctx = "cached (%ss old)" % age
    elif CONTEXT_KEY in _context_builds:
        ctx = "building"
    else:
        ctx = "not-built"
    return {"ready": rag_chain is not None, "context": ctx}


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


if __name__ == "__main__":
    # `python app.py` used to import and exit silently — always give it a
    # server to run (was HANDOFF outstanding #6). PORT env overrides 5001.
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "5001")))
