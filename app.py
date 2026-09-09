import asyncio
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

load_dotenv()

from rag import (build_chain, ingest, ingest_append, load_vectorstore, CHROMA_DIR,
                 ingest_pdf, list_corpus, list_pdfs, MAX_PDF_BYTES, generate_brief)
from context.brief import format_brief_prompt
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

# --- Phase 6 brief cache (spec §7.2) ----------------------------------------------------
BRIEF_TTL_S = 900        # success cache, per clinic point
BRIEF_FAIL_TTL_S = 90    # negative cache: an LLM failure blocks re-dials
BRIEF_COOLDOWN_S = 60    # min interval between forced regenerations
_brief_cache = {}        # key -> (monotonic_ts, payload)
_brief_last_gen = {}     # key -> monotonic_ts of last successful generation
_brief_builds = {}       # key -> asyncio.Future (in-flight guard, D5)


def _brief_lookup(key):
    hit = _brief_cache.get(key)
    if not hit:
        return None
    ts, payload = hit
    ttl = BRIEF_FAIL_TTL_S if (payload and "error" in payload) else BRIEF_TTL_S
    if time.monotonic() - ts > ttl:
        _brief_cache.pop(key, None)
        return None
    return payload


def _build_brief(key, point, pname):
    """Sync build (runs in the thread-pool executor): snapshot + one LLM call."""
    snap = _context_cache_lookup(key)[0]
    if snap is None:
        snap = build_snapshot(point[0], point[1], pname)
    stats = list_corpus()
    stats = stats["totals"] if stats else {}
    system, user = format_brief_prompt(snap, stats)
    try:
        result = generate_brief(system, user)
    except Exception as e:
        _brief_cache[key] = (time.monotonic(), {"error": str(e)})  # 90-s sentinel
        raise
    _brief_cache[key] = (time.monotonic(), {
        "as_of": datetime.now().isoformat(timespec="seconds"),
        "snapshot_as_of": snap["meta"]["generated_at"],
        "brief": result,
    })
    _brief_last_gen[key] = time.monotonic()
    b = {"headline": result.get("headline", ""),
         "watch": result.get("watch", []),
         "outlook": result.get("outlook", "")}
    if result.get("parse_error"):
        b["parse_error"] = True
        b["raw"] = result.get("raw", "")
    return {
        "cached": False,
        "as_of": _brief_cache[key][1]["as_of"],
        "snapshot_as_of": snap["meta"]["generated_at"],
        "brief": b,
        "provenance_drops": result.get("provenance_drops", 0),
        "cache_ttl_s": BRIEF_TTL_S,
    }


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
    # PDF chunks carry doc_title — show the guideline name, not the raw URL
    sources = list({(doc.metadata.get("doc_title") or doc.metadata.get("source", ""))
                    for doc in result["context"]})
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


@app.post("/api/ingest-pdf")
async def run_ingest_pdf(file: UploadFile = File(...),
                         title: str = Form(""), source: str = Form("")):
    """Ingest one uploaded guideline PDF (parse + embed, dedupe by content
    hash). Parse/embed is blocking — executor per the no-freeze rule."""
    global rag_chain
    # Bounded read: stream 1 MB chunks and refuse at the cap, so an
    # oversized upload is rejected with at most ~26 MB buffered — not
    # the whole body in memory first (2026-09-09 debt fix).
    data = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        data += chunk
        if len(data) > MAX_PDF_BYTES:
            raise HTTPException(status_code=400, detail="PDF exceeds 25 MB cap")
    data = bytes(data)
    if data[:5] != b"%PDF-":
        raise HTTPException(status_code=400, detail="not a PDF file")
    t = title.strip() or (file.filename or "untitled.pdf")
    try:
        added, skipped, doc_hash = await asyncio.get_running_loop().run_in_executor(
            None, ingest_pdf, data, t, source.strip())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail="PDF ingest failed: %s" % e)
    if not skipped:
        # rebuild the chain so the retriever sees the new chunks
        vectorstore = load_vectorstore()
        rag_chain = build_chain(vectorstore, context_provider=get_context_snapshot)
    return {"status": "ok", "title": t, "chunks_added": added,
            "skipped": skipped, "doc_hash": doc_hash}


@app.get("/api/pdfs")
async def get_pdfs():
    """Ingested guideline PDFs (title, source, chunk count)."""
    pdfs = await asyncio.get_running_loop().run_in_executor(None, list_pdfs)
    return {"pdfs": pdfs}


@app.get("/api/library")
async def get_library():
    """Corpus inventory: guideline PDFs + web sources grouped by site (0 credits)."""
    if rag_chain is None:
        raise HTTPException(503, "Knowledge base not ready — run `venv/bin/python -m rag` first")
    try:
        data = await asyncio.get_running_loop().run_in_executor(None, list_corpus)
    except Exception as e:
        raise HTTPException(500, "Library scan failed: %s" % e)
    return data


@app.get("/api/brief")
async def brief_status(lat: float = None, lon: float = None,
                       postcode: str = None, name: str = None):
    """Free brief status/cached read — NEVER triggers a build or LLM call."""
    clat, clon, cname = _resolve_clinic_point(lat=lat, lon=lon,
                                              postcode=postcode, name=name)
    key = "%.3f,%.3f" % (clat, clon)
    snap, age = _context_cache_lookup(key)
    if key in _context_builds:
        snap_state = "building"
    elif snap:
        snap_state = "cached"
    else:
        snap_state = "none"
    payload = _brief_lookup(key)
    brief, last_error, drops, as_of = None, None, 0, None
    if payload:
        if payload.get("error"):
            last_error = payload["error"]
        else:
            brief = {"headline": payload["brief"].get("headline", ""),
                     "watch": payload["brief"].get("watch", []),
                     "outlook": payload["brief"].get("outlook", "")}
            if payload["brief"].get("parse_error"):
                brief["parse_error"] = True
                brief["raw"] = payload["brief"].get("raw", "")
            drops = payload["brief"].get("provenance_drops", 0)
            as_of = payload.get("as_of")
    fresh = bool(brief) and snap and payload.get("snapshot_as_of") == snap["meta"]["generated_at"]
    return {
        "status": "fresh" if fresh else ("stale" if (snap or snap_state == "building") else "none"),
        "brief": brief,
        "last_error": last_error,
        "provenance_drops": drops,
        "as_of": as_of,
        "snapshot": {"state": snap_state, "age_s": int(age) if age is not None else None},
        "clinic": cname,
        "cache_ttl_s": BRIEF_TTL_S,
        "force_cooldown_s": BRIEF_COOLDOWN_S,
    }


@app.post("/api/brief/generate")
async def brief_generate(lat: float = None, lon: float = None,
                         postcode: str = None, name: str = None,
                         force: int = 0):
    """The ONLY path that can call the LLM (spec §7.2 guard order)."""
    clat, clon, cname = _resolve_clinic_point(lat=lat, lon=lon,
                                              postcode=postcode, name=name)
    if rag_chain is None:
        raise HTTPException(503, "Knowledge base not ready — run `venv/bin/python -m rag` first")
    key = "%.3f,%.3f" % (clat, clon)
    now = time.monotonic()
    payload = _brief_lookup(key)
    snap, _age = _context_cache_lookup(key)
    fresh = bool(payload and not payload.get("error") and snap
                 and payload.get("snapshot_as_of") == snap["meta"]["generated_at"])
    if fresh and not force:
        out = {"headline": payload["brief"].get("headline", ""),
               "watch": payload["brief"].get("watch", []),
               "outlook": payload["brief"].get("outlook", "")}
        if payload["brief"].get("parse_error"):
            out["parse_error"] = True
            out["raw"] = payload["brief"].get("raw", "")
        return {"cached": True, "as_of": payload.get("as_of"),
                "snapshot_as_of": payload.get("snapshot_as_of"),
                "brief": out,
                "provenance_drops": payload["brief"].get("provenance_drops", 0),
                "cache_ttl_s": BRIEF_TTL_S}
    if payload and payload.get("error"):           # D7: live failure sentinel
        raise HTTPException(502,
                            "Brief generation failed: %s (cached error — "
                            "retry in up to %ss)" % (payload["error"], BRIEF_FAIL_TTL_S))
    if key in _brief_builds:                       # D5: await the shared build
        try:
            return await _brief_builds[key]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, "Brief generation failed: %s" % e)
    if force:                                      # D6: server-side cooldown
        last = _brief_last_gen.get(key)
        if last is not None and now - last < BRIEF_COOLDOWN_S:
            wait = int(BRIEF_COOLDOWN_S - (now - last))
            raise HTTPException(429, "Regenerate cooldown — try again in %ss" % wait,
                                headers={"Retry-After": str(wait)})
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    _brief_builds[key] = fut
    try:
        try:
            result = await loop.run_in_executor(None, _build_brief, key, (clat, clon), cname)
        except Exception as e:
            he = HTTPException(502, "Brief generation failed: %s" % e)
            fut.set_exception(he)
            raise he
        fut.set_result(result)
        return result
    finally:
        _brief_builds.pop(key, None)


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
