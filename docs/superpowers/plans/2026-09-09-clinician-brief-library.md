# Clinician Brief Dashboard + Library — Implementation Plan (Phase 6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default **Brief** dashboard (free KPI tiles + one-button,
cache-guarded LLM brief) and a **Library** screen (corpus inventory + PDF
upload) to CP_RAG, behind three hash-routed tabs.

**Architecture:** Free deterministic KPI tiles render client-side from the
existing snapshot JSON (`GET /api/context`). The inferred brief is a two-phase
API: free `GET /api/brief` (status/cached read) + paid
`POST /api/brief/generate` (the only LLM path; in-flight guard + 60 s cooldown
+ 90 s failure sentinel). The Protocol spotlight renders 100% from structured
`snapshot.protocol_links.active` — the brief LLM never sees protocol names.
Library = read-only Chroma metadata scan (`GET /api/library`) rendered in a
new tab. UI stays in the single existing `static/index.html`.

**Tech Stack:** Python 3.9, FastAPI, LangChain 0.3.x (`ChatOpenAI` via
OpenRouter), Chroma persisted store, vanilla JS/CSS in `static/index.html`.

**Spec:** `docs/superpowers/specs/2026-09-09-clinician-brief-library-design.md`
(read it first — this plan argues from it; §4 decision IDs D1–D11 refer to it)

## Global Constraints (binding every task — copied from spec §2/§12)

- **Python 3.9 only.** No new dependencies, no new libraries, stdlib additions
  only in `context/`.
- **No test suite** (repo convention): verification = server boot +
  `GET /api/status` + `node --check` on the page script + deliberate curls +
  ad-hoc scratchpad scripts (never permanent test files).
- **API credits are a budget** (`429 credit_balance_exhausted` history): no
  loop/retry against the LLM; every task states its credit cost in its gate;
  the only paid verification call in the whole plan is the one in Task 2.
- `chroma_db/` is **read-only** for this feature (no re-embed, no re-ingest).
- Blocking work only inside thread-pool executors in `async def` endpoints
  (never blocking the event loop directly).
- **Provenance rule:** live signals are observations, never protocol content;
  URA written permission ≠ opened facility; `basis` tags stay visible.
- Scope guards: no auth, no Docker, no streaming, no framework/build step in
  the UI, no changes to the RAG chat chain/prompt/retrieval.
- **One focused commit per task.** After each task: server boots,
  `GET /api/status` → `ready: true`, `GET /` → 200.

## File structure

| File | Change | Responsibility |
|------|--------|----------------|
| `context/brief.py` | **Create** | Pure prompt builder: `format_brief_prompt(snapshot, corpus_stats) -> (system, user)`. Stdlib only, no I/O. |
| `rag.py` | Modify | Add `list_corpus()`, `_derive_title(url)`, `generate_brief(system, user)`, `_parse_brief(text)`, `_apply_provenance_guard(watch)`, `_parse_and_guard(text)`. |
| `app.py` | Modify | Add brief cache/guard state, `GET /api/brief`, `POST /api/brief/generate`, `GET /api/library`, sync `_build_brief()`. |
| `static/index.html` | Modify | Nav row, `#view-brief` / `#view-chat` / `#view-library` wrappers, KPI tiles + brief renderer, Library renderer, upload-bar relocation. |
| `docs/signal-research.md`, `CLAUDE.md`, `HANDOFF.md`, SDD ledger | Modify (Task 5) | Phase 6 documentation. |

**Interfaces produced/consumed across tasks:**
- Task 1 → `list_corpus() -> dict` with keys `totals{chunks, guideline_pdfs,
  protocol_pages, public_guidance_pages}`, `pdfs[{title, source, chunks,
  doc_hash}]`, `web_sources{site: [{url, chunks, derived_title}]}`; `None` if
  no vectorstore. Consumed by Task 2 (stats) and Task 4 (UI data).
- Task 2 → `format_brief_prompt(snapshot, corpus_stats) -> (system, user)`;
  `generate_brief(system, user) -> dict` with
  `headline/watch[{finding, why_it_matters, action, source}]/outlook/
  provenance_drops` or `parse_error` shape. Endpoints per spec §7. Consumed by
  Task 3 (UI) and Task 5 (docs).
- Task 3 → JS `switchTab()`, `applyTab(tab)`, `renderBriefTiles(snap)`,
  `renderBrief(payload)`, `loadBrief()`, `generateBrief(force)`.
- Task 4 → JS `loadLibrary()`, `renderLibrary(data)`.

---

### Task 1: `list_corpus()` + `GET /api/library` (0 credits)

**Files:**
- Modify: `rag.py` (append near the existing `list_pdfs()`)
- Modify: `app.py` (new endpoint; reuse `run_in_executor` + 503 pattern of `/api/pdfs`)

**Interfaces:**
- Consumes: existing `vectorstore` global, `list_pdfs()` scan pattern.
- Produces: `list_corpus() -> dict` (shape in the File structure table).

- [ ] **Step 1: Read the existing scan to mirror it exactly**

Read `list_pdfs()` in `rag.py` (the `load_vectorstore()` +
`vectorstore.get(include=["metadatas"])` loop that groups by `doc_hash`) and
the `GET /api/pdfs` handler in `app.py` (`asyncio.get_running_loop().
run_in_executor(None, list_pdfs)`). Note: `app.py` has **no module-level
`vectorstore` global** — it is a local in `lifespan()`/handlers; the
readiness signal is the module global `rag_chain` (set only when
`chroma_db/` exists at boot), so all "not ready" 503 checks use
`if rag_chain is None` (the same signal `/api/status.ready` uses).
Confirm the chunk metadata keys in use: `source`, `source_site`,
`doc_title`, `doc_hash`.

- [ ] **Step 2: Add `_derive_title()` + `list_corpus()` to `rag.py`**

```python
def _derive_title(url):
    """Display title from a URL (cosmetic only — the raw URL is always shown
    beside it in the Library; web-crawled chunks store no page title).

    Spec: last URL path segment, %20/- → spaces, title-cased; a bare web file
    name (e.g. diabetes.html) drops the extension.
    """
    path = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    segs = [s for s in path.split("/") if s]
    seg = segs[-1] if segs else "index"
    base, dot, ext = seg.rpartition(".")
    if base and dot and ext.lower() in ("html", "htm", "php", "asp", "aspx", "jsp"):
        seg = base
    return seg.replace("%20", " ").replace("-", " ").title() or "index"


def _site_of(url):
    """Site label for legacy chunks that predate the source_site field
    (the original primarycarepages.sg crawl stored no site metadata)."""
    if not url.startswith("http"):
        return ""
    host = url.split("/", 3)[2]
    return host[4:] if host.startswith("www.") else host


def list_corpus():
    """Full corpus inventory (read-only metadata scan, 0 API credits).

    PDFs grouped by doc_hash (mirrors list_pdfs()); web sources grouped by
    source_site — legacy no-site chunks fall back to the URL host
    (`_site_of`). `protocol_pages` counts unique URLs under the
    `/care-protocols/` path (the depth-2 crawl also captured site nav pages
    and a few link artifacts; they stay in `web_sources` for the raw
    inventory). Derived titles are cosmetic only — the raw URL is always
    shown beside them by the UI. The endpoint 503s on `rag_chain is None`
    before calling this (see Step 1 note).
    """
    vs = load_vectorstore()
    rows = vs.get(include=["metadatas"]).get("metadatas") or []
    pdfs, web = {}, {}
    for m in rows:
        m = m or {}
        src = m.get("source") or ""
        site = m.get("source_site") or _site_of(src)
        if m.get("doc_hash"):
            entry = pdfs.setdefault(m["doc_hash"], {
                "title": m.get("doc_title") or "?", "source": src,
                "chunks": 0, "doc_hash": m["doc_hash"]})
            entry["chunks"] += 1
        elif site and src:
            entry = web.setdefault(site, {}).setdefault(src, {
                "url": src, "chunks": 0, "derived_title": _derive_title(src)})
            entry["chunks"] += 1
    protocol_pages = sum(1 for urls in web.values()
                         for e in urls.values() if "/care-protocols/" in e["url"])
    return {
        "totals": {
            "chunks": len(rows),
            "guideline_pdfs": len(pdfs),
            "protocol_pages": protocol_pages,
            "public_guidance_pages": sum(1 for _ in web.get("moh.gov.sg", {})),
        },
        "pdfs": sorted(pdfs.values(), key=lambda p: p["title"].lower()),
        "web_sources": {site: sorted(urls.values(), key=lambda e: e["url"])
                        for site, urls in web.items()},
    }
```

- [ ] **Step 3: Add `GET /api/library` to `app.py`**

```python
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
```

(Executor idiom matches `/api/pdfs`; `rag_chain is None` is the 503 signal —
there is no module-level `vectorstore` in `app.py`.)

- [ ] **Step 4: Boot + 0-credit pre-check (the review's URL-shape gate)**

```bash
venv/bin/python -m app &   # or the repo's usual boot command
curl -s localhost:5001/api/status
curl -s localhost:5001/api/library | venv/bin/python -c "import json,sys; d=json.load(sys.stdin); print(d['totals']); print(len(d['web_sources'].get('primarycarepages.sg', [])), 'protocol URLs'); [print(e['url'], e['chunks']) for e in d['web_sources'].get('primarycarepages.sg', [])[:30]]"
```

Expected: `totals.guideline_pdfs == 96`; PDF chunks 2,975 (ledger);
`protocol_pages` = 28 (observed — 22 real protocol pages + crawl artifacts:
`https&`, `mailto&`, `_vti_bin/spsdisco.aspx`, a trailing-slash duplicate,
a zero-width-char link; the original Phase-0 set was 17 — the depth-2 crawl
pulled preventive/administrative protocols too). **Record in the SDD
ledger** the actual counts (77 unique primarycarepages.sg URLs / 3,785
chunks legacy no-`source_site`; 49 non-protocol site pages; 1 MOH page /
4 chunks). This observation fixes Task 4's rendering: 28 > 25 → the
protocol section uses the collapsed site-level card (spec §9), with an
"other site pages" count line.

- [ ] **Step 5: Standing checks + commit**

`GET /` → 200; `GET /api/status` → ready; `node --check` (unchanged but
confirm page still loads). Then:

```bash
git add rag.py app.py
git commit -m "feat: GET /api/library corpus inventory (PDFs + web sources by site)"
```

**Gate:** all curls 0 credits; server boots; `list_corpus()` totals match the
ledger's known corpus (96 PDFs / 2,975 chunks; protocol pages counted by
`/care-protocols/` URL prefix — 28 observed; 1 MOH page); ledger records the
URL count and the no-`source_site` legacy finding.

### Task 2: brief backend — `context/brief.py`, `rag.generate_brief()`, the two brief endpoints

**Credit budget for this task: exactly ONE paid LLM call** (Step 8) + one
expected 429 (0 credits). Steps 1–7 are all 0 credits.

**Files:**
- Create: `context/brief.py`
- Modify: `rag.py` (append brief functions; import `PROTOCOLS` from
  `context.linkage`)
- Modify: `app.py` (brief cache state + `GET /api/brief` +
  `POST /api/brief/generate` + `_build_brief`)

**Interfaces:**
- Consumes: Task 1 `list_corpus()`; existing `_resolve_clinic_point()`,
  `_context_cache_lookup(key)`, `_context_builds`, `build_snapshot()`.
- Produces: endpoint contracts per spec §7.1/§7.2; `generate_brief()`,
  `_parse_and_guard()` for Task 3's UI and Task 5's docs.

- [ ] **Step 1: Create `context/brief.py` (pure, stdlib-only)**

```python
"""Brief prompt builder (Phase 6) — stdlib only, pure functions.

Builds the (system, user) prompt pair for the one-shot LLM brief. The user
payload is a *projection* of the snapshot; `protocol_links` is deliberately
excluded — the brief LLM must never see protocol names or basis notes. The
Protocol spotlight UI section is rendered directly from structured
snapshot.protocol_links.active, so basis-tag provenance holds structurally,
not by model compliance. Spec:
docs/superpowers/specs/2026-09-09-clinician-brief-library-design.md §8.1.
"""
import json

_SYSTEM = (
    "You write a one-screen daily clinical briefing for a Singapore "
    "primary-care clinic from live population-level data.\n"
    "Rules:\n"
    "1. The JSON provided is the ONLY data source. Never invent values, "
    "places, or statistics. If a block is absent, that source is unavailable "
    "this cycle.\n"
    "2. Live signals are observations about the area. NEVER attribute them to "
    "clinic protocols, care protocols, or clinical guidelines. You are NOT "
    "given protocol names or guideline content, and you must not use the "
    "words protocol or guideline (in any form) anywhere in your output. "
    "Protocol relevance is rendered separately from structured data.\n"
    "3. A URA written permission is NOT an opened facility — never present "
    "one as an existing service.\n"
    "4. Reflect data_gaps: name the affected area in outlook or the relevant "
    "watch item (\"unavailable this cycle\").\n"
    "5. If nothing is elevated, say so plainly — a calm week is a valid "
    "headline. Do not manufacture urgency.\n"
    "6. Output STRICT JSON only — no markdown fences, no commentary: "
    "{\"headline\": string (max 25 words), \"watch\": [{\"finding\": max 25 "
    "words, \"why_it_matters\": max 30 words, \"action\": max 30 words, "
    "\"source\": one of NEA | data.gov.sg | CDA WIDB | URA | derived}] (max 6 "
    "items, most clinically important first), \"outlook\": string (max 40 "
    "words)}."
)


def _clip_dicts(lst, n):
    return [x for x in (lst or [])[:n] if isinstance(x, dict)]


def _project(snapshot, corpus_stats):
    """Compact projection of the snapshot for the prompt (spec §8.1)."""
    meta = snapshot.get("meta") or {}
    clinic = meta.get("clinic") or {}
    air = snapshot.get("air_quality") or {}
    psi, pm = air.get("psi") or {}, air.get("pm25") or {}
    wx = snapshot.get("weather") or {}
    today = wx.get("today") or {}
    den = snapshot.get("dengue") or {}
    dw = snapshot.get("disease_week") or {}
    d_den, d_flu = dw.get("dengue") or {}, dw.get("influenza") or {}
    cc = snapshot.get("catchment_change") or {}
    ns = snapshot.get("nearest_services") or {}
    payload = {
        "clinic": {k: clinic[k] for k in
                   ("name", "lat", "lon", "nea_region_approx", "town")
                   if clinic.get(k) is not None},
        "generated_at": meta.get("generated_at"),
        "air": {"psi_peak": psi.get("peak"),
                "psi_peak_region": psi.get("peak_region"),
                "pm25_national": pm.get("national")},
        "weather": {"today_date": today.get("date"),
                    "today_high_c": today.get("high_c"),
                    "today_low_c": today.get("low_c"),
                    "flood_alerts": wx.get("flood")},
        "dengue": {"clusters_active_total": den.get("clusters_active_total"),
                   "nearby_clusters": _clip_dicts(den.get("nearby_clusters"), 3),
                   "in_high_aedes_area": den.get("in_high_aedes_area")},
        "widb": {"epi_week": dw.get("epi_week"),
                 "date_range": dw.get("date_range"),
                 "notable": dw.get("notable"),
                 "dengue_week": d_den.get("week"),
                 "dengue_median_5yr": d_den.get("median_5yr"),
                 "flu_ili_positivity_pct": d_flu.get("ili_positivity_pct")},
        "planning": {"window": cc.get("window"),
                     "healthcare_decisions_90d_count":
                         cc.get("healthcare_decisions_90d_count"),
                     "recent_decisions":
                         _clip_dicts(cc.get("healthcare_decisions_90d"), 3)},
        "polyclinics": _clip_dicts(ns.get("pyclinics"), 3),
        "data_gaps": snapshot.get("data_gaps") or [],
        "corpus": corpus_stats or {},
    }
    # A source that failed (gap) simply omits its block — absent = unavailable.
    return {k: v for k, v in payload.items()
            if not (isinstance(v, dict) and not any(x is not None for x in v.values()))}


def format_brief_prompt(snapshot, corpus_stats):
    """Return (system, user) strings for the one-shot brief LLM call."""
    return _SYSTEM, json.dumps(_project(snapshot, corpus_stats), ensure_ascii=False)
```

- [ ] **Step 2: Add `generate_brief()` + helpers to `rag.py`**

Append after the chat-chain code. At the top of `rag.py` add
`from context.linkage import PROTOCOLS` (stdlib-only module — safe):

```python
_BRIEF_FORBIDDEN = ["protocol", "guideline"] + [p.lower() for p in PROTOCOLS]


def _apply_provenance_guard(watch):
    """Drop watch items that name protocols/guidelines (spec §8.2, D9).

    Substring matching may over-suppress — deliberate: a false positive
    costs one UI line, a false negative violates the provenance rule.
    """
    kept, drops = [], 0
    for it in watch:
        blob = (it["finding"] + " " + it["why_it_matters"] + " " + it["action"]).lower()
        if any(p in blob for p in _BRIEF_FORBIDDEN):
            drops += 1
        else:
            kept.append(it)
    return kept, drops


def _parse_brief(text):
    t = (text or "").strip()
    if t.startswith("```"):
        body = t[3:]
        if body.lstrip().lower().startswith("json"):
            body = body.lstrip()[4:]
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        t = body.strip()
    data = json.loads(t)
    if not isinstance(data, dict):
        raise ValueError("brief is not a JSON object")
    items = []
    watch = data.get("watch")
    for it in (watch if isinstance(watch, list) else [])[:6]:
        if not isinstance(it, dict):
            continue
        items.append({"finding": str(it.get("finding") or ""),
                      "why_it_matters": str(it.get("why_it_matters") or ""),
                      "action": str(it.get("action") or ""),
                      "source": str(it.get("source") or "derived")})
    return {"headline": str(data.get("headline") or ""),
            "watch": items, "outlook": str(data.get("outlook") or "")}


def _parse_and_guard(text):
    """Parse + coerce + provenance-guard an LLM brief response (no I/O)."""
    try:
        brief = _parse_brief(text)
    except ValueError:
        return {"parse_error": True, "raw": (text or "")[:2000],
                "headline": "", "watch": [], "outlook": "",
                "provenance_drops": 0}
    watch, drops = _apply_provenance_guard(brief["watch"])
    brief["watch"] = watch
    brief["provenance_drops"] = drops
    return brief


def generate_brief(system, user):
    """One-shot brief from the snapshot projection (spec §8.2/§8.3).

    Cost: ONE LLM call — ≤2 provider round-trips (max_retries=1), 0
    embedding calls, 0 retrieval.
    """
    llm = ChatOpenAI(model_name=CHAT_MODEL, openai_api_base=OPENAI_BASE_URL,
                     temperature=0, timeout=300, max_retries=1)
    resp = llm.invoke([("system", system), ("human", user)])
    text = resp if isinstance(resp, str) else str(getattr(resp, "content", resp))
    return _parse_and_guard(text)
```

- [ ] **Step 3: Add brief state + `_build_brief()` to `app.py`**

Next to the existing `_context_cache` / `_context_builds` declarations:

```python
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
```

Imports needed in `app.py` (check what already exists): `from context.brief import format_brief_prompt`, `from rag import generate_brief, list_corpus` (extend the existing `from rag import ...` line), `from context.snapshot import build_snapshot` (likely already imported).

- [ ] **Step 4: Add the two brief endpoints to `app.py`**

```python
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
```

```python
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
```

Note: `HTTPException` is `fastapi.HTTPException` (already imported); confirm
`asyncio` is imported in `app.py` (the existing background-refresh code may
already import it).

- [ ] **Step 5: 0-credit dry run #1 — prompt inspection (no LLM)**

Scratchpad script (run then delete; do NOT commit it):

```bash
venv/bin/python - <<'EOF'
from context.brief import format_brief_prompt
snap = {  # minimal fixture — shape per context/snapshot.py
  "meta": {"clinic": {"name": "Test Clinic", "lat": 1.4, "lon": 103.8,
                      "nea_region_approx": "Clementi", "town": "Clementi"},
           "generated_at": "2026-09-09T09:00:00"},
  "air_quality": {"psi": {"peak": 72, "peak_region": "Clementi", "national": 60},
                  "pm25": {"peak": 34, "national": 25}},
  "weather": {"today": {"date": "2026-09-09", "high_c": 31, "low_c": 26},
              "flood": []},
  "dengue": {"clusters_active_total": 12, "in_high_aedes_area": True,
             "nearby_clusters": [{"locality": "Clementi", "case_size": 4, "km": 1.2}]},
  "disease_week": {"epi_week": "2026-W36", "date_range": "8-14 Sep",
                   "dengue": {"week": 18, "median_5yr": 11},
                   "influenza": {"ili_positivity_pct": 8.4}},
  "catchment_change": {"window": "90d", "healthcare_decisions_90d_count": 7,
                       "healthcare_decisions_90d": [{"title": "clinic"}]},
  "nearest_services": {"pyclinics": [{"name": "Woodlands Polyclinic", "km": 1.1}]},
  "protocol_links": {"active": [{"signal": "pm25", "protocols": ["Asthma"]}]},
  "data_gaps": ["ura: no API key"],
}
system, user = format_brief_prompt(snap, {"guideline_pdfs": 96, "protocol_pages": 17,
                                          "public_guidance_pages": 1, "total_chunks": 3120})
print(system)
print("---USER---")
print(user)
assert "Asthma" not in user, "protocol material leaked into user payload"
print("OK: no protocol material in user payload")
EOF
```

Expected: both prompts print; the assert passes (the fixture deliberately
includes `protocol_links` to prove the projection strips it).

- [ ] **Step 6: 0-credit dry run #2 — parse/guard logic against mocks (no LLM)**

Scratchpad script:

```bash
venv/bin/python - <<'EOF'
import json, rag
good = json.dumps({"headline": "Hazy day, elevated PSI.",
  "watch": [{"finding": "PSI 72 peak in Clementi", "why_it_matters": "airway irritation likely",
             "action": "advise N95 for outdoor work", "source": "NEA"}],
  "outlook": "Improving by Friday."})
bad_json = "Sure! Here is your brief: it looks hazy."
violating = json.dumps({"headline": "h",
  "watch": [
    {"finding": "PSI 72", "why_it_matters": "per the asthma protocol, review inhalers",
     "action": "x", "source": "NEA"},
    {"finding": "dengue week up", "why_it_matters": "national rise", "action": "counsel",
     "source": "CDA WIDB"}],
  "outlook": "ok"})
r1 = rag._parse_and_guard(good)
assert r1["provenance_drops"] == 0 and len(r1["watch"]) == 1, r1
r2 = rag._parse_and_guard(bad_json)
assert r2["parse_error"] and r2["watch"] == [], r2
r3 = rag._parse_and_guard(violating)
assert r3["provenance_drops"] == 1 and len(r3["watch"]) == 1, r3
assert r3["watch"][0]["finding"] == "dengue week up"
print("OK: parse + provenance guard behave as specified")
EOF
```

Expected: `OK: parse + provenance guard behave as specified`. (Review
hardening: JSON reliability and the guard are proven at 0 credits before the
single paid call.)

- [ ] **Step 7: Boot + free-endpoint checks**

Boot the server; then (all 0 credits):

```bash
curl -s "localhost:5001/api/brief" | venv/bin/python -m json.tool     # expect status "none" or "stale"
curl -s "localhost:5001/api/brief?lat=1.429&lon=103.783" | venv/bin/python -c "import json,sys; d=json.load(sys.stdin); print(d['status'], d['snapshot']['state'])"
curl -s -o /dev/null -w '%{http_code}\n' -X POST "localhost:5001/api/brief/generate?lat=99&lon=99"   # expect 400
```

- [ ] **Step 8: THE ONE PAID GATE (1 LLM call) + 429 check (0 credits)**

```bash
# 1 LLM call (≤2 provider round-trips). First run for the default clinic
# point may take ~60–90 s if the snapshot is cold (15–40 s build + LLM).
curl -s -X POST "localhost:5001/api/brief/generate" | venv/bin/python -m json.tool
#   expect: cached:false, brief.headline non-empty, watch[] items with
#   finding/why_it_matters/action/source, provenance_drops (0 or more)
curl -s "localhost:5001/api/brief" | venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['status'])"   # expect: fresh
curl -s -o /dev/null -w '%{http_code} retry-after:%{header_retry_after}\n' -X POST "localhost:5001/api/brief/generate?force=1"
#   expect: 429 with a Retry-After header, and 0 new LLM calls
```

If the paid call fails (e.g. credits exhausted), do NOT retry: record the
error, and the 90-s sentinel + 502 behaviour can be verified from the server
log + one follow-up `curl -X POST` (which must return the cached 502, not a
new call). Note this in the ledger and stop the task.

- [ ] **Step 9: Standing checks + commit**

`GET /` → 200; `GET /api/status` → ready; `venv/bin/python -m context` →
exit 0 (snapshot module still healthy). Then:

```bash
git add context/brief.py rag.py app.py
git commit -m "feat: on-demand clinician brief (GET /api/brief + POST /api/brief/generate) with stampede/cooldown/failure guards"
```

**Gate:** dry runs passed before the paid call; exactly one paid LLM call
made; `GET /api/brief` → `fresh`; forced regenerate → 429 + `Retry-After`;
no auto-generation paths exist.

---

### Task 3: UI — hash-routed tabs + Brief dashboard (0 credits)

**Files:**
- Modify: `static/index.html` only (CSS + HTML restructure + JS). No backend changes.

**Interfaces:**
- Consumes: Task 2's `GET /api/brief` + `POST /api/brief/generate`; existing
  `ctxQuery`, `refreshContext()`, `renderContext(snap, cached)`,
  `escapeHtml()`, `.ctx-basis b-*` / `.ctx-proto` / `.ctx-counsel` classes.
- Produces (for Task 4): the `#view-library` section, the tab router (which
  calls `loadLibrary()` once Task 4 defines it), `.lib-*`-free baseline CSS.

**DOM rule (hard, spec §5):** tab switching toggles the `hidden` attribute
**only**. The context-strip nodes (`#ctx-chips`, `#ctx-cards`, `#ctx-locbar`,
`#ctx-stamp`) are never removed or re-created — the 15-min `refreshContext()`
interval (line ~807) and the 10-s status re-poll (line ~810) fire regardless
of the active tab and would throw on null nodes.

- [ ] **Step 1: Read the page end-to-end and confirm anchors**

Read all of `static/index.html` (~813 lines). Confirm these anchors exist as
described (line numbers from this plan's authoring — re-locate by name, not
line): `<header>` (title + `#status-badge`); `<section id="ctx-strip">`
containing `#ctx-bar` (`#ctx-toggle`, `#ctx-chips`, `#ctx-stamp`,
`#ctx-refresh`) and `#ctx-detail` (`#ctx-cards`, `#ctx-locbar` with
`#ctx-pc`/`#ctx-lat`/`#ctx-lon`, `#ctx-src`); `<main>` containing
`#ingest-bar`, `#pdf-bar`, `#chat-window`, `#input-area`; script globals
`ctxQuery` (`{}` = default clinic), `refreshContext()`,
`renderContext(snap, cached)`, `toggleCtx()` (labels `'Context ▸'` /
`'Context ▾'`), `updateCtxLocation()`, `resetCtxLocation()`, bootstrap
`checkStatus(); refreshContext();` + two `setInterval`s, `escapeHtml()`; the
function-local `BASIS_LABEL` map inside `renderContext`; CSS rule
`#ctx-strip { width:100%; max-width:800px; padding:10px 16px 0; }`.

- [ ] **Step 2: HTML restructure (moves + wrappers — no node deletion)**

a. Insert after `</header>`:

```html
<nav id="tabs">
  <button class="tab" data-tab="brief" onclick="switchTab('brief')">Brief</button>
  <button class="tab" data-tab="chat" onclick="switchTab('chat')">Chat</button>
  <button class="tab" data-tab="library" onclick="switchTab('library')">Library</button>
</nav>
```

b. Make `<main>` a three-view container (its existing CSS keeps
max-width/flex/overflow/padding). Inside `<main>`:

```html
<section id="view-brief">
  <div id="brief-head">
    <div id="brief-title-row">
      <span id="brief-title">Clinic brief</span>
      <span id="brief-stamp"></span>
    </div>
    <!-- MOVE the existing #ctx-locbar block here, verbatim (same IDs/handlers) -->
    <div id="brief-cta-row">
      <button id="brief-generate" onclick="generateBrief()">Generate Brief</button>
      <span id="brief-progress"></span>
    </div>
  </div>
  <div id="brief-tiles" class="brief-tiles"></div>
  <div id="brief-error" class="brief-error" hidden></div>
  <div id="brief-content"></div>
  <!-- MOVE the existing #ctx-strip here, verbatim, with #ctx-locbar removed
       from #ctx-detail and the following two tweaks:
       - #ctx-detail gets class="hidden" initially
       - #ctx-toggle: aria-expanded="false", label "Raw context data ▸" -->
</section>
<section id="view-chat" hidden>
  <!-- MOVE the existing #ingest-bar, #chat-window, #input-area here, verbatim.
       #pdf-bar STAYS here for now — Task 4 moves it to Library. -->
</section>
<section id="view-library" hidden>
  <div id="library"><div class="empty-state" style="flex:none;padding:20px 0"><p>Library</p></div></div>
</section>
```

c. JS touch-ups that go with the move:
- `toggleCtx()`: change the two label strings to `'Raw context data ▸'` /
  `'Raw context data ▾'`.
- `#ctx-strip` CSS: `padding: 10px 16px 0;` → `padding: 10px 0 0;`
  (horizontal padding now comes from `main`).

- [ ] **Step 3: CSS additions** (append in `<style>`, reusing the existing
palette `#1a56db` / `#16a34a` / `#d97706` / `#dc2626` / card shadows):

```css
/* --- Phase 6: tabs + brief view ------------------------------------------ */
#tabs { width: 100%; max-width: 800px; display: flex; gap: 6px; padding: 10px 16px 0; }
#tabs .tab { flex: 1; padding: 8px 0; font-size: 0.85rem; font-weight: 600;
  background: #e2e8f0; color: #334155; border: none; border-radius: 8px; cursor: pointer; }
#tabs .tab:hover { background: #cbd5e1; }
#tabs .tab.active { background: #1a56db; color: #fff; }

main > section { flex: 1; display: flex; flex-direction: column; min-height: 0; }
main > section[hidden] { display: none; }
#view-brief, #view-library { overflow-y: auto; gap: 12px; }
#view-chat { overflow: hidden; }

#brief-title-row { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
#brief-title { font-size: 1.05rem; font-weight: 600; }
#brief-stamp { font-size: 0.72rem; color: #64748b; }
#brief-cta-row { display: flex; align-items: center; gap: 10px; }
#brief-generate { padding: 8px 14px; font-size: 0.85rem; font-weight: 600;
  background: #1a56db; color: #fff; border: none; border-radius: 8px; cursor: pointer; }
#brief-generate:hover { background: #1e40af; }
#brief-generate:disabled { background: #94a3b8; cursor: default; }
#brief-progress { font-size: 0.75rem; color: #64748b; }
.brief-error { background: #fef2f2; border: 1px solid #fecaca; color: #b91c1c;
  border-radius: 8px; padding: 8px 12px; font-size: 0.82rem; }

.brief-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.tile { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
  padding: 10px 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-top: 3px solid #94a3b8; }
.tile.good { border-top-color: #16a34a; }
.tile.warn { border-top-color: #d97706; }
.tile.bad  { border-top-color: #dc2626; }
.tile .t { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.03em; }
.tile .v { font-size: 1.05rem; font-weight: 600; margin-top: 2px; }
.tile .s { font-size: 0.72rem; color: #64748b; margin-top: 2px; }

.brief-headline { font-size: 1.25rem; font-weight: 600; line-height: 1.4; }
.brief-section-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: #64748b; margin-top: 14px; }
.watch-card { background: #fff; border: 1px solid #e2e8f0; border-left: 4px solid #94a3b8;
  border-radius: 10px; padding: 10px 12px; margin-top: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
.watch-card .f { font-weight: 600; font-size: 0.9rem; }
.watch-card .w { font-size: 0.85rem; color: #334155; margin-top: 2px; }
.watch-card .a { font-size: 0.85rem; margin-top: 2px; }
.watch-card .a b { color: #1a56db; }
.watch-card.src-nea { border-left-color: #1a56db; }
.watch-card.src-dgsg { border-left-color: #0e7490; }
.watch-card.src-widb { border-left-color: #b45309; }
.watch-card.src-ura { border-left-color: #15803d; }
.brief-src { display: inline-block; font-size: 0.7rem; background: #f1f5f9; color: #334155;
  border-radius: 999px; padding: 2px 8px; margin-top: 6px; }
.brief-outlook { font-size: 0.95rem; line-height: 1.5; }
.brief-prose { background: #fffbeb; border-left: 4px solid #d97706; border-radius: 8px;
  padding: 10px 12px; font-size: 0.88rem; white-space: pre-wrap; }
.brief-footer { font-size: 0.72rem; color: #64748b; margin-top: 12px; line-height: 1.5; }
.spot-empty { font-size: 0.85rem; color: #16a34a; }
```

- [ ] **Step 4: JS — tab router + brief state**

a. Tab router (insert after the `ctxLoading` declaration block):

```js
// --- Phase 6: tabs + brief ------------------------------------------------
let lastSnap = null;           // latest snapshot; renderContext() sets it
let brief = { status: 'none', data: null, generating: false };
const SRC_CLASS = { 'NEA': 'nea', 'data.gov.sg': 'dgsg', 'CDA WIDB': 'widb', 'URA': 'ura' };

const TABS = ['brief', 'chat', 'library'];
function currentTab() {
  const t = (location.hash || '').replace(/^#\//, '');
  return TABS.includes(t) ? t : 'brief';
}
function applyTab(tab) {
  document.getElementById('view-brief').hidden = tab !== 'brief';
  document.getElementById('view-chat').hidden = tab !== 'chat';
  document.getElementById('view-library').hidden = tab !== 'library';
  document.querySelectorAll('#tabs .tab').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === tab));
  if (tab === 'brief') {
    if (!lastSnap) refreshContext();   // first visit: populate tiles (free)
    loadBrief();                       // free GET — never a POST (credit guard)
  }
  if (tab === 'library' && typeof loadLibrary === 'function') loadLibrary();
  // Task 4 removes the typeof guard once loadLibrary() exists.
}
function switchTab(tab) {
  if (location.hash !== '#/' + tab) location.hash = '#/' + tab;
  else applyTab(tab);
}
window.addEventListener('hashchange', () => applyTab(currentTab()));
```

b. Hooks into existing context code (2-line + 2-call additions, nothing
else in `refreshContext`/`renderContext` changes):
- First statement of `renderContext(snap, cached)`: `lastSnap = snap;`
- Last statements of `renderContext(snap, cached)`:
  `renderBriefTiles(snap); renderSpotlight(snap);`
- End of `updateCtxLocation()` and `resetCtxLocation()` (after their
  `refreshContext();`): `loadBrief();` — new clinic point = new brief cache
  key; the free GET updates the CTA state.
- Bootstrap: after the existing `checkStatus(); refreshContext();` line,
  add `applyTab(currentTab());` (before the two `setInterval`s).

c. KPI tile renderer — spec §6.1 table, colour rules + D10 gap prefixes
(`psi`/`pm25` → Air; `forecast_24hr`/`wbgt`/`flood_alerts`/`outlook_4day` →
Weather; `dengue_clusters`/`aedes_areas` → Dengue; `widb` → WIDB; `ura` →
Planning; `polyclinics` → Nearest polyclinic). On a gap the tile renders
**"unavailable — <gap text>"** in warn style:

```js
function gapFor(snap, prefixes) {
  return (snap.data_gaps || []).find(g => prefixes.some(p => g.startsWith(p)));
}
function tileHtml(title, cls, value, sub) {
  return `<div class="tile ${cls}"><div class="t">${title}</div>`
    + `<div class="v">${value}</div><div class="s">${sub || ''}</div></div>`;
}
function renderBriefTiles(snap) {
  const el = document.getElementById('brief-tiles');
  const c = (snap.meta || {}).clinic || {};
  const aq = snap.air_quality || {}, w = snap.weather || {}, d = snap.dengue || {};
  const dw = snap.disease_week || {}, cc = snap.catchment_change || {}, ns = snap.nearest_services || {};
  if (c.name) document.getElementById('brief-title').textContent = 'Brief — ' + c.name;
  const gap = g => gapFor(snap, g);
  let tiles = '';
  // Air (PSI) — good ≤50 · warn 51–100 · bad >100
  const airGap = gap(['psi', 'pm25']);
  if (airGap) tiles += tileHtml('Air', 'warn', 'unavailable', escapeHtml(airGap));
  else if (aq.psi && aq.psi.peak != null) {
    const cls = aq.psi.peak > 100 ? 'bad' : (aq.psi.peak >= 51 ? 'warn' : 'good');
    const sub = (aq.psi.peak_region || '')
      + ((aq.pm25 && aq.pm25.peak != null) ? ' · PM2.5 ' + aq.pm25.peak.toFixed(0) : '');
    tiles += tileHtml('Air (PSI)', cls, String(aq.psi.peak), escapeHtml(sub));
  } else tiles += tileHtml('Air', '', 'no signal', 'PSI unavailable');
  // Weather — warn if a flood alert is present, else good
  const wGap = gap(['forecast_24hr', 'wbgt', 'flood_alerts', 'outlook_4day']);
  if (wGap) tiles += tileHtml('Weather', 'warn', 'unavailable', escapeHtml(wGap));
  else if (w.today) {
    const nFlood = (w.flood && (w.flood.active_alerts || []).length) || 0;
    const cls = nFlood ? 'warn' : 'good';
    const sub = w.town_2hr
      ? escapeHtml(w.town_2hr.town || '') + ' 2hr: ' + escapeHtml(w.town_2hr.forecast || '')
      : escapeHtml(w.today.text || '');
    tiles += tileHtml('Weather', cls, w.today.high_c + '°/' + w.today.low_c + '°' + (nFlood ? ' ⚠' : ''), sub);
  } else tiles += tileHtml('Weather', '', 'no signal', 'forecast unavailable');
  // Dengue nearby — bad if any nearby cluster ≥3 cases · warn if high-aedes
  // or a small nearby cluster · good otherwise
  const dGap = gap(['dengue_clusters', 'aedes_areas']);
  const near = d.nearby_clusters || [];
  if (dGap) tiles += tileHtml('Dengue nearby', 'warn', 'unavailable', escapeHtml(dGap));
  else if (near.some(x => (x.case_size || 0) >= 3))
    tiles += tileHtml('Dengue nearby', 'bad', Math.max(...near.map(x => x.case_size || 0)) + ' cases',
      escapeHtml(near[0].locality || '') + ' ' + near[0].km + ' km');
  else if (near.length || d.in_high_aedes_area)
    tiles += tileHtml('Dengue nearby', 'warn',
      near.length ? (near[0].case_size || 0) + ' cases' : 'no nearby clusters',
      (d.in_high_aedes_area ? 'high aedes area' : escapeHtml(near[0].locality || ''))
        + (d.clusters_active_total != null ? ' · ' + d.clusters_active_total + ' active island-wide' : ''));
  else tiles += tileHtml('Dengue nearby', 'good', 'no clusters',
    d.clusters_active_total != null ? d.clusters_active_total + ' active island-wide' : '');
  // WIDB week — bad if week > 1.5×5-yr median · warn if > median · else neutral
  const wdGap = gap(['widb']);
  const dwD = dw.dengue || {}, flu = (dw.influenza || {}).ili_positivity_pct;
  if (wdGap) tiles += tileHtml('WIDB week', 'warn', 'unavailable', escapeHtml(wdGap));
  else if (dwD.week != null) {
    let cls = '';
    if (dwD.median_5yr != null) cls = dwD.week > 1.5 * dwD.median_5yr ? 'bad'
      : (dwD.week > dwD.median_5yr ? 'warn' : 'good');
    const sub = (dw.epi_week || '') + (dwD.median_5yr != null ? ' · 5yr median ' + dwD.median_5yr : '')
      + (flu != null ? ' · flu ILI ' + flu + '%' : '');
    tiles += tileHtml('WIDB week', cls, dwD.week + ' dengue cases', escapeHtml(sub));
  } else tiles += tileHtml('WIDB week', '', 'no signal', 'bulletin unavailable');
  // Planning 90d — informational; sub-line: permission ≠ opened facility
  const pGap = gap(['ura']);
  if (pGap) tiles += tileHtml('Planning 90d', 'warn', 'unavailable', escapeHtml(pGap));
  else if (cc.healthcare_decisions_90d_count != null)
    tiles += tileHtml('Planning 90d', '', cc.healthcare_decisions_90d_count + ' decisions',
      'permission ≠ opened facility');
  else tiles += tileHtml('Planning 90d', '', 'no signal', 'URA unavailable');
  // Nearest polyclinic — neutral: name + km
  const nGap = gap(['polyclinics']);
  const py = (ns.pyclinics || [])[0];
  if (nGap) tiles += tileHtml('Nearest polyclinic', 'warn', 'unavailable', escapeHtml(nGap));
  else if (py) tiles += tileHtml('Nearest polyclinic', '', py.km + ' km', escapeHtml(py.name || ''));
  else tiles += tileHtml('Nearest polyclinic', '', 'no data', '');
  el.innerHTML = tiles;
}
```

d. Protocol spotlight — spec §6.2 item 3: rendered **100% from structured
`snapshot.protocol_links.active`**, no LLM; reuses the existing
`.ctx-basis b-*` / `.ctx-proto` / `.ctx-counsel` classes (the label map
mirrors the function-local `BASIS_LABEL` in `renderContext`):

```js
const SPOT_BASIS = {
  corpus: 'in protocol text', partial: 'partial: mechanism in text',
  derived: 'derived: not protocol text', none: 'counselling only'
};
function renderSpotlight(snap) {
  const el = document.getElementById('brief-spotlight');
  if (!el || !snap) return;
  const active = ((snap.protocol_links || {}).active) || [];
  el.innerHTML = active.length
    ? active.map(l => {
        const b = SPOT_BASIS[l.basis] ? l.basis : 'derived';
        return `<div class="watch-card"><div class="f">${escapeHtml(l.signal || '')}</div>`
          + `<div class="w">${escapeHtml(l.detail || '')} `
          + `<span class="ctx-basis b-${b}" title="${escapeHtml(l.basis_note || '')}">${SPOT_BASIS[b]}</span></div>`
          + `<div class="ctx-protos">${(l.protocols || []).map(p =>
              `<span class="ctx-proto">${escapeHtml(p)}</span>`).join('')}</div>`
          + (l.population_counselling
              ? `<div class="a">Counsel: ${escapeHtml(l.population_counselling)}</div>` : '')
          + '</div>';
      }).join('')
    : '<div class="spot-empty">No elevated signals — all context signals within normal range.</div>';
}
```

`#brief-spotlight` is created inside `#brief-content` by `renderBrief()`
(next step); `renderContext` re-calls `renderSpotlight(lastSnap)` so a
context refresh updates an already-rendered brief.

e. Brief state + renderers (spec §6.2 render order: headline → Watch →
Spotlight → Outlook → footer; degraded states from spec §6.2):

```js
async function loadBrief() {
  try {
    const qs = new URLSearchParams(ctxQuery).toString();
    const res = await fetch('/api/brief' + (qs ? '?' + qs : ''));
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || 'HTTP ' + res.status);
    brief.data = d;
    renderBrief(d);
  } catch (e) {
    const err = document.getElementById('brief-error');
    err.hidden = false;
    err.textContent = 'Brief status unavailable: ' + (e.message || e);
  }
}

function renderBrief(d) {
  const errEl = document.getElementById('brief-error');
  const btn = document.getElementById('brief-generate');
  const stamp = document.getElementById('brief-stamp');
  errEl.hidden = true;
  if (d.last_error) { errEl.hidden = false; errEl.textContent = 'Last generation failed: ' + d.last_error; }
  if (d.status === 'fresh') { btn.textContent = 'Regenerate (costs 1 LLM call)'; btn.dataset.force = '1'; }
  else { btn.textContent = 'Generate Brief'; btn.dataset.force = '0'; }
  if (d.as_of) stamp.textContent = 'brief as of ' + new Date(d.as_of).toLocaleString();

  const c = document.getElementById('brief-content');
  const b = d.brief;
  if (!b) {
    c.innerHTML = '<div class="empty-state" style="flex:none;padding:12px 0"><p>'
      + 'No brief yet — the tiles above are live data. Generate Brief adds the inference '
      + 'layer (1 LLM call, 0 embeddings, cached 15 min).</p></div>';
    return;
  }
  const drops = d.provenance_drops || 0;
  const gaps = (lastSnap && lastSnap.data_gaps) ? lastSnap.data_gaps : [];
  let html = '<div class="brief-headline">' + escapeHtml(b.headline || '') + '</div>';
  if (b.parse_error) {   // degraded: unparseable LLM output → labelled prose
    html += '<div class="brief-section-title">Watch today</div>'
      + '<div class="brief-prose">' + escapeHtml(b.raw || '') + '</div>'
      + '<div class="brief-footer">Unstructured model output — shown as-is (JSON parse failed).</div>';
    c.innerHTML = html;
    renderSpotlight(lastSnap);
    return;
  }
  html += '<div class="brief-section-title">Watch today</div>';
  if (!b.watch.length) html += '<div class="watch-card"><div class="w">No AI items passed the '
    + 'provenance check — see the tiles and Protocol spotlight above.</div></div>';
  html += b.watch.map(it =>
    `<div class="watch-card src-${SRC_CLASS[it.source] || ''}">`
    + `<div class="f">${escapeHtml(it.finding)}</div>`
    + `<div class="w">${escapeHtml(it.why_it_matters)}</div>`
    + `<div class="a"><b>Action:</b> ${escapeHtml(it.action)}</div>`
    + `<span class="brief-src">${escapeHtml(it.source || 'derived')}</span></div>`).join('');
  html += '<div class="brief-section-title">Protocol spotlight</div><div id="brief-spotlight"></div>';
  if (b.outlook) html += '<div class="brief-section-title">Outlook</div>'
    + '<div class="brief-outlook">' + escapeHtml(b.outlook) + '</div>';
  html += '<div class="brief-footer">'
    + (gaps.length ? 'Data gaps: ' + gaps.map(escapeHtml).join(' · ') : 'All sources OK')
    + (drops ? ' · ' + drops + ' watch item(s) withheld by the provenance check' : '')
    + '<br>Generated from live data ' + (d.as_of || '')
    + ' · 1 LLM call (≤2 provider round-trips, 0 embeddings) · cached 15 min</div>';
  c.innerHTML = html;
  renderSpotlight(lastSnap);
}
```

f. The one paid action — `generateBrief()` (the only POST in the UI; staged
progress text per spec §6.2, same elapsed-timer pattern as chat):

```js
async function generateBrief() {
  if (brief.generating) return;               // double-click guard (client side)
  brief.generating = true;
  const btn = document.getElementById('brief-generate');
  const prog = document.getElementById('brief-progress');
  const force = btn.dataset.force === '1';
  btn.disabled = true;
  const t0 = Date.now();
  prog.textContent = 'Fetching live data… 0s';
  const timer = setInterval(() => {
    const s = Math.floor((Date.now() - t0) / 1000);
    prog.textContent = (s < 15 ? 'Fetching live data… ' : 'Writing brief… ') + s + 's';
  }, 1000);
  try {
    const qs = new URLSearchParams(ctxQuery).toString();
    const res = await fetch('/api/brief/generate' + (qs ? '?' + qs + '&' : '?')
      + 'force=' + (force ? 1 : 0), { method: 'POST' });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || 'HTTP ' + res.status);
    renderBrief({ status: 'fresh', last_error: null, as_of: d.as_of,
                  brief: d.brief, provenance_drops: d.provenance_drops || 0 });
  } catch (e) {
    const err = document.getElementById('brief-error');
    err.hidden = false;
    err.textContent = 'Generate failed: ' + (e.message || e);
  } finally {
    clearInterval(timer);
    btn.disabled = false;
    brief.generating = false;
    prog.textContent = '';
  }
}
```

- [ ] **Step 5: `node --check` on the page script**

```bash
sed -n '/<script>/,/<\/script>/p' static/index.html | sed '1d;$d' > /tmp/cp_rag_page.js && node --check /tmp/cp_rag_page.js
```

Expected: no output (pass).

- [ ] **Step 6: Boot + UI manual checklist (0 credits)**

Restart the server; `GET /` → 200. **Cache-warmth rule:** the Generate click
below must be served from Task 2's 15-min brief cache (0 credits). If more
than 15 min have elapsed since Task 2's paid call, re-run Task 2 Step 8's
single paid curl first to re-warm the cache (a re-run of the phase's one paid
verification call — not a new one; note it in the ledger).

1. Load `localhost:5001` (defaults to `#/brief`): 6 KPI tiles render from the
   live snapshot; any gap tile shows "unavailable — <gap text>" in warn style;
   title row shows the clinic name.
2. Click **Generate Brief** (warm cache → `cached: true`, 0 credits): staged
   progress text appears, then headline + Watch cards (source chips) +
   Protocol spotlight + Outlook + footer render. The spotlight must match
   `curl -s localhost:5001/api/context` → `protocol_links.active` **exactly**
   (same signals, protocols, basis badges; if empty, the green "No elevated
   signals…" line).
3. Button now reads **"Regenerate (costs 1 LLM call)"**; clicking it within
   60 s → UI error line shows the 429 cooldown message; 0 LLM calls (server
   log).
4. **Double-click** Generate rapidly (after the cooldown or on a fresh
   point): exactly one build — the second request awaits the same in-flight
   future (server log shows one LLM round-trip or one cache hit).
5. Switch to the **Chat tab**: console clean; from the console call
   `refreshContext()` manually → **zero console errors** (ctx nodes
   present-but-hidden — the review's hardening gate). Chat view renders
   (no messages sent — 0 credits).
6. "Raw context data ▸" at the bottom of the Brief tab expands/collapses the
   verbatim 7-card panel; chips/stamp/locbar behave as before.
7. Direct reload at `#/library`: Library tab active, placeholder section,
   no JS errors. Direct reload at `#/bogus`: routes to Brief.

- [ ] **Step 7: Standing checks + commit**

`GET /api/status` → `ready: true`; `venv/bin/python -m context` → exit 0.
Then:

```bash
git add static/index.html
git commit -m "feat: hash-routed tabs (Brief default / Chat / Library) + clinician brief dashboard UI"
```

**Gate:** `node --check` passes; all checklist items green; 0 new LLM
credits this task (Generate served from warm cache); DOM rule respected
(ctx nodes never removed; router toggles `hidden` only); chat behaviour
unchanged; no automatic POST anywhere in the UI.

---

### Task 4: Library screen UI (0 credits)

**Files:**
- Modify: `static/index.html` only.

**Interfaces:**
- Consumes: Task 1's `GET /api/library` (`totals`, `pdfs[]`,
  `web_sources{site: [{url, chunks, derived_title}]}`); Task 3's
  `#view-library` section + router + `.brief-error` class.
- Produces: JS `loadLibrary()`, `renderLibrary(data)`.

**Credit note:** the moved upload bar is existing code, but a **live upload
spends embedding credits** (`POST /api/ingest-pdf` re-embeds the new PDF).
Do NOT perform a live upload in this task without explicit operator approval
— the upload path is unchanged existing code; we verify placement/wiring only.

- [ ] **Step 1: Pre-check the live corpus inventory (0 credits) and record it**

With the server running:

```bash
curl -s "localhost:5001/api/library" | venv/bin/python - <<'EOF'
import json, sys
d = json.load(sys.stdin)
t = d["totals"]
print("totals:", t)
print("pdfs:", len(d["pdfs"]), "| sample:", d["pdfs"][0] if d["pdfs"] else None)
for site, rows in d["web_sources"].items():
    print("site:", site, "| urls:", len(rows))
    for r in rows[:3]:
        print("   ", r["url"], "|", r["derived_title"], "|", r["chunks"])
EOF
```

**Record in the SDD ledger** (`.superpowers/sdd/2026-09-09-clinician-brief-library/ledger.md`):
the actual protocol-page URL count and shape (watch for the
`chronic-care-protocols → chronic` rewrite and any anchor/subpage sprawl from
the depth-2 crawl). Note which branch of the collapse rule (>25 URLs) is live.
This is spec §9's Task-1 pre-check, executed here because it drives the
Task-4 rendering decision.

- [ ] **Step 2: Move the upload bar from Chat to Library**

Cut the existing `#pdf-bar` block (label + `#pdf-file` input + `#pdf-status`
+ `#pdf-count`) out of `#view-chat` and paste it into `<section
id="view-library">` **before** `<div id="library">` (a sibling — NOT inside
it, because `renderLibrary()` replaces `#library`'s `innerHTML` on every
load). Same IDs, same `onchange="uploadPdf(this)"`; `refreshPdfs()` is
already called on page load and after uploads, so the count keeps working.
In `uploadPdf`'s success path (after the existing `refreshPdfs();` call),
add one line:

```js
if (typeof loadLibrary === 'function') loadLibrary();
```

- [ ] **Step 3: `loadLibrary()` + wire the router**

```js
// --- Phase 6: library ------------------------------------------------------
let lastLibrary = null;
async function loadLibrary() {
  const el = document.getElementById('library');
  if (!el) return;
  try {
    const res = await fetch('/api/library');
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || 'HTTP ' + res.status);
    lastLibrary = d;
    renderLibrary(d);
  } catch (e) {
    el.innerHTML = '<div class="brief-error">Library unavailable: '
      + escapeHtml(String(e.message || e)) + '</div>';
  }
}
```

In Task 3's `applyTab()`, remove the `typeof loadLibrary === 'function'`
guard (it now exists): `if (tab === 'library') loadLibrary();`

- [ ] **Step 4: `renderLibrary(data)` — spec §9 layout (top → bottom)**

Totals bar · PDF cards (with the moved upload bar above them) · protocol
pages · MOH guidance. **The raw URL is always rendered beside the derived
title** (titles are cosmetic only). Collapse rule: a web section with **>25
URLs** renders as a single site-level card instead of per-URL rows (the
Step-1 ledger note records which branch is live for the real corpus):

```js
function renderLibrary(d) {
  const el = document.getElementById('library');
  const t = d.totals || {};
  const docs = (t.guideline_pdfs || 0) + (t.protocol_pages || 0) + (t.public_guidance_pages || 0);
  let html = '<div id="lib-totals">' + t.chunks + ' chunks · ' + docs + ' documents: '
    + (t.guideline_pdfs || 0) + ' guideline PDFs · ' + (t.protocol_pages || 0)
    + ' protocol pages · ' + (t.public_guidance_pages || 0) + ' guidance pages</div>';

  html += '<h3 class="lib-h">Clinical guideline PDFs</h3><div class="pdf-grid">';
  html += (d.pdfs || []).map(p => {
    const src = p.source || '';
    const link = /^https?:/.test(src)
      ? ' <a class="s" href="' + escapeHtml(src) + '" target="_blank" rel="noopener">source</a>' : '';
    return '<div class="lib-card"><div class="f">' + escapeHtml(p.title || 'Untitled PDF') + '</div>'
      + '<div class="s">' + p.chunks + ' chunks' + link
      + (p.doc_hash ? ' · <span title="' + escapeHtml(p.doc_hash) + '">' + escapeHtml(p.doc_hash.slice(0, 8)) + '</span>' : '')
      + '</div></div>';
  }).join('');
  html += '</div>';

  const webSec = (name, rows) => {
    let h = '<h3 class="lib-h">' + name + '</h3>';
    if (!rows || !rows.length) { h += '<div class="s dim">none in corpus</div>'; return h; }
    if (rows.length > 25) {   // URL-sprawl collapse rule (spec §9)
      h += '<div class="lib-card"><div class="s">' + rows.length
        + ' pages — collapsed (URL sprawl). Full list: `curl /api/library`.</div></div>';
      return h;
    }
    h += rows.map(r => '<div class="lib-row"><span class="f">' + escapeHtml(r.derived_title) + '</span> '
      + '<a class="s" href="' + escapeHtml(r.url) + '" target="_blank" rel="noopener">'
      + escapeHtml(r.url) + '</a> <span class="s dim">' + r.chunks + ' chunks</span></div>').join('');
    return h;
  };
  const ws = d.web_sources || {};
  html += webSec('Care-protocol pages (primarycarepages.sg)', ws['primarycarepages.sg']);
  html += webSec('Public health guidance (moh.gov.sg)', ws['moh.gov.sg']);
  el.innerHTML = html;
}
```

- [ ] **Step 5: Library CSS** (append; reuses the card/tile look):

```css
#lib-totals { font-size: 0.85rem; font-weight: 600; background: #fff;
  border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px 12px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
.lib-h { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; }
.pdf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; }
.lib-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
  padding: 8px 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
.lib-card .f, .lib-row .f { font-weight: 600; font-size: 0.85rem; }
.lib-card .s, .lib-row .s { font-size: 0.75rem; color: #64748b; }
.lib-row { padding: 4px 0; border-bottom: 1px solid #f1f5f9; font-size: 0.8rem; }
.lib-row a, .lib-card a { color: #1a56db; text-decoration: none; word-break: break-all; }
.lib-row a:hover, .lib-card a:hover { text-decoration: underline; }
.s.dim { color: #94a3b8; }
```

- [ ] **Step 6: `node --check` + manual checklist (0 credits — no live upload)**

`node --check` (same command as Task 3 Step 5). Then in the browser:
1. Direct reload at `#/library`: totals bar numbers match the Step-1 curl
   exactly; PDF grid renders (96 guideline PDFs, chunk counts, source links,
   8-char hashes with full hash in the tooltip); protocol-pages and MOH
   sections render per-URL rows (or the collapsed card, per the Step-1
   finding); **every derived title has its raw URL beside it**.
2. The upload bar ("Clinical guidelines / Upload PDF") now sits in the
   Library view, not Chat; the Chat view is free of it.
3. **No live upload test** (embedding spend). Verify only: the bar renders,
   the file input is present and wired to `uploadPdf(this)`.
4. Tab round-trip Brief → Library → Chat → Brief: zero console errors;
   Brief tiles/brief still render; `#pdf-count` still populates on load.

- [ ] **Step 7: Standing checks + commit**

`GET /api/status` → `ready: true`; `GET /` → 200. Then:

```bash
git add static/index.html
git commit -m "feat: Library screen (corpus inventory, raw URLs) + move PDF upload to Library"
```

**Gate:** `node --check` passes; Library matches `GET /api/library`
exactly; raw URL always beside derived title; collapse branch consistent
with the ledger pre-check; upload bar moved with handlers untouched; 0
credits spent.

---

### Task 5: Documentation + final standing checks (0 credits)

**Files:**
- Modify: `docs/signal-research.md`, `CLAUDE.md`, `HANDOFF.md`
- Create (untracked, local-only): `.superpowers/sdd/2026-09-09-clinician-brief-library/ledger.md`
  (directory follows the existing `.superpowers/sdd/<date>-<slug>/ledger.md`
  convention — `.superpowers/` is git-ignored)

- [ ] **Step 1: SDD ledger**

Create `.superpowers/sdd/2026-09-09-clinician-brief-library/ledger.md`
(matching the Phase-4 ledger format: header + one line per task + rulings):
- Header: plan path, spec path.
- One line per task 1–4: commit hash, gate results, credit count
  (Tasks 1/3/4 = 0; Task 2 = exactly one paid call — model, timestamp,
  response shape, `provenance_drops`, 429 negative-check result; note any
  re-warm used by Task 3 Step 6).
- Task 4 Step-1 pre-check numbers (protocol URL count/shape; collapse branch).
- Rulings log (any deviation from plan text, with reason).
- Final line: `PLAN COMPLETE — all 5 tasks done, gates green, committed
  (<range>, ahead of origin/main <sha>)`.

- [ ] **Step 2: `docs/signal-research.md`**

Read the file first (it carries the §9 phase table + §15 build notes). Add:
- §9 phase table: Phase 6 row → DONE (with date).
- New Phase 6 section (after §15, following its style): what was built —
  three hash-routed tabs (Brief default / Chat / Library);
  `context/brief.py` prompt projection (protocol_links structurally absent);
  `rag.py` `generate_brief()`/`_parse_brief()`/`_apply_provenance_guard()`;
  `app.py` two-phase brief API + `GET /api/library`; UI tiles/spotlight.
  Guards: free GET never builds; POST guard order (400 → 503 → fresh cache →
  cached-error 502 → in-flight future → 60-s 429 → build); 15-min success
  cache; 90-s failure sentinel; provenance drop guard (patterns + trade-off).
  Cost model verbatim from spec §8.3. Verification results: paid-call record,
  429 check, parse/guard mock results, protocol-URL pre-check numbers,
  UI checklist results.

- [ ] **Step 3: `CLAUDE.md`**

Read the file first; edit in place keeping its structure:
- Repo layout: add `context/brief.py` (pure brief prompt builder, stdlib-only).
- API/commands: add `GET /api/brief`, `POST /api/brief/generate` (the only
  LLM path outside `/api/chat`; guard order + 60-s cooldown + 90-s failure
  sentinel + 15-min cache), `GET /api/library`.
- Golden rules / notes: brief cache is in-memory (server restart = cold);
  one generate = ≤2 provider round-trips, 0 embeddings, 0 retrieval;
  provenance drop guard on LLM watch items; Protocol spotlight is
  structured-data-only.
- UI: three hash-routed tabs (`#/brief` default, `#/chat`, `#/library`);
  PDF upload moved to Library; context strip survives as the "Raw context
  data" drill-down in the Brief tab (DOM never removed — intervals depend on
  it).

- [ ] **Step 4: `HANDOFF.md`**

Read the file first; rewrite following the Phase-4 format:
- Header: `updated 2026-09-09, Phase 6 — clinician brief dashboard + Library`.
- "Done this session (Phase 6, all gates green, committed)": per-file
  summary (like Phase 4's), endpoints, guards, verified list (boot,
  `/api/status`, `node --check`, paid-call record, 429 check, UI checklist,
  pre-check numbers), scope rulings (D1–D11 one-liner).
- Outstanding: refresh — remove what Phase 6 closed; add: live PDF upload
  untested this phase (embedding spend); brief + snapshot caches lost on
  server restart (by design); primarycarepages refresh + postcode geocoding
  carry-over items stay.
- Next-session prompt: update state line (Phases 0–6 all done, N commits
  ahead of `origin/main` — push when you commit; SDD ledgers in
  `.superpowers/sdd/`).

- [ ] **Step 5: Final standing checks (0 credits)**

```bash
# restart server, then:
curl -s localhost:5001/api/status | venv/bin/python -m json.tool      # ready: true
curl -s -o /dev/null -w '%{http_code}\n' localhost:5001/               # 200
curl -s "localhost:5001/api/brief" | venv/bin/python -m json.tool      # no crash; status fresh/stale/none
curl -s "localhost:5001/api/library" | venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['totals'])"
sed -n '/<script>/,/<\/script>/p' static/index.html | sed '1d;$d' > /tmp/cp_rag_page.js && node --check /tmp/cp_rag_page.js
venv/bin/python -m context                                            # exit 0
git status                                                            # clean — only intended files committed
git log --oneline -6                                                  # 4 phase commits (Tasks 1–4) visible
```

- [ ] **Step 6: Commit docs**

```bash
git add docs/signal-research.md CLAUDE.md HANDOFF.md
git commit -m "docs: Phase 6 clinician brief + library (signal-research, CLAUDE, handoff)"
```

(The SDD ledger is untracked/local-only — do NOT `git add` it.)

**Gate:** all standing checks pass; `git status` clean after the docs
commit; total LLM calls for the whole phase = 1 (plus a re-warm of that same
call only if Task 3 Step 6 required it); ledger complete with all rulings.

---

## Plan complete

All five tasks: 1) `list_corpus()` + `GET /api/library` (0 credits),
2) brief backend + the single paid verification call, 3) tabs + Brief
dashboard UI (0 credits), 4) Library UI (0 credits), 5) docs + final checks
(0 credits). Credit total for the phase: exactly one brief generation
(≤2 provider round-trips) + its possible one re-warm; everything else is
free. Provenance safety is structural: `protocol_links` never reaches the
brief LLM, the server drops violating `watch` items, and the Protocol
spotlight renders only from structured snapshot data.






