# ACE Clinical Guidelines (PDF) Ingest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put Singapore ACE clinical guidelines (PDFs) into the RAG corpus so the chat can answer guideline questions with citations — manual PDF upload for prototyping (core), plus an optional crawler that seeds all 29 guidelines from ace-hta.gov.sg.

**Architecture:** `pypdf` extracts text from a PDF → the existing `RecursiveCharacterTextSplitter` (1000/200) chunks it → chunks are appended to the existing persisted Chroma store with `source_site` / `doc_title` / `doc_hash` metadata (dedupe by sha256 of file bytes) → a new 4th prompt section "Clinical guidelines" in `rag.py` routes those chunks separately from protocol content → new `POST /api/ingest-pdf` (multipart) + `GET /api/pdfs` endpoints in `app.py` (blocking work in `run_in_executor` per project rule) → a slim upload bar + PDF list in `static/index.html` → optional `scripts/ace_guidelines.py` CLI crawler (sitemap → detail pages → "Download the ACG" PDF links → download → `rag.ingest_pdf`).

**Tech Stack:** FastAPI `UploadFile`, `pypdf` (new dependency — see Global Constraints), existing LangChain 0.3.x / Chroma / OpenRouter embeddings (`openai/text-embedding-ada-002`), `requests` + `BeautifulSoup` + stdlib in the crawler.

**Spec:** This plan. Research findings verified live on 2026-09-08: (1) the ACE repository listing page (https://www.ace-hta.gov.sg/healthcare-professionals/ace-repository-for-clinical-guidelines/) is client-side React pagination — `?page=2` returns identical HTML; (2) the site **sitemap.xml lists all 31 guideline URLs** (index + overview + 29 detail pages), which bypasses pagination entirely; (3) each detail page's "Download the ACG" section carries **direct public PDF URLs** on `isomer-user-content.by.gov.sg` (e.g. `https://isomer-user-content.by.gov.sg/68/de592078-…/When to order MRI for low back pain (June 2026) [PDF].pdf`), no auth; some guidelines link 2–3 PDFs (main + EtR framework + references).

## Global Constraints

- **Python 3.9 only** (system `/usr/bin/python3`). No PEP 604 `X | Y` unions, no `match` statements. `list[str]` is fine (PEP 585 is 3.9).
- **One new runtime dependency: `pypdf>=4.0.0`** — requires explicit user sign-off before Task 1 (the project's "stack (decided)" rule). Pure-Python, MIT-licensed, the only sane way to parse PDFs. No other new dependencies (no test framework — the project's scope guard parks "a test suite"; verification is curl + `venv/bin/python -c` smoke checks + server logs, the project's standing convention).
- **Credits are a budget.** Embedding cost ≈ 1 batch call per ≤2048 chunks (ada-002 via OpenRouter). One guideline PDF ≈ 50–100 chunks ≈ a fraction of a cent; the full 29-guideline seed ≈ 3–5k chunks ≈ a few cents. Every ingest step is user-initiated; the full ACE seed (Task 6 final step) runs only on explicit GO.
- **Never call blocking code directly in an `async def` endpoint** — PDF parse + embed goes through `run_in_executor`, the same rule that fixed the server freeze.
- **`chroma_db/` is precious** — append only, never a full `ingest()` rebuild, never delete it.
- **UI stays one static HTML file**, no framework, no build step, escaped plain text only.
- **`.env` is machine-parsed** (one `KEY=value` per line); keep `.env.example` in sync.
- Server runs at `http://localhost:5001` (`venv/bin/uvicorn app:app --port 5001`); free readiness check is `curl -s localhost:5001/api/status`.
- Commit after every task (project convention: frequent commits per task).

---

### Task 0: Commit current state (model switch + earlier session fixes)

The model switch is already done and verified live (chat probe with `deepseek/deepseek-v4-flash-0731` returned in **8 s** with correct protocol citations — vs 4–5 min on the free tier). This task documents and commits it together with the still-uncommitted executor/port/UX fixes from the 2026-09-08 morning session.

**Files:**
- Modified (already updated this session, verify they contain the expected content): `CLAUDE.md` (model name lines 3 + 18 now say `deepseek/deepseek-v4-flash-0731`; port 8000→5001 on lines 52–53), `HANDOFF.md`, `app.py`, `rag.py`, `static/index.html`, `.env.example`
- Modified: `.env` (gitignored; `CHAT_MODEL=deepseek/deepseek-v4-flash-0731` — NOT committed)

- [ ] **Step 1: Append model-switch note to `HANDOFF.md`** (after the "Slow-chat UX fix" bullet in the "Done this session" list):

```markdown
- **Chat model switched** — `CHAT_MODEL` in `.env` changed `nvidia/nemotron-3.5-lightning:free` → `deepseek/deepseek-v4-flash-0731` (paid, fast; verified on OpenRouter's model list 2026-09-08). Live probe: COPD smoking-cessation question answered in 8 s with correct primarycarepages.sg citations (free tier took 4–5 min). Requires OpenRouter credits — the account has them. `.env.example` documents the option.
- **New plan** — `docs/superpowers/plans/2026-09-08-ace-guidelines-pdf.md` (Phase 5: ACE clinical-guidelines PDF ingest — manual upload core + optional ACE repository crawler).
```

- [ ] **Step 2: Standing checks**

Run:
```bash
curl -s localhost:5001/api/status && echo && curl -s -o /dev/null -w "UI: HTTP %{http_code}\n" localhost:5001/
```
Expected: `{"ready":true,...}` and `UI: HTTP 200`.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "fix: unblock event loop on chat/ingest, bound LLM calls, plain-text UI, deepseek-v4-flash model"
```

---

### Task 1: pypdf dependency + PDF text extraction in `rag.py`

**Files:**
- Modify: `requirements.txt` (append one line)
- Modify: `rag.py` (add imports + `MAX_PDF_BYTES` + `pdf_text()`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `pdf_text(data: bytes) -> str` (used by Task 2); `MAX_PDF_BYTES` int = 25 MB (used by Task 3).

- [ ] **Step 0: Get user sign-off for the new dependency** (ask: "Add `pypdf` (pure-Python PDF parser, MIT) to requirements.txt?"). Do not proceed without a yes.

- [ ] **Step 1: Add the dependency and install**

Append to `requirements.txt`:
```
pypdf>=4.0.0
```
Run:
```bash
venv/bin/python -m pip install -r requirements.txt
```
Expected: `Successfully installed pypdf-x.y.z`.

- [ ] **Step 2: Add extraction code to `rag.py`**

After `import os` at the top of `rag.py`:
```python
import hashlib
import io
```
After the existing `from bs4 import BeautifulSoup` import:
```python
from pypdf import PdfReader
```
After the `CHROMA_DIR = "./chroma_db"` line:
```python
MAX_PDF_BYTES = 25 * 1024 * 1024  # guideline PDFs run 1-10 MB; bounded upload cap


def pdf_text(data: bytes) -> str:
    """Extract readable text from a PDF, one labelled block per page.

    Raises ValueError for password-protected PDFs. Returns "" when the
    PDF has no text layer (scanned image) — caller treats that as a
    reject.
    """
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        if not reader.decrypt(""):
            raise ValueError("PDF is password-protected")
    blocks = []
    for i, page in enumerate(reader.pages, start=1):
        t = (page.extract_text() or "").strip()
        if t:
            blocks.append("[page %d]\n%s" % (i, t))
    return "\n\n".join(blocks)
```

- [ ] **Step 3: Smoke-test extraction on a real ACE guideline PDF** (0 API credits; one download)

```bash
curl -s -m 60 -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" \
  "https://isomer-user-content.by.gov.sg/68/de592078-23a4-453e-ac60-63bc1bab2ea8/When%20to%20order%20MRI%20for%20low%20back%20pain%20(June%202026)%20%5BPDF%5D.pdf" \
  -o /tmp/ace_mri.pdf && ls -la /tmp/ace_mri.pdf
venv/bin/python -c "
import rag
t = rag.pdf_text(open('/tmp/ace_mri.pdf','rb').read())
print('chars:', len(t))
print(t[:400])
assert len(t) > 10000, 'suspiciously little text'
assert 'MRI' in t
print('OK')
"
```
Expected: `chars: 40000+`, readable clinical English mentioning MRI, then `OK`. If this file has no text layer (scanned), use one of the other two PDFs on the same detail page (links are in `/tmp/ace_page.html` / the detail page) — at least one must extract cleanly before proceeding.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt rag.py
git commit -m "feat: pypdf dependency + pdf_text() extraction (Phase 5 groundwork)"
```

---

### Task 2: `ingest_pdf()` + `list_pdfs()` in `rag.py` (append-only, dedupe by content hash)

**Files:**
- Modify: `rag.py` (add code after `ingest_append()`, before `_active_protocol_names()`)

**Interfaces:**
- Consumes: `pdf_text()` (Task 1), existing `_split()`, `load_vectorstore()`, `_norm_url()`.
- Produces: `ingest_pdf(data: bytes, title: str, source: str = "", source_site: str = "guideline-upload") -> (chunks_added: int, skipped: bool, doc_hash: str)` (used by Task 3 + Task 6); `list_pdfs() -> list` of dicts `doc_hash`, `title`, `source`, `chunks` (Task 3); `GUIDELINE_SITES = {"ace-hta.gov.sg", "guideline-upload"}` (Task 4 routing).

- [ ] **Step 1: Add the code to `rag.py`**

```python
# Phase 5: clinical-guideline PDFs (ACE ACGs + manual uploads) route to the
# "Clinical guidelines" prompt section — never presented as protocol content.
GUIDELINE_SITES = {"ace-hta.gov.sg", "guideline-upload"}


def _store_pdf_index(vectorstore: Chroma):
    """(doc_hash set, normalized source-URL set) already in the store
    (local read, 0 credits) — dedupe for PDF ingest."""
    hashes, urls = set(), set()
    for m in vectorstore.get(include=["metadatas"]).get("metadatas") or []:
        m = m or {}
        if m.get("doc_hash"):
            hashes.add(m["doc_hash"])
        u = m.get("source")
        if u:
            urls.add(_norm_url(u))
    return hashes, urls


def ingest_pdf(data: bytes, title: str, source: str = "",
               source_site: str = "guideline-upload"):
    """Parse + chunk + embed + append one PDF. Dedupes by sha256 of the file
    bytes (and by source URL when given), so re-ingests cost 0 credits.

    Returns (chunks_added, skipped, doc_hash). Raises ValueError when the
    PDF has no extractable text (scanned image / password-protected).
    """
    text = pdf_text(data)
    if not text.strip():
        raise ValueError("no extractable text in PDF (scanned image?)")
    doc_hash = hashlib.sha256(data).hexdigest()[:16]
    vectorstore = load_vectorstore()
    have_hashes, have_urls = _store_pdf_index(vectorstore)
    src = source or "upload:%s" % title
    if doc_hash in have_hashes or _norm_url(src) in have_urls:
        return 0, True, doc_hash
    doc = Document(
        page_content=text,
        metadata={"source": src, "source_site": source_site,
                  "doc_title": title, "doc_hash": doc_hash},
    )
    splits = _split([doc])
    vectorstore.add_documents(splits)
    return len(splits), False, doc_hash


def list_pdfs():
    """One row per ingested PDF: doc_hash, title, source, chunks."""
    vectorstore = load_vectorstore()
    agg = {}
    for m in vectorstore.get(include=["metadatas"]).get("metadatas") or []:
        m = m or {}
        if not m.get("doc_hash"):
            continue
        e = agg.setdefault(m["doc_hash"], {
            "title": m.get("doc_title", "?"),
            "source": m.get("source", ""),
            "chunks": 0})
        e["chunks"] += 1
    return [{"doc_hash": k, "title": v["title"], "source": v["source"],
             "chunks": v["chunks"]} for k, v in agg.items()]
```

- [ ] **Step 2: Smoke-test ingest of the MRI PDF** (one embedding batch call, a fraction of a cent — the only credit cost in this task)

```bash
venv/bin/python -c "
import rag
P = 'https://isomer-user-content.by.gov.sg/68/de592078-23a4-453e-ac60-63bc1bab2ea8/When to order MRI for low back pain (June 2026) [PDF].pdf'
T = 'When to order MRI for low back pain (ACG, Jun 2026)'
data = open('/tmp/ace_mri.pdf','rb').read()
added, skipped, h = rag.ingest_pdf(data, T, source=P, source_site='ace-hta.gov.sg')
print('added=%d skipped=%s hash=%s' % (added, skipped, h))
assert added > 20 and not skipped
added2, skipped2, _ = rag.ingest_pdf(data, T, source=P, source_site='ace-hta.gov.sg')
print('second: added=%d skipped=%s' % (added2, skipped2))
assert added2 == 0 and skipped2
rows = [p for p in rag.list_pdfs() if p['title'].startswith('When to order MRI')]
print(rows); assert rows and rows[0]['chunks'] > 20
print('OK')
"
```
Expected: `added=40+ skipped=False ...`, `second: added=0 skipped=True`, one list row, `OK`.

- [ ] **Step 3: Restart the server** (running app holds the old `rag` module + chain):

```bash
pkill -f "uvicorn app:app --port 5001"; sleep 2
(nohup venv/bin/uvicorn app:app --port 5001 >> /tmp/cp_rag_server.log 2>&1 &)
sleep 5; curl -s -m 8 localhost:5001/api/status
```
Expected: `{"ready":true,...}`.

- [ ] **Step 4: Commit**

```bash
git add rag.py
git commit -m "feat: ingest_pdf() + list_pdfs() — append-only PDF corpus with hash dedupe"
```

---

### Task 3: `POST /api/ingest-pdf` + `GET /api/pdfs` endpoints in `app.py`

**Files:**
- Modify: `app.py` (extend the two import lines; add two endpoints after `run_ingest_append()`)

**Interfaces:**
- Consumes: `ingest_pdf`, `list_pdfs`, `MAX_PDF_BYTES` from `rag` (Tasks 1–2).
- Produces: `POST /api/ingest-pdf` — multipart, field `file` required, `title` + `source` optional form fields → `{"status","title","chunks_added","skipped","doc_hash"}`; `GET /api/pdfs` → `{"pdfs": [...]}`. Used by Task 5 (UI).

- [ ] **Step 1: Extend imports in `app.py`**

The FastAPI import (line 8) becomes:
```python
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
```
The rag import (line 15) becomes:
```python
from rag import (build_chain, ingest, ingest_append, load_vectorstore, CHROMA_DIR,
                 ingest_pdf, list_pdfs, MAX_PDF_BYTES)
```

- [ ] **Step 2: Add the endpoints** (immediately after `run_ingest_append()`, before `@app.get("/api/status")`):

```python
@app.post("/api/ingest-pdf")
async def run_ingest_pdf(file: UploadFile = File(...),
                         title: str = Form(""), source: str = Form("")):
    """Ingest one uploaded guideline PDF (parse + embed, dedupe by content
    hash). Parse/embed is blocking — executor per the no-freeze rule."""
    global rag_chain
    data = await file.read()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=400, detail="PDF exceeds 25 MB cap")
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
```

- [ ] **Step 3: Restart + verify** (restart command as in Task 2 Step 3), then:

```bash
# 1. upload the MRI PDF (deduped — already in store from Task 2: 0 credits)
curl -s -F "file=@/tmp/ace_mri.pdf" -F "title=When to order MRI for low back pain (ACG, Jun 2026)" localhost:5001/api/ingest-pdf
# expected: {"status":"ok","title":"...","chunks_added":0,"skipped":true,...}
# 2. non-PDF rejected
echo "hello" > /tmp/notapdf.txt
curl -s -F "file=@/tmp/notapdf.txt" localhost:5001/api/ingest-pdf
# expected: 400 "not a PDF file"
# 3. list
curl -s localhost:5001/api/pdfs
# expected: one row, title "When to order MRI ...", chunks > 20
# 4. loop still responsive
curl -s -m 3 localhost:5001/api/status
# expected: {"ready":true,...}
```

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: POST /api/ingest-pdf (multipart, executor) + GET /api/pdfs"
```

---

### Task 4: "Clinical guidelines" prompt section + routing + source display

**Files:**
- Modify: `rag.py` (`PROMPT_TEMPLATE` and `_prepare_sections()`)
- Modify: `app.py` (source mapping in `chat()`, line 117)

**Interfaces:**
- Consumes: `GUIDELINE_SITES` (Task 2), chunk metadata `doc_title` / `source_site` (Task 2).
- Produces: a 4-section prompt (template vars `protocol_content`, `guidelines`, `public_guidance`, `local_context`, `clinic`, `as_of`, `question`); chat `sources` show the PDF's `doc_title` instead of the raw URL for PDF chunks.

- [ ] **Step 1: Replace `PROMPT_TEMPLATE` in `rag.py`** (whole new template — adds the guidelines bullet + section; everything else byte-identical):

```python
PROMPT_TEMPLATE = """Answer the question using only the four sections below. If they do not cover the question, say you don't know.

- "Protocol content": excerpts from the clinic's chronic-care protocols. This is the only material you may cite as protocol content. Where basis notes are given they state exactly what the protocol text does and does not support — do not over-claim beyond them.
- "Clinical guidelines": Singapore ACE clinical guidelines (ACGs) and other uploaded clinical guidance PDFs. Cite them as guideline content — never as protocol content.
- "Public health guidance": government public health guidance (e.g. MOH). You may cite it, but as public guidance — never as protocol content.
- "Local context": live population-level signals (air quality, weather, dengue) for the clinic point. These are observations about the area right now, NOT protocol content: use them to frame the answer (environmental triggers, sick-day rules, counselling), but never attribute them to the protocols.

Formatting: the answer renders as PLAIN TEXT in a chat window — it has NO markdown support. Never use **bold**, *italic*, # headings, or backticks; write plain words only. Simple dash bullets and a blank line between sections are fine.

== Protocol content ==
{protocol_content}

== Clinical guidelines ==
{guidelines}

== Public health guidance ==
{public_guidance}

== Local context — clinic: {clinic}, as of {as_of} ==
{local_context}

Question: {question}
"""
```

- [ ] **Step 2: Route guideline chunks in `_prepare_sections()` in `rag.py`**

Replace the per-doc loop body:
```python
    protocol_parts, public_parts = [], []
    for d in docs or []:
        site = d.metadata.get("source_site", "primarycarepages.sg")
        body = (d.page_content or "").strip()
        if not body:
            continue
        ref = "Source: %s" % d.metadata.get("source", "")
        if site in PUBLIC_GUIDANCE_SITES:
            public_parts.append(
                "%s\n[%s — public health guidance, not clinic protocol content] %s"
                % (body, site, ref))
        else:
            protocol_parts.append("%s\n[%s] %s" % (body, site, ref))
```
with:
```python
    protocol_parts, public_parts, guideline_parts = [], [], []
    for d in docs or []:
        site = d.metadata.get("source_site", "primarycarepages.sg")
        body = (d.page_content or "").strip()
        if not body:
            continue
        ref = "Source: %s" % d.metadata.get("source", "")
        if site in PUBLIC_GUIDANCE_SITES:
            public_parts.append(
                "%s\n[%s — public health guidance, not clinic protocol content] %s"
                % (body, site, ref))
        elif site in GUIDELINE_SITES:
            guideline_parts.append(
                "%s\n[%s] %s"
                % (body, d.metadata.get("doc_title", "guideline"), ref))
        else:
            protocol_parts.append("%s\n[%s] %s" % (body, site, ref))
```
And in the returned dict, add after the `"protocol_content": ...` entry:
```python
        "guidelines": ("\n\n---\n\n".join(guideline_parts)
                       or "(no clinical guidelines retrieved for this question)"),
```

- [ ] **Step 3: Show the PDF title in chat sources** — in `app.py` `chat()`, replace:
```python
    sources = list({doc.metadata.get("source", "") for doc in result["context"]})
```
with:
```python
    # PDF chunks carry doc_title — show the guideline name, not the raw URL
    sources = list({(doc.metadata.get("doc_title") or doc.metadata.get("source", ""))
                    for doc in result["context"]})
```

- [ ] **Step 4: Restart the server** (Task 2 Step 3 command), then run the ONE chat probe for this task (1 credit — deepseek-v4-flash, fast):

```bash
curl -s -m 120 localhost:5001/api/chat -H 'Content-Type: application/json' \
  -d '{"question":"When should a clinician order an MRI for low back pain?"}'
```
Expected: an answer drawn from the MRI guideline (red-flag / MRI-timing content, not generic), `sources` contains the PDF title `When to order MRI for low back pain (ACG, Jun 2026)` (not the isomer URL), no markdown. Also sanity-check the protocol path is intact (0 extra credits — skip; retrieval split is proven by the two different section labels in the prompt).

- [ ] **Step 5: Commit**

```bash
git add rag.py app.py
git commit -m "feat: 4th prompt section 'Clinical guidelines' + doc_title in chat sources"
```

---

### Task 5: UI — upload bar + ingested-PDF list in `static/index.html`

**Files:**
- Modify: `static/index.html` (CSS after the `#ingest-bar` rules ~line 188; HTML between `#ingest-bar` and `#chat-window` ~line 380; JS after `runIngest()` ~line 511)

**Interfaces:**
- Consumes: `POST /api/ingest-pdf` (multipart `file`, `title`) and `GET /api/pdfs` (Task 3).
- Produces: visible upload control; no new global state beyond `pdfStatus`/`pdfCount` elements.

- [ ] **Step 1: CSS** — append after the `#ingest-bar button:hover...` rule:

```css
    #pdf-bar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
      margin: 0 0 10px; padding: 7px 12px; border-radius: 10px;
      background: #f0f9ff; border: 1px solid #bae6fd; font-size: 0.82rem; }
    #pdf-bar .pdf-label { font-weight: 600; color: #075985; }
    #pdf-bar label.pdf-btn { cursor: pointer; padding: 4px 10px; border-radius: 8px;
      background: #0ea5e9; color: #fff; font-weight: 600; font-size: 0.8rem; }
    #pdf-bar label.pdf-btn:hover { background: #0284c7; }
    #pdf-status { color: #0369a1; }
    #pdf-count { margin-left: auto; color: #64748b; font-size: 0.75rem; }
```

- [ ] **Step 2: HTML** — insert between `</div>` of `#ingest-bar` (line 379) and `<div id="chat-window">` (line 381):

```html
  <div id="pdf-bar">
    <span class="pdf-label">Clinical guidelines</span>
    <label class="pdf-btn">Upload PDF<input id="pdf-file" type="file"
      accept="application/pdf,.pdf" style="display:none" onchange="uploadPdf(this)"></label>
    <span id="pdf-status"></span>
    <span id="pdf-count"></span>
  </div>
```

- [ ] **Step 3: JS** — append after the `runIngest()` function (top level of the existing `<script>`):

```js
  async function refreshPdfs() {
    try {
      const res = await fetch('/api/pdfs');
      const data = await res.json();
      const n = (data.pdfs || []).length;
      document.getElementById('pdf-count').textContent =
        n ? (n + ' guideline PDF' + (n > 1 ? 's' : '') + ' in store') : '';
    } catch (e) { /* server not up yet */ }
  }

  async function uploadPdf(input) {
    const f = input.files && input.files[0];
    input.value = '';
    if (!f) return;
    const status = document.getElementById('pdf-status');
    status.textContent = 'Uploading + embedding...';
    const t0 = Date.now();
    const timer = setInterval(() => {
      status.textContent = 'Uploading + embedding... ' +
        Math.floor((Date.now() - t0) / 1000) + 's';
    }, 1000);
    try {
      const fd = new FormData();
      fd.append('file', f);
      fd.append('title', f.name.replace(/\.pdf$/i, ''));
      const res = await fetch('/api/ingest-pdf', { method: 'POST', body: fd });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || ('HTTP ' + res.status));
      status.textContent = data.skipped
        ? 'Already in store (duplicate)'
        : 'Added ' + data.chunks_added + ' chunks';
      refreshPdfs();
    } catch (e) {
      status.textContent = 'Upload failed: ' + e.message;
    } finally {
      clearInterval(timer);
    }
  }
  refreshPdfs();
```

- [ ] **Step 4: Verify in the browser** — hard-refresh http://localhost:5001; expected: the "Clinical guidelines" bar shows `1 guideline PDF in store` (the MRI one from Task 2). Click "Upload PDF" and re-upload `/tmp/ace_mri.pdf` → status reads "Already in store (duplicate)" (0 credits). If available, open DevTools → Network to confirm the POST returned 200.

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat: guideline PDF upload bar + ingested-PDF count in UI"
```

---

### Task 6 (OPTIONAL, user-gated): ACE repository crawler — seed all 29 guidelines

Skip this whole task if the user wants manual uploads only. It is a one-off CLI seeder: sitemap → 29 detail pages → every "Download the ACG" PDF → download → `rag.ingest_pdf`. Polite: 1 s sleep between requests, browser UA, ~60 requests total.

**Files:**
- Create: `scripts/ace_guidelines.py`

**Interfaces:**
- Consumes: `rag.ingest_pdf`, `rag._BROWSER_UA` (Tasks 1–2).
- Produces: `venv/bin/python scripts/ace_guidelines.py --list | --ingest [--limit N]`.

- [ ] **Step 1: Create `scripts/ace_guidelines.py`**

```python
#!/usr/bin/env python3
"""ACE clinical-guidelines crawler (Phase 5, optional seeder).

sitemap.xml -> 29 guideline detail pages -> "Download the ACG" PDF links ->
download + ingest into the RAG store via rag.ingest_pdf (dedupes by hash,
so re-runs cost 0 credits for already-ingested PDFs).

Usage:
  venv/bin/python scripts/ace_guidelines.py --list             # dry run
  venv/bin/python scripts/ace_guidelines.py --ingest --limit 2  # try 2 pages first
  venv/bin/python scripts/ace_guidelines.py --ingest           # the full seed

Cost: one embedding batch per NEW pdf (~a few cents for all 29).
"""
import argparse
import re
import time
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup

import rag

BASE = "https://www.ace-hta.gov.sg"
UA = rag._BROWSER_UA
DETAIL_RE = re.compile(
    r"^https://www\.ace-hta\.gov\.sg/healthcare-professionals/"
    r"ace-repository-for-clinical-guidelines/[a-z0-9-]+/?$")


def _get(url):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


def detail_pages():
    """All guideline detail-page URLs from the sitemap.

    (The list page paginates client-side React; the sitemap has all 29.)
    The regex admits only /ace-repository-for-clinical-guidelines/<slug>/ —
    it excludes the index page (no slug) and ...-guidelines-overview/.
    """
    root = ET.fromstring(_get(BASE + "/sitemap.xml"))
    locs = [el.text.strip() for el in root.iter()
            if el.tag == "loc" or el.tag.endswith("}loc")]
    return sorted({u for u in locs if u and DETAIL_RE.match(u)})


def fetch_detail(url):
    """(guideline title, [pdf urls]) from one detail page."""
    soup = BeautifulSoup(_get(url), "html.parser")
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else url.rsplit("/", 2)[-2]
    pdfs = []
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h.lower().endswith(".pdf") and h not in pdfs:
            pdfs.append(h)
    return title, pdfs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="dry run")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max detail pages")
    args = ap.parse_args()
    if not args.list and not args.ingest:
        ap.error("pass --list or --ingest")

    pages = detail_pages()
    print("detail pages found: %d" % len(pages))
    found = []
    for i, u in enumerate(pages):
        if args.limit and i >= args.limit:
            break
        title, pdfs = fetch_detail(u)
        print("%d. %s" % (i + 1, title))
        for p in pdfs:
            print("     %s" % p)
            found.append((title, p))
        time.sleep(1.0)
    print("PDFs found: %d" % len(found))
    if not args.ingest:
        print("dry run only — re-run with --ingest")
        return

    ingested = skipped = failed = 0
    for title, pdf in found:
        try:
            r = requests.get(pdf, headers={"User-Agent": UA}, timeout=120)
            r.raise_for_status()
            added, was_skipped, _ = rag.ingest_pdf(
                r.content, title, source=pdf, source_site="ace-hta.gov.sg")
        except Exception as e:
            failed += 1
            print("  FAIL %s (%s)" % (pdf, e))
            time.sleep(1.0)
            continue
        if was_skipped:
            skipped += 1
            print("  skip (already in store) %s" % pdf)
        else:
            ingested += 1
            print("  + %d chunks  %s" % (added, title))
        time.sleep(1.0)
    print("done: %d ingested, %d skipped, %d failed" % (ingested, skipped, failed))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry run** (0 credits, ~30 s of polite fetching)

```bash
venv/bin/python scripts/ace_guidelines.py --list
```
Expected: `detail pages found: 29`, then 29 numbered titles each with 1–3 PDF URLs, `PDFs found: ~50-60`. If fewer than 29 pages (sitemap changed) or a page yields 0 PDFs, STOP and re-inspect that detail page's HTML before ingesting anything.

- [ ] **Step 3: Pilot ingest** (explicit user GO required; 2 pages ≈ 3–5 PDFs ≈ a few cents)

```bash
venv/bin/python scripts/ace_guidelines.py --ingest --limit 2
```
Expected: `+ N chunks` per new PDF (the MRI one prints `skip (already in store)`), `done: ... 0 failed`.

- [ ] **Step 4: Verify a pilot guideline end-to-end** (1 credit): pick a guideline ingested in Step 3 (e.g. Allergic rhinitis) and probe:

```bash
curl -s -m 120 localhost:5001/api/chat -H 'Content-Type: application/json' \
  -d '{"question":"Summarise the ACE guideline approach to allergic rhinitis diagnosis."}'
```
Expected: answer citing `Allergic Rhinitis – Diagnosis and Management` in `sources`, content specific to that ACG.

- [ ] **Step 5: Full seed** (explicit user GO; ~55 new PDFs ≈ 3–5k chunks ≈ a few cents, a few minutes wall time)

```bash
venv/bin/python scripts/ace_guidelines.py --ingest
```
Expected: `done: ~50 ingested, ~4 skipped, 0 failed`.

- [ ] **Step 6: Restart the server** (Task 2 Step 3 command) and confirm `/api/pdfs` shows all guidelines. Commit:

```bash
git add scripts/ace_guidelines.py
git commit -m "feat: ACE guidelines crawler (sitemap -> detail pages -> PDF ingest)"
```

---

### Task 7: Docs — phase registry + CLAUDE.md + HANDOFF.md

**Files:**
- Modify: `docs/signal-research.md` (§9 phase table)
- Modify: `CLAUDE.md` (repo layout + stack + endpoints)
- Modify: `HANDOFF.md` (session wrap-up)

- [ ] **Step 1: Add Phase 5 to the §9 phase table** in `docs/signal-research.md` (after the Phase 4 row):

```markdown
| 5 — ACE guidelines | Clinical-guidelines PDFs in the RAG corpus: `pypdf` extraction, append-only `ingest_pdf()` with hash dedupe, `POST /api/ingest-pdf` + `GET /api/pdfs`, 4th prompt section "Clinical guidelines", UI upload bar, optional `scripts/ace_guidelines.py` sitemap crawler for all 29 ACE ACGs. Plan: `docs/superpowers/plans/2026-09-08-ace-guidelines-pdf.md` | ~1–2 credits (upload path) / a few cents (full ACE seed) |
```

- [ ] **Step 2: Update `CLAUDE.md`**
  - Stack bullet (line 20 area): append to the BeautifulSoup line: "PDFs (Phase 5): `pypdf` text extraction + `ingest_pdf()` (append-only, sha256 dedupe) for clinical-guideline PDFs."
  - Repo layout: add `  scripts/          # ace_guidelines.py (Phase 5 optional crawler; stdlib + requests + bs4)`.
  - Endpoint list (line 28): `app.py # FastAPI app: /, /api/chat, /api/ingest, /api/ingest/append, /api/ingest-pdf, /api/pdfs, /api/status, /api/context`.
  - Prompt-section line (line 3): "The RAG prompt has four labelled sections — *Protocol content* / *Clinical guidelines* / *Public health guidance* / *Local context*".

- [ ] **Step 3: Update `HANDOFF.md`** — Done: Phase 5 core (tasks 0–5, and 6 if run) with verification evidence; In Progress: none; Outstanding: full ACE seed if not yet run (command above), revisit if ACE rehosts PDFs or changes the sitemap.

- [ ] **Step 4: Final standing checks + commit**

```bash
curl -s localhost:5001/api/status && echo && curl -s -o /dev/null -w "UI: HTTP %{http_code}\n" localhost:5001/
git add -A
git commit -m "docs: Phase 5 (ACE guidelines PDF ingest) in phase registry + CLAUDE/HANDOFF"
```

---

## Self-Review Notes (run at plan completion, before execution handoff)

1. **Spec coverage:** manual upload (Tasks 1–5), crawl-vs-upload question answered with verified research (sitemap + direct PDF links — both paths feasible; manual upload is the core, crawler optional Task 6), model switch (Task 0), phase placement (new Phase 5, row added in Task 7). No gaps.
2. **Placeholder scan:** every code step has full code; every verify step has expected output; no "TBD"/"similar to Task N".
3. **Type consistency:** `pdf_text(bytes)->str`, `ingest_pdf(bytes,str,str,str)->(int,bool,str)`, `list_pdfs()->list[dict]`, `MAX_PDF_BYTES`, `GUIDELINE_SITES` — used identically in Tasks 2/3/4/6; prompt vars `guidelines` match template + dict key in Task 4; UI fetches match the Task 3 response shapes.
4. **Risk flags:** (a) a guideline PDF could be a scanned image with no text layer — Task 1 Step 3 proves extraction on a real ACE PDF before any credit spend, Task 6 treats no-text PDFs as `FAIL` lines to review; (b) `pypdf` is a new dependency — gated on user sign-off (Task 1 Step 0); (c) the full ACE seed is the only multi-cent step — gated on explicit GO (Task 6 Steps 3/5).







