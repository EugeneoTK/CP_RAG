# Phase 6 Design Spec — Clinician Brief Dashboard + Library Screen

- **Date:** 2026-09-09
- **Status:** approved (design v2, external-review findings incorporated — see Appendix A)
- **Implements:** user request "a screen to see captured PDFs" + "a simple button to see a brief — a graphical dashboard for the clinician on what is happening around the clinic, with some inference, not solely care-protocol-based"
- **Supersedes:** nothing (additive phase; Phases 0–5 all done)

---

## 1. Problem

1. **Invisible corpus.** The product ingests care-protocol pages, MOH guidance, and
   96 ACE guideline PDFs into `chroma_db/`, and `GET /api/pdfs` already lists the
   PDFs — but there is **no screen** showing what is captured. Users cannot see the
   PDFs already in the product.
2. **Cluttered context page.** The F2 context strip + 7-card detail panel
   (`static/index.html`) is text-dense raw data. A clinician cannot get "what does
   this mean for my clinic today" from it. We need a **graphical, inference-capable
   brief** behind a simple button.

## 2. Product & repo context (abridged — full rules in `CLAUDE.md`)

- **Stack:** FastAPI + uvicorn, LangChain 0.3.x, Chroma persisted at `./chroma_db`,
  UI = one `static/index.html` (vanilla JS, no framework, no build step),
  `context/` package (stdlib-only) builds the live snapshot.
- **Corpus:** 17 chronic-care protocol pages (primarycarepages.sg), 1 MOH haze
  guidance page, 96 ACE guideline PDFs (~2,975 chunks). Chunk metadata: `source`
  (URL), `source_site`; PDFs additionally `doc_title`, `doc_hash`. **Web-crawled
  chunks carry no stored page title** — only the URL.
- **Snapshot:** `build_snapshot(lat, lon, name)` → air (PSI/PM2.5 by region),
  weather (`today`, `outlook_4day`, `town_2hr`, `wbgt`, `flood`), dengue
  (`clusters_active_total`, `nearby_clusters`, `in_high_aedes_area`), WIDB
  (`disease_week`), URA (`catchment_change`), `nearest_services`,
  `protocol_links` (active signal→protocol links with `basis` tags),
  `data_gaps`. 15-min in-memory cache keyed `"%.3f,%.3f"` (lat,lon); cold build
  ~15–40 s; a failing source never crashes the build — it lands in `data_gaps`.
- **Golden rules binding this feature:** API credits are a budget (account has hit
  `429 credit_balance_exhausted`; never loop/retry the API); Python 3.9 only;
  `context/` stdlib-only; **no test suite** (verification = boot + `/api/status`
  + `node --check` + deliberate curls); `chroma_db/` is read-only here (no
  re-embed/re-ingest); blocking work only via thread-pool executor in `async def`
  endpoints; **provenance rule** — live signals are observations, never protocol
  content; a URA written permission is NOT an opened facility; `derived` links are
  clinical synthesis, not protocol text. Scope guards: no auth/Docker/streaming/
  test suite/model swaps.

## 3. Scope

**In:**
- Three-tab UI (hash-routed): **Brief** (new, default), **Chat** (existing,
  unchanged), **Library** (new).
- `context/brief.py` (new, stdlib-only prompt builder), `rag.py` additions
  (`list_corpus()`, `generate_brief()`), `app.py` endpoints
  (`GET /api/brief`, `POST /api/brief/generate`, `GET /api/library`).
- PDF upload bar moved from Chat to Library.
- Docs: `docs/signal-research.md` Phase 6 notes, `CLAUDE.md`, `HANDOFF.md`,
  SDD ledger.

**Out (explicit non-goals):**
- No auth, no streaming, no Docker, no test suite, no new dependencies.
- No PDF text preview in Library (would need per-page content fetch).
- No per-clinic catchment geocoding (existing Outstanding item).
- No timer-based auto-generation of briefs (credit guard).
- No changes to the RAG chat chain/prompt/retrieval. No re-ingest, no
  re-embedding, no writes to `chroma_db/`.

## 4. Locked decisions (v2 — includes external review, Appendix A)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Library screen scope | **Full library**: guideline PDFs (main) + protocol pages + MOH pages + corpus totals — not PDFs only |
| D2 | Context strip fate | **Redesign around the Brief**: Brief is the home view; the existing 7-card panel survives verbatim as a collapsed "Raw context data" drill-down at the bottom of Brief |
| D3 | Brief generation | **On-demand LLM**: one small chat-model call, **no retrieval, no embeddings**; 15-min server cache per clinic point |
| D4 | API shape | **Two-phase**: free `GET /api/brief` (status/cached read only, never triggers work) + paid `POST /api/brief/generate` (only LLM path) |
| D5 | Stampede guard | `_brief_builds` in-flight dict (same pattern as `_context_builds`), keyed per clinic point; concurrent POSTs await the same future → 0 extra LLM calls |
| D6 | Regenerate cooldown | Server-side **60 s** minimum between forced regenerations per clinic point → `429` + `Retry-After`, no LLM call |
| D7 | Failure caching | On LLM failure, cache a **90 s error sentinel** in the brief cache → rapid retries during an outage cost 0 |
| D8 | Cost accounting | "1 Generate" = **≤2 provider round-trips** (`max_retries=1`), ~2–3 k tokens, 0 embeddings, 0 retrieval — stated in UI microcopy and docs |
| D9 | Clinical safety | `protocol_links` **stripped from the LLM prompt input**; Protocol spotlight rendered **100% from structured `protocol_links.active`** (no LLM prose); server-side **provenance drop guard** on LLM `watch[]` text |
| D10 | Data-gap display | Per-tile explicit **"unavailable (source)"** state keyed off `data_gaps`, in addition to the footer list |
| D11 | Task order | Library backend first (0-credit task, provides corpus stats + URL-shape pre-check), then brief backend, then UI tasks, then docs |

## 5. Information architecture

```
header (title + #status-badge)          ← unchanged
nav  [ Brief ] [ Chat ] [ Library ]     ← new row; hash-routed (#/brief default,
                                                 #/chat, #/library)
#view-brief      (default)              ← new: location bar · KPI tiles · brief
                                           CTA + rendered brief · "Raw context
                                           data" drill-down (old 7-card panel)
#view-chat       (existing <main>)      ← unchanged: ingest bar, chat, input
#view-library    (new)                  ← totals bar · PDF cards + upload bar ·
                                           protocol pages · MOH guidance
```

**DOM rules (hard):**
- Tab switching toggles CSS `display`/`[hidden]` **only**. The context-strip
  DOM nodes (`#ctx-chips`, `#ctx-cards`, `#ctx-locbar`, `#ctx-stamp`) are
  **never removed or re-created** by the tab router: the existing 15-min
  `refreshContext()` interval and the 10-s status re-poll fire regardless of
  the active tab and would throw on null nodes.
- The 7-card panel moves inside `#view-brief` (bottom, collapsed by default,
  toggle label becomes "Raw context data ▸"). Its internal markup, classes, and
  JS handlers are otherwise **untouched**.
- The page keeps the `100dvh` flex column; each view is a `flex: 1;
  overflow-y: auto` container so the chat view's existing scroll behaviour is
  preserved.

## 6. Brief view

### 6.1 Layer 1 — KPI tiles (free, deterministic, client-side)

Rendered from the snapshot JSON (same `GET /api/context` data as today; 0
credits). CSS grid, `auto-fit minmax(150px, 1fr)`. Colour semantics reuse the
existing `.ctx-chip good/warn/bad` palette.

| Tile | Data path (snapshot) | Colour rule |
|------|---------------------|-------------|
| **Air (PSI)** | `air_quality.psi.peak` + `peak_region` | good ≤ 50 · warn 51–100 · bad > 100 |
| **Weather** | `weather.today.high_c/low_c`, `weather.flood`, `weather.town_2hr.town` | warn if a flood alert is present, else good; sub-line: town + 2-hr forecast |
| **Dengue nearby** | `dengue.nearby_clusters` (≤ X km per `config.DENGUE_NEAR_KM`), `dengue.in_high_aedes_area`, `dengue.clusters_active_total` | bad if any nearby cluster with `case_size` ≥ 3 · warn if `in_high_aedes_area` or a nearby cluster with `case_size` < 3 · good otherwise |
| **WIDB week** | `disease_week.dengue.week` vs `dengue.median_5yr`; flu ILI% sub-line from `disease_week.influenza.ili_positivity_pct` | bad if week > 1.5 × 5-yr same-week median · warn if week > median · good otherwise; if median unknown, neutral with raw week count |
| **Planning 90d** | `catchment_change.healthcare_decisions_90d_count`, `window` | neutral/informational (no good/bad semantics); sub-line: "permission ≠ opened facility" |
| **Nearest polyclinic** | `nearest_services.pyclinics[0]` | neutral: name + km |

**Gap states (D10):** each tile maps to `data_gaps` prefixes —
`psi`/`pm25` → Air, `forecast_24hr`/`wbgt`/`flood_alerts`/`outlook_4day` →
Weather, `dengue_clusters`/`aedes_areas` → Dengue, `widb` → WIDB, `ura` →
Planning, `polyclinics` → Nearest polyclinic. On a gap the tile renders
**"unavailable — <gap text>"** in the warn style (distinguishes "source down"
from "no signal"). The footer still lists all gaps.

### 6.2 Layer 2 — the inferred brief (button-driven)

**Controls:** location bar (postcode / lat / lon + Update/Default — moved from
the old strip; drives the clinic point for both snapshot and brief). Primary
CTA **"Generate Brief"**; when a fresh brief is cached the button reads
**"Regenerate (costs 1 LLM call)"**. During generation: staged progress text
("Fetching live data… Ns" → "Writing brief… Ns", same elapsed-timer pattern as
chat).

**Rendered sections (top → bottom):**
1. **Situation headline** — one sentence from the LLM + "as of <generated_at>" stamp.
2. **Watch today** — ≤ 6 cards; each: `finding` (bold) · `why_it_matters` ·
   `action` · source tag chip (enum: `NEA` / `data.gov.sg` / `CDA WIDB` / `URA` /
   `derived`). When `provenance_drops > 0`, a one-line note: "N item(s) withheld
   by the provenance check."
3. **Protocol spotlight** — rendered **directly from
   `snapshot.protocol_links.active`** (structured, no LLM): per link — signal,
   detail, basis badge (reusing `.ctx-basis b-corpus/b-partial/b-derived/b-none`
   + legend), related-protocol chips, population-counselling line. When no
   active links: "No elevated signals — all context signals within normal range."
4. **Outlook** — 1–2 sentences (LLM).
5. **Footer** — `data_gaps` list (or "All sources OK") · provenance drops note ·
   "Generated from live data at X · 1 LLM call · cached 15 min" ·
   **"Raw context data ▸"** expanding the verbatim 7-card panel.

**Degraded renderings:**
- LLM returned unparseable text → prose block (styled, clearly labelled
  "unstructured model output") — the brief is still shown, never an error page.
- `watch` empty after the drop guard → deterministic line: "No AI items passed
  the provenance check — see the tiles and Protocol spotlight above."
- Snapshot not ready (never built for this point) → tiles show "building…"
  placeholders; CTA remains enabled (the POST path builds the snapshot inline).

### 6.3 Brief-view behaviour

- On load / on switching to the tab: `GET /api/context` (free; tiles) +
  `GET /api/brief` (free; renders cached brief if `fresh`). **No automatic
  POST** anywhere (credit guard).
- Changing the clinic point resets both (new cache key, same as today's context).
- Server restart loses the in-memory brief cache (brief re-generates on demand;
  snapshot cache also cold — existing behaviour).

## 7. API design

All three endpoints reuse `_resolve_clinic_point()` (400s: lat/lon without
partner, out-of-Singapore bounds, unknown postcode first digit). Clinic key =
`"%.3f,%.3f"` — identical to the snapshot cache keying.

### 7.1 `GET /api/brief` — free read, **never triggers a build or LLM call**

Response:
```json
{
  "status": "fresh | stale | none",
  "brief": {"headline": "...", "watch": [...], "outlook": "..."} | null,
  "last_error": "..." | null,
  "provenance_drops": 0,
  "as_of": "..." | null,
  "snapshot": {"state": "cached | building | none", "age_s": 12 | null},
  "clinic": "Woodlands Polycline",
  "cache_ttl_s": 900,
  "force_cooldown_s": 60
}
```
- `fresh` — a success brief exists and its `snapshot_as_of` equals the current
  snapshot's `meta.generated_at` → UI renders the brief immediately.
- `stale` — snapshot exists (cached or building) but no matching brief →
  tiles + "Generate Brief" CTA.
- `none` — no snapshot for this point yet → CTA enabled (POST will build).
- A live failure sentinel makes `last_error` non-null and `status` `stale`.

### 7.2 `POST /api/brief/generate?force=0|1` — **the only LLM path**

Guard order (short-circuit; each step before it costs 0 credits):

| Step | Condition | Result |
|------|-----------|--------|
| 1 | clinic point invalid | 400 |
| 2 | `chroma_db/` missing / no vectorstore | 503 "Knowledge base not ready" |
| 3 | success brief cached, fresh vs current snapshot, `force=0` | 200 `{"cached": true, ...brief}` |
| 4 | in-flight build for this key (`_brief_builds`) | `await` the same future → its result (0 extra calls) |
| 5 | `force=1` and `now − last_generation[key] < 60 s` | **429** + `Retry-After: <seconds remaining>` header, no LLM call |
| 6 | otherwise | run build (below) |

**Build (step 6, all in the thread-pool executor):**
snapshot (cache hit, or `build_snapshot` ~15–40 s on a miss — acceptable here:
explicit user action) → `list_corpus()` stats (local scan) →
`format_brief_prompt(snapshot, corpus_stats)` → `generate_brief(system, user)`
→ on success: cache `(monotonic, payload)` in `_brief_cache[key]`, record
`last_generation[key] = now`, 200:
```json
{"cached": false, "as_of": "...", "snapshot_as_of": "...",
 "brief": {"headline": "...", "watch": [...], "outlook": "..."},
 "provenance_drops": 0, "cache_ttl_s": 900}
```
→ on LLM exception: cache failure sentinel `{"error": str(e)}` with a **90 s
TTL** in the same dict, 502 `{"detail": "LLM request failed: ..."}`
(consistent with `/api/chat`).

**In-flight guard mechanics (D5):** before entering the executor, create
`fut = loop.create_future()`, store `_brief_builds[key] = fut`; on completion
`fut.set_result(...)` (or `set_exception`), and in `finally`
`_brief_builds.pop(key, None)`. A second concurrent POST hits step 4 and
`await fut` — it receives the same result or the same exception (→ 502).
Pattern mirrors `_context_builds` in `app.py` (lines 37–54), adapted from
"background build" to "shared synchronous build".

### 7.3 `GET /api/library` — free corpus inventory

One local Chroma metadata scan (0 credits; same scan pattern as
`list_pdfs()`/`_store_pdf_index()`). Response:
```json
{
  "totals": {"chunks": 3120, "guideline_pdfs": 96,
             "protocol_pages": 17, "public_guidance_pages": 1},
  "pdfs": [{"title": "...", "source": "upload:...|url", "chunks": 31,
            "doc_hash": "ab12cd34..."}],
  "web_sources": {
    "primarycarepages.sg": [{"url": "...", "chunks": 12, "derived_title": "..."}],
    "moh.gov.sg":          [{"url": "...", "chunks": 4, "derived_title": "..."}]
  }
}
```
- `derived_title`: last URL path segment, `%20`/`-` → spaces, title-cased.
  **The raw URL is always rendered beside it** (titles are cosmetic only).
- `GET /api/pdfs` remains unchanged (backward compatibility).

## 8. Brief generation internals

### 8.1 `context/brief.py` (new, stdlib-only)

`format_brief_prompt(snapshot, corpus_stats) -> (system, user)` — pure
functions, no I/O, testable with fixtures.

**`user` payload** = a **projection** (not a raw snapshot dump) built from the
snapshot:
```json
{"clinic": {"name","lat","lon","nea_region_approx","town?"},
 "generated_at": "...",
 "air":   {"psi_peak","psi_peak_region","pm25_peak"},
 "weather": {"today_date","today_high_c","today_low_c","flood_alerts"},
 "dengue":  {"clusters_active_total","nearby_clusters(≤3)","in_high_aedes_area"},
 "widb":    {"epi_week","date_range","notable","dengue_week","dengue_median_5yr",
             "flu_ili_positivity_pct"},
 "planning":{"window","healthcare_decisions_90d_count","examples(≤3)"},
 "polyclinics(≤3)",
 "data_gaps": [...],
 "corpus":  {"guideline_pdfs","protocol_pages","public_guidance_pages","total_chunks"}}
```
**`protocol_links` is structurally absent from this payload (D9).** A source
that failed (gap) simply omits its block.

**`system` prompt rules (verbatim requirements):**
1. Role: "You write a one-screen daily clinical briefing for a Singapore
   primary-care clinic from live population-level data."
2. The JSON provided is the **only** data source. Never invent values, places,
   or statistics. If a block is absent, treat that source as unavailable.
3. Live signals are **observations about the area** — never attribute them to
   clinic protocols, care protocols, or clinical guidelines. You are **not
   given** protocol names or guideline content, and you must **not mention
   "protocol" or "guideline" or any named care condition's protocol document**
   anywhere in your output. (Protocol relevance is rendered separately from
   structured data.)
4. A URA written permission is **NOT an opened facility** — never present one
   as an existing service.
5. `data_gaps` must be reflected: name the affected area in `outlook` or the
   relevant `watch` item ("unavailable this cycle").
6. If nothing is elevated, say so plainly (a calm week is a valid headline) —
   do not manufacture urgency.
7. Output **strict JSON only** (no markdown fences, no commentary):
   `{"headline": ≤25 words, "watch": [{"finding": ≤25 words,
   "why_it_matters": ≤30 words, "action": ≤30 words,
   "source": "NEA"|"data.gov.sg"|"CDA WIDB"|"URA"|"derived"}] (≤6 items,
   most clinically important first), "outlook": ≤40 words}`.

### 8.2 `rag.py generate_brief(system, user) -> dict`

- LLM: `ChatOpenAI(model_name=CHAT_MODEL, openai_api_base=OPENAI_BASE_URL,
  temperature=0, timeout=300, max_retries=1)` — same config as the chat chain.
  **Documented cost: one generate = ≤2 provider round-trips** (D8).
- Parse: strip optional ```` ```json ```` fences → `json.loads` → shape
  coercion (clip `watch` to 6 items; force str fields; missing field → `""`;
  non-list `watch` → `[]`). On failure: `{"parse_error": true, "raw": text}`
  (UI prose fallback).
- **Provenance drop guard (D9, server-side, ~10 lines):** forbidden patterns =
  `["protocol", "guideline"]` + the 17 names from `context/linkage.py
  PROTOCOLS` (imported, stdlib). For each `watch` item, case-insensitive
  substring match across `finding`/`why_it_matters`/`action`; a hit **drops
  the item** and increments `provenance_drops`. Documented trade-off:
  substring matching may over-suppress (e.g. "gout" inside another word) —
  deliberate; a false positive costs one UI line, a false negative violates
  the provenance golden rule.
- Returns `{"headline","watch","outlook","provenance_drops"}` or the
  parse-error dict.

### 8.3 Cost model (stated verbatim in UI microcopy + docs)

> One "Generate" = **≤2 provider round-trips** (chat model via OpenRouter,
> ~2–3 k tokens), **0 embedding calls, 0 retrieval**. Cache hit, a
> cooldown-blocked regenerate, a failure-retry within 90 s, and
> double-click / second-tab duplicates all cost **0** (15-min success cache,
> 90-s failure sentinel, in-flight guard). No path in this feature calls the
> embedding API or the retriever.

## 9. Library screen

**Layout (top → bottom):**
1. **Totals bar:** "N chunks · M documents: A guideline PDFs · B protocol pages
   · C guidance pages".
2. **Clinical guideline PDFs** (main section): card per PDF — title, source
   (hyperlink when a real URL), chunk count, short `doc_hash`
   (`title` attribute = full hash). The **Upload PDF** bar (moved here from the
   chat view; same `uploadPdf()` logic) sits at the top of this section.
3. **Care-protocol pages (primarycarepages.sg):** list of
   `derived_title — raw URL — N chunks`. **Collapse rule:** if the
   Task-1 pre-check (below) finds > 25 protocol URLs, this section renders as
   site-level totals + "N pages — see server log/curl for the list" instead of
   per-URL rows.
4. **Public health guidance (moh.gov.sg):** same list pattern.
5. Empty state (no `chroma_db/`): reuses the existing ingest-bar flow.

**Task-1 pre-check (0 credits):** before the Library UI is built (Task 4),
`curl localhost:5001/api/library` once and record the actual protocol-page URL
count and shape in the SDD ledger (context: `ingest_append`'s docstring notes
primarycarepages.sg's URL rewrite `chronic-care-protocols → chronic`, and a
depth-2 crawl can pull anchors/subpages). The collapse rule above applies based
on what is actually observed.

## 10. UI implementation notes

- **One file** (`static/index.html`), no framework, no build step — convention
  preserved.
- New CSS reuses the existing palette (`#1a56db` primary, `good/warn/bad`
  chip colours, card shadows/radii). Tiles: CSS grid
  `auto-fit minmax(150px, 1fr)`; brief cards: white cards with a left source
  accent; headline: 1.15–1.3 rem semibold.
- Nav: three buttons in a row under the header; active tab = filled style;
  hash change handler on `load` + `hashchange`; default `#/brief` when the hash
  is empty/unknown.
- JS additions: `switchTab()`, Brief state (tile render fn, brief render fn,
  `loadBriefStatus()`, `generateBrief(force)`), Library render fn. Existing
  chat/context functions unchanged. `node --check` must pass on the page
  script.
- **No client-side brief cache** — the server cache is the source of truth;
  the client re-GETs on tab show.

## 11. Failure & degradation matrix

| Failure | Behaviour | Credits |
|---------|-----------|---------|
| LLM 429 (credits exhausted) / timeout | 502 on the POST; 90-s error sentinel; tiles + last cached brief (if any) remain; UI shows `last_error` with the reason; **no auto-retry** | the failed attempt only |
| User retries during outage | sentinel served → 0 new calls for 90 s | 0 |
| Double-click / two tabs / impatient second press | in-flight guard → same future | 0 extra |
| Regenerate < 60 s after last | 429 + `Retry-After` | 0 |
| Non-JSON / malformed LLM output | prose fallback block; `parse_error` surfaced subtly | 1 (attempt) |
| Provenance guard hits | item dropped; count shown in footer | 1 (attempt) |
| Cold snapshot build (first point) | staged progress "Fetching live data… Ns"; total first-brief latency can be ~60–90 s | 0 (build) + 1 (LLM) |
| URA key missing / WIDB parse failure | gap in `data_gaps` → per-tile "unavailable" + brief names it | 0 |
| `chroma_db/` missing | `GET /api/brief` → `status: none`; POST → 503; Library → 503; ingest bar visible | 0 |
| Bad clinic point | 400 (all brief endpoints) | 0 |
| Server restart | brief + snapshot caches cold; everything re-builds on demand (existing behaviour) | as used |
| Concurrent library scans | read-only Chroma metadata reads; safe | 0 |

## 12. Verification & acceptance (no test suite — repo convention)

- **0-credit dry runs before any paid call:**
  1. `format_brief_prompt()` against a hand-written fixture snapshot → inspect
     the rendered prompt: protocol names absent, caveats present, JSON schema
     stated. (ad-hoc scratchpad script — **no permanent test files**)
  2. `generate_brief()` against 3 hand-written mock LLM responses (valid JSON,
     malformed JSON, prose refusal + a provenance-violating item) → verify
     parse/clip/drop/fallback behaviour.
- **Paid gate (exactly once):** `curl -X POST localhost:5001/api/brief/generate`
  → valid brief JSON; then one negative check: immediate forced regenerate →
  429 (0 credits).
- `node --check` on the page script; `GET /` → 200; `GET /api/status` →
  `ready: true`.
- **UI manual checklist (Task 3 gate, including the review hardening):**
  - Brief tab: tiles render from live snapshot; gap tile shows "unavailable".
  - Generate → staged progress → headline/Watch/Spotlight/Outlook render;
    Protocol spotlight matches `protocol_links.active` exactly.
  - Switch to **Chat** tab, trigger one `refreshContext()` tick (manual call) →
    **zero console errors** (ctx nodes present-but-hidden).
  - Reload the page at `#/library` directly → Library renders (hash routing).
  - Double-click Generate → only one LLM call (verify via response `cached` /
    timing or server log).
- Final standing checks (all tasks): server boots, `/api/status` ready, `/`
  200, `venv/bin/python -m context` exit 0, `git status` shows only the
  intended commits.

## 13. Risks & mitigations (top risks from the review, carried into gates)

1. **Credit burn from concurrent/rapid generates** → D4 two-phase + D5
   in-flight guard + D6 cooldown + D7 negative cache; Task 2 gate includes the
   429 check and a double-click UI check.
2. **Clinical mislabeling via LLM prose** → D9 (prompt input stripped of
   protocol material + structural spotlight + server-side drop guard); Task 2
   dry run #2 covers a violating mock.
3. **Paid action firing from browser mechanics** (tab reopen, back/forward,
   hash reload) → only a **POST** can trigger the LLM; page load and tab
   switches only ever GET (free).

## 14. Limitations (accepted)

- Protocol-page titles in Library are **derived from URLs** (no title stored in
  chunk metadata; a re-ingest to store titles is out of scope — it would cost
  the full corpus re-embed). Raw URLs always shown.
- The provenance drop guard may over-suppress (documented, intentional).
- Brief is per clinic point (default: test clinic Woodlands Polycline, env
  `CONTEXT_LAT/LON/NAME`); WIDB/URA signals are national/island-wide by nature
  and labelled as such.
- `list_corpus()` does a full O(n) metadata scan per Library load (same class
  of accepted debt as the existing `list_pdfs()`; ~3 k chunks — fine at this
  scale, noted for the Outstanding list if the corpus grows 10×).
- Brief cache is memory-only (lost on restart) — acceptable: regeneration is
  one cheap call.

## Appendix A — External review disposition (2026-09-09)

All 10 findings accepted and incorporated:

| Finding | Severity | Where it landed |
|---------|----------|-----------------|
| No in-flight guard on paid path (stampede) | blocker | D5, §7.2 step 4, §13.1 |
| No regenerate cooldown | blocker | D6, §7.2 step 5, §12 |
| Failure not negatively cached | major | D7, §7.2, §11 |
| `max_retries=1` ⇒ ≤2 round-trips | major | D8, §8.2, §8.3 |
| Prompt-only provenance safety | major | D9, §8.1 rule 3, §8.2 guard |
| `data_gaps` footer-only | major | D10, §6.1 gap states |
| `GET` as paid trigger | major (API) | D4, §7.1/§7.2 split, §13.3 |
| "Verbatim reuse, zero risk" overstatement | minor | §5 DOM rules, §12 UI gate |
| URL-derived titles likely noisy | minor | §9 pre-check + collapse rule |
| Single-curl Task 1 gate optimistic | minor | §12 0-credit dry runs, D11 |

Reviewer-confirmed strengths kept intact: free tile layer client-side (0
embedding/retrieval cost), Protocol spotlight from structured data (inherits
basis-tag guarantee), single-file hash-routed tabs (no new page/build step).






