# CLAUDE.md — CP_RAG

RAG chat app over Primary Care SG chronic-care care protocols + MOH public health guidance (Phase 2). FastAPI serves a single-page UI (`static/index.html`); `rag.py` scrapes the source sites (`SOURCES`), embeds chunks into a persisted Chroma store, and answers via `deepseek/deepseek-v4-flash-0731` (paid, fast) routed through **OpenRouter** (OpenAI-compatible gateway; provider + models are env-configured, see `.env.example`). The RAG prompt has four labelled sections — *Protocol content* / *Clinical guidelines* / *Public health guidance* / *Local context* (live snapshot, provenance rules in `docs/signal-research.md` §12–13).

Read this file at the start of every session. Make **one focused change per session**, verify it boots (see Commands), then stop and let me commit.

## Golden rules (do not violate)

- **OpenAI credits are a budget, not infinite.** The account has hit 429 `credit_balance_exhausted` before. Every `/api/chat` costs an embedding call + a chat-model call; `POST /api/ingest` re-embeds the *entire* corpus (fresh-directory use only); `POST /api/ingest/append` is the cheap path — it skips sites already in the store (host-level) and embeds only new sources, with a 500-chunk refusal cap (OpenRouter's 300k-token/request limit). Keep test API calls to the bare minimum — prefer `/api/status` (free) over chat probes, and never loop or retry against the API. The clinician brief (`POST /api/brief/generate`) is the only other LLM path: one generate = 1 chat-model call (≤2 provider round-trips), 0 embeddings, 0 retrieval, 15-min success cache, 60-s force-regenerate cooldown (429), 90-s failure sentinel (502); `GET /api/brief` is free and never builds.
- **Secrets live in `.env` only** (gitignored). Never print a full key, never commit it, and keep `.env.example` in sync with the variables the code reads. `.env` is machine-parsed: exactly one `KEY=value` per line, no commentary — a stray pasted line breaks the parse with warnings on every import.
- **`chroma_db/` is a persisted artifact, not source.** It is gitignored and is expensive to rebuild (full scrape + full re-embed). Don't delete it casually, don't commit it, and don't re-ingest "just to test". If the app returns 503 "Knowledge base not ready", check that `chroma_db/` exists before touching ingest.
- **Python 3.9 only** (system `/usr/bin/python3`). No 3.10+ syntax. The venv lives in the repo root; if this folder ever moves, the venv's shebangs break silently — recreate it (`python3 -m venv venv && venv/bin/python -m pip install -r requirements.txt`) rather than patching it.
- **The scrape target is a live external site** (`SOURCE_URL` in `rag.py`). If the site changes, ingest can silently return junk or zero chunks — always check the `chunks` count in the `/api/ingest` response before trusting the result.

## Stack (decided — don't re-litigate)

- **FastAPI** + **uvicorn** (async endpoints, `lifespan` builds the RAG chain at startup if `chroma_db/` exists).
- **LangChain 0.3.x** (`langchain-community`, `langchain-openai`) — chain: retriever (k=12 since Phase 5's guideline corpus) + `ChatPromptTemplate` + chat model (env `CHAT_MODEL`, currently `deepseek/deepseek-v4-flash-0731` via OpenRouter, temperature 0).
- **Chroma** persisted at `./chroma_db`, embeddings `text-embedding-ada-002` (env `EMBED_MODEL`; currently served via OpenRouter). Changing the embedding model invalidates the index — re-ingest.
- **BeautifulSoup** text extraction, chunking 1000/200. PDFs (Phase 5): `pypdf` text extraction + `ingest_pdf()` (append-only, sha256 dedupe) for clinical-guideline PDFs (`GUIDELINE_SITES` routing → prompt section *Clinical guidelines*).
- UI is one static HTML file talking to ten JSON endpoints (incl. the Phase 5 guideline-PDF upload bar and the Phase 6 brief API + library inventory). No build step, no framework. Phase 6: three hash-routed tabs — `#/brief` (default, clinician dashboard: 6 KPI tiles + generated brief + Protocol spotlight rendered only from structured `protocol_links.active`), `#/chat`, `#/library` (corpus inventory; the PDF upload bar lives here). The context strip survives in the Brief view as the collapsed "Raw context data" drill-down — its DOM nodes are never removed, the 15-min refresh intervals depend on them.
- **Context layer** (Phases 0–4, done): `context/` package (stdlib-only) builds a live population-level snapshot from key-free data.gov.sg feeds; `context/widb.py` (Phase 3) fetches the newest CDA Weekly Infectious Diseases Bulletin PDF (archive crawl → newest-to-oldest walk-back → `pypdf` table/narrative parse, 3-day disk cache) and adds the `disease_week` block (national dengue/influenza/ARI/HFMD counts) to the snapshot; `context/ura.py` (Phase 4, needs `URA_ACCESS_KEY` in .env) fetches the URA daily token + `Planning_Decision` rows (90-day `last_dnload_date` window) and filters healthcare-related written permissions into the `catchment_change` block (island-wide; a written permission is NOT an opened facility; 24 h disk cache; no key → `ura:` data gap, the signal degrades like any other source); `GET /api/context` serves it with a 15-min in-memory cache (cold builds take ~15–40 s); the UI strip at the top of `static/index.html` renders it (WIDB chip + "Infectious diseases (WIDB, national)" card, URA 90d chip + "Planning decisions (URA, island-wide)" card) and auto-refreshes every 15 min. Every active protocol link is provenance-tagged (`basis`: corpus/partial/derived/none + `basis_note`, from the corpus audit of 2026-09-08 — `docs/signal-research.md` §12) and rendered as a badge on the strip. **The RAG prompt never presents `derived` links as protocol content** — `context/prompts.py` enforces the split: corpus/partial basis notes → *Protocol content*; all live signals (incl. the URA planning line, labelled "not protocol content" / "not an opened facility") → *Local context*; moh.gov.sg chunks → *Public health guidance*. Clinic point for the prompt: `CONTEXT_LAT`/`CONTEXT_LON`/`CONTEXT_NAME` env (default: test clinic in `context/config.py`); chat reuses the 15-min cache and never blocks on a build. **Phase 6 (done 2026-09-09):** `context/brief.py` (pure, stdlib-only) projects the snapshot into a brief prompt — `protocol_links` is structurally absent, so protocol material can never reach the brief LLM; `rag.py` `generate_brief()` / `_parse_brief()` / `_apply_provenance_guard()` add the single chat-model call + strict-JSON parse + server-side drop of watch items mentioning protocols/guidelines (counted as `provenance_drops`); the brief cache is in-memory (server restart = cold). Details: `docs/signal-research.md` §16.

## Repo layout

```
CP_RAG/
  app.py            # FastAPI app: /, /api/chat, /api/ingest, /api/ingest/append, /api/ingest-pdf, /api/pdfs, /api/status, /api/context, /api/brief, /api/brief/generate, /api/library
  rag.py            # scrape → chunk → embed → persist; retriever + chain; ingest_pdf() for guideline PDFs; list_corpus() (Phase 6) + generate_brief()/_parse_brief()/_apply_provenance_guard()
  static/index.html # Phase 6: three hash-routed tabs — Brief (default: KPI tiles + generated brief + spotlight + "Raw context data" drill-down) / Chat / Library (PDF upload bar + corpus inventory)
  context/          # live SG population-context layer (Phase 0–4; stdlib-only, key-free except ura.py which uses URA_ACCESS_KEY; widb.py also uses pypdf) + brief.py (Phase 6: pure brief prompt builder, stdlib-only)
  scripts/          # ace_guidelines.py (Phase 5 optional crawler; stdlib + requests + bs4)
  docs/             # signal-research.md (data sources, phase plan)
  chroma_db/        # persisted Chroma store (gitignored, expensive to rebuild)
  requirements.txt
  .env.example      # keep in sync with .env variable names
  .env              # gitignored — OPENAI_API_KEY
  CLAUDE.md
  HANDOFF.md        # session state — read/update each session
```

## Conventions

- All config via env; `.env.example` stays current. No secrets in code or logs.
- Keep endpoints small. Long blocking work (the LLM call in `/api/chat`, the ingest crawls) runs in a thread-pool executor — never call blocking chain/ingest code directly in an `async def` endpoint, or the event loop freezes and the whole server stops responding (no background jobs needed yet).
- No test suite exists. The standing check is: server boots, `/api/status` returns `{"ready": true}`, and `/` serves 200.

## Commands (keep these working)

- `venv/bin/uvicorn app:app --port 5001` (or `python app.py`, same default) — run the app, then open http://localhost:5001
- `curl -s localhost:5001/api/status` — free readiness check (run this, not chat, when verifying a change)
- `curl -s localhost:5001/api/context` — live snapshot (15-min cache; first hit builds, ~15–40 s, key-free)
- `curl -s localhost:5001/api/brief` — free brief status/cache (never builds; `status` is fresh/stale/none)
- `curl -X POST "localhost:5001/api/brief/generate"` — generate the clinician brief (**costs 1 chat-model call** — the only LLM path outside `/api/chat`; 429 if force within 60 s of the last generation)
- `curl -s localhost:5001/api/library` — free corpus inventory (PDFs + web sources by site + totals)
- `curl -X POST localhost:5001/api/ingest` — rebuild the knowledge base (**costs API credits; fresh directories only**)
- `curl -X POST localhost:5001/api/ingest/append` — append-only ingest: embeds only sources not in the store (host-level skip + 500-chunk cap); cheap and idempotent
- `venv/bin/python -m pip install -r requirements.txt` — (re)install deps

## Not now (scope guard)

Streaming responses, auth, Docker, a test suite, and swapping models/embedding providers are parked. Don't pull them in; ask first. (Multiple source sites is implemented — `SOURCES` list in `rag.py` — but adding another site means crawling it once via append ingest; keep the list short.)