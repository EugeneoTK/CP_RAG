# HANDOFF — CP_RAG (updated 2026-09-08, Phase 5 — ACE clinical-guidelines PDF ingestion + post-completion code review)

## Independent code review + fixes (2026-09-08, after Phase 5 plan completion)
An independent review of the full Phase 5 diff (`62bf8ed~1..187589c`, not just the per-task self-reviews) found 6 issues the task reports missed. Fixed the one security-relevant one plus a stale-copy bug; logged the rest as outstanding (below) rather than scope-creeping into an unplanned refactor.
- **Fixed — stored XSS in chat sources** (`static/index.html`): `sources` were rendered into `innerHTML` with no escaping (`<a href="${u}">${u}</a>`), unlike the answer text on the line above which does go through `escapeHtml`. This was latent but harmless while `source` was always a hardcoded scrape URL; Phase 5 made it exploitable because `doc_title` (shown in `sources` for PDF chunks) is attacker-controlled — anyone can upload a PDF via `POST /api/ingest-pdf` with a script-tag title/filename, and it fires in every browser that later asks a question retrieving that chunk. Fix: both the `href` and link text now go through `escapeHtml`.
- **Fixed — stale loading-hint copy** (`static/index.html`): "Thinking… (free-tier model: can take a few minutes)" was left over from the pre-Phase-5 free model; Task 0 (`62bf8ed`) switched to the paid `deepseek-v4-flash-0731` (~8 s responses) in the same commit but didn't update this string. Simplified to "Thinking… Ns" (no model-tier claim, so it can't go stale again on the next model swap).
- Verified: `node --check` on extracted `<script>` block OK; `/api/status` → `ready:true`; `/` → HTTP 200.

## Done this session (Phase 5 — full plan `docs/superpowers/plans/2026-09-08-ace-guidelines-pdf.md`, all tasks committed)
- **Task 0** — `CHAT_MODEL` → `deepseek/deepseek-v4-flash-0731` (paid, ~8 s/chats vs 4–5 min free tier); prior fixes + Phase 5 plan.
- **Task 1** — `pypdf` extraction: `MAX_PDF_BYTES` (25 MB) + `pdf_text()` in `rag.py`; live smoke extracted 13,945 chars from the real ACE MRI ACG before any credit spend.
- **Task 2** — `GUIDELINE_SITES`, append-only `ingest_pdf()` (sha256 + source-URL dedupe) + `list_pdfs()`; MRI ACG ingested as 19 chunks, duplicate ingest skipped.
- **Task 3** — `POST /api/ingest-pdf` (multipart, 25 MB cap, `%PDF-` magic check, executor) + `GET /api/pdfs`; `python-multipart` added; all four live curl checks passed.
- **Task 4** — 4th prompt section *Clinical guidelines*; `_prepare_sections()` now routes `GUIDELINE_SITES` chunks there (never as protocol content); chat sources show `doc_title` for PDFs. Live probe: MRI question answered from the ACG, source = PDF title, no markdown.
- **Task 5** — UI "Clinical guidelines" upload bar + ingested-PDF count (`static/index.html`); `node --check` on the page script; endpoints return exactly what the bar renders.
- **Task 6** (user-approved full seed) — `scripts/ace_guidelines.py` (sitemap → 29 detail pages → 98 PDF links → download → `ingest_pdf`); dry run found 29 pages / 98 PDFs (plan estimated 50–60 — extras are reference lists, EG details, patient-education aids, appendices). Full seed: **95 ingested, 2 deduped, 1 failed** (`Appendix 2 - National One Rehab Framework.pdf` — scanned image, no text layer; would need OCR, out of scope). Two plan deviations, both necessary: `sys.path` insert for `import rag` and `load_dotenv()` before import (`rag.py` never loads `.env` itself — first run failed 100% on missing credentials).
- **Retrieval fix** — first post-seed probe (allergic rhinitis diagnosis) showed k=5 surfaced only title/intro chunks of a 36-chunk ACG; raised retriever k 5 → 12 (commented in `rag.py`); re-probe on-target (symptom-based diagnosis, ocular-symptom cue, ACG cited).
- **Store state** — 96 guideline PDFs / 2,975 guideline chunks (plus the pre-existing protocol + MOH corpus); `GET /api/pdfs` verified after restart.
- **Docs** — Phase 5 row in `docs/signal-research.md` §9; `CLAUDE.md` (four prompt sections, k=12, PDF stack bullet, endpoints, `scripts/` layout); this file.

## In progress
- None.

## Outstanding
1. 1 scanned PDF uningested (rehab appendix) — OCR would be needed; revisit only if that appendix matters.
2. Revisit if ACE rehosts its PDFs or changes `sitemap.xml` — `venv/bin/python scripts/ace_guidelines.py --list` (free) shows drift; `--ingest` re-runs are idempotent (sha256 dedupe, 0 credits for known files).
3. k=12 is a corpus-size heuristic — revisit if answers over- or under-retrieve as the corpus grows.
4. Refreshing primarycarepages content still requires a full `ingest()` rebuild (URL rewrite defeats host-level append skip) — do it on purpose, not to test.
5. Postcode geocoding is district-centroid approx (OneMap DNS-blocked on this network); `--lat/--lon` is the exact path.
6. From the 2026-09-08 post-completion review, not yet fixed (all pre-existing patterns Phase 5 extended, none security-relevant like the XSS above):
   - `app.py` `/api/ingest-pdf`: the 25 MB cap is checked only after `await file.read()` has already buffered the full body — doesn't bound upload cost, just rejects after the fact.
   - `rag.py` `ingest_pdf()`: hash/URL dedupe is check-then-act with no lock — two concurrent uploads of the same PDF can both pass the check and both get embedded (double spend, duplicate rows). Same TOCTOU shape as the pre-existing `ingest_append()`.
   - `rag.py` `list_pdfs()` / `_store_pdf_index()`: full metadata scan of the whole Chroma store on every call (GET /api/pdfs, every upload's dedupe check) — O(n), no caching; fine at ~3k chunks, will slow as the corpus grows.
   - `rag.py` `_store_pdf_index()` duplicates `_store_url_index()`'s scan loop instead of sharing one helper.

## Next-session prompt
Project: CP_RAG at `/Users/ugeneo/Documents/Project Codes/CP_RAG` — FastAPI + LangChain RAG Q&A over Singapore primary-care chronic-care protocols **plus the full ACE clinical-guidelines repository (96 ACG PDFs, 2,975 chunks; `source_site` metadata routes them to a dedicated *Clinical guidelines* prompt section — never cited as protocol content)** + MOH public guidance + live local-context signals. LLM `deepseek/deepseek-v4-flash-0731` + `text-embedding-ada-002` via OpenRouter (env-configured in `.env`, git-ignored — never commit it). Golden rules in `CLAUDE.md`: credits are a budget (prefer `/api/status` over chat probes), Python 3.9, `chroma_db/` is expensive — don't rebuild; new guideline PDFs go through `POST /api/ingest-pdf` (append-only, sha256 dedupe) — the UI has an upload bar, and `scripts/ace_guidelines.py --ingest` re-syncs the ACE repo. Corpus audit + phase table: `docs/signal-research.md` §9/§12/§13. Read `CLAUDE.md` first.

State: Phases 0–5 DONE, verified live, committed (Phase 5 = top commits in `git log`; SDD ledger + task reports in `.superpowers/sdd/2026-09-08-ace-guidelines-pdf/`). Server running detached on port 5001 (log: `/tmp/cp_rag_server.log`). Test clinic: Woodlands Polycline (lat 1.4309, lon 103.7752, PC 738579). Corpus: protocols + MOH + 96 ACE guideline PDFs (~2,975 guideline chunks).
