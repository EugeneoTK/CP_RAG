# HANDOFF — CP_RAG (overwritten 2026-09-15, 8th session)

## Done this session

**Ecosystem Insights (Track A usage layer)** — the "listening system" from
the IMDA/OGP two-track framing (Track A = GP e-Protocol Bag/PPC WS3; Track B
= governance sandbox — deliberately NOT built):
- `usage/` package (stdlib-only): append-only daily JSONL (`usage/*.jsonl`,
  gitignored data, module tracked). One `chat` line per turn (query_id,
  question, provider, answer_chars, disclaimer flag, sources) + `feedback`
  lines. `log_chat`/`log_feedback` never raise. `insights()` aggregates
  server-side — **free: file reads only, 0 credits**.
- Endpoints: `POST /api/feedback` (rating good/bad; bad takes a tag:
  missing/wrong/unclear/other; last rating per query_id wins),
  `GET /api/insights?days=N` (clamped 1..365).
- UI: 4th always-visible tab **Ecosystem Insights** (`#/insights`): 4 KPI
  tiles (queries 30d, good-rating %, gap-flagged, top-cited) + Top questions /
  Gap signals (clinician-flagged + "I don't know" disclaimers) / Most cited
  sources. 👍/👎 under each bot answer; 👎 reveals the tag picker, selecting a
  tag submits.
- **Committed + pushed `b2bd7ca`** (verified by independent reviewer; its 2
  findings fixed: mis-highlighted button on first 👎, window off-by-one).

**PSI/PM2.5 data-basis fixes** (user caught via haze.gov.sg cross-check):
- **BUG (critical for a clinical brief):** `fetch_nea_reading` picked "the
  first dict in readings" — for the psi feed that was `co_sub_index`
  (carbon monoxide), so the app showed "PSI 8 Good" while the official 24-hr
  PSI was 153. Fix (`1d62834`): `READING_KEYS` maps explicit metrics —
  psi → `psi_twenty_four_hourly` (spot sub-index fallback); pm25 feed carries
  ONLY `pm25_one_hourly` (labelled "1-hr" in UI; sub_index is a PSI
  contribution index, never a µg/m³). `basis` key surfaced in snapshot.
  Haze guidance (MOH page in corpus) and the 101/201 thresholds are
  24-hr-PSI-based — everything is now consistent.
- **Clinic-region semantics** (`e487a02`): the brief is for THIS clinic's
  patients. Air tiles/chips/brief-prompt now lead with the clinic's NEA
  region (`clinic_value`/`clinic_region`, from existing `nea_region`
  heuristic); island peak is the sub-line ("island peak 154 central").
  Protocol linkage TRIGGERS on the clinic region (verified: clean clinic
  region + elevated island peak → no haze triggers).
- UI labels: tile "Air (PSI 24h)", chips "PSI (24h)" / "PM2.5 (1h)";
  brief system prompt rule 4 names region + window.

**Brief view layout cleanup** (user: "cluttered, use the white space"):
- `33203df`: header content wrapped in a `max-width:800px margin:auto`
  container (was full-bleed, left edge didn't align with the centred body);
  Brief tiles 4-col → **3-col** (6 tiles = two full rows; 4-col left a
  half-empty 2nd row); tile sub-text line-clamped to 3 (long URA/dengue
  lists), pinned to tile bottom, full text on hover `title`;
  `236f106`: Insights' 4 tiles keep their own 4-col row (shared
  `.brief-tiles` class); `3ad8c7d`: brief title breathing room + Generate
  button no longer touching the clinic inputs.
- `c8fe1ff`: CSS bug — `.tag-pick { display:inline-flex }` overrode the
  `hidden` attribute, so the 4 tag chips rendered on EVERY answer without a
  👎 click. Added `.tag-pick[hidden] { display:none }`.

## Outstanding

1. **App naming (user request — decide in the NEXT session):** "Care
   Protocol Assistant" / "Powered by HSG Primary Care Pages" → rename
   pending (strings: `static/index.html` header + `<title>`; check docs).
2. **Bare-URL answers**: the chat sometimes answers with a lone link (seen:
   `moh.gov.sg/others/haze/` as the whole answer) — prompt-quality follow-up
   (the plain-text-only rule may make the model lazy on link-heavy sources).
3. **Provider does not survive restart**: boot default `openrouter`; re-set
   local with `POST /api/provider` or pin `CHAT_PROVIDER=local` in `.env`.
   (Server was left on `local` = comp9:gpu0-vllm at handoff.)
4. Carry-over items from the 7th session that are still open: AAP
   dashboard/prompt-only (not in brief); NTUC calendar PDF parse noise
   (~186/386 chunks); URA street-hint coverage is partial by design; tile
   count vs drill-down 50-cap ("showing 50 of N" note); "approvals" wording
   (permission ≠ opened facility).

## Next-session prompt

CP_RAG is a FastAPI RAG assistant for SG primary care (uvicorn,
127.0.0.1:5001, log `/tmp/cp_rag_uvicorn.log`; startup ~2 min while the
context snapshot builds; **restart resets provider to openrouter — re-set
local**). Corpus: 96 ACG guideline PDFs + 17 primarycarepages.sg protocols
+ MOH public guidance + NTUC Active Ageing calendars in Chroma
(`chroma_db/` — expensive, don't rebuild). Live local-context signals via
the key-free `context/` package (NEA air — now the correct metrics, see
READING_KEYS in `context/fetchers.py`; PM2.5 is 1-hourly, PSI is 24-hr;
clinic's region primary / island peak secondary — see `context/snapshot.py`
air block + `context/linkage.py`), WIDB, URA planning, NTUC Active Ageing.
UI: four hash-routed tabs — Brief (default) / Clinical Flight Bag (chat,
with rating buttons) / Library / **Ecosystem Insights** (usage JSONL under
`usage/`, gitignored). CHAT LLM runtime-switchable: `openrouter` (paid) /
`local` vLLM. All committed + pushed: **HEAD = origin/main = `c8fe1ff`,
working tree clean.** Do next: **decide the app naming with the user
(Outstanding #1)**; then optionally the bare-URL answer fix (#2) or pin
`CHAT_PROVIDER` (#3). Golden rules in `CLAUDE.md` (read it first). IMDA
meeting context: this app is the Track A "GP e-Protocol Bag" prototype for
the two-track programme — the Ecosystem Insights tab is the "listening
system" slide evidence.
