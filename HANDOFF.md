# HANDOFF — CP_RAG (overwritten 2026-09-10, 6th session)

## Done this session

**Phase 9 — 2 km catchment + Active Ageing Programmes card, implemented +
live-verified** (user request: the 90-day planning view now uses a 2 km
radius, its title is layperson-friendly, and the dashboard gains an
Active Ageing Programmes card for NTUC centres within 2 km):
- `context/config.py`: `URA_NEAR_KM` 10 → **2.0** (the "near" band of the
  URA planning view now = walkable catchment). New NTUC block:
  `NTUC_CALENDAR_URL`, `NTUC_UA` (browser UA — ntuchealth.sg is
  Akamai-fronted, non-browser UAs 403, same wall class as WIDB),
  `NTUC_CACHE_TTL_SECONDS = 24 h`, `ACTIVE_AGING_NEAR_KM = 2.0`, and
  `ACTIVE_AGING_CENTRES` — 27×(name, lat, lon) **verified live
  2026-09-10** from ntuchealth.sg/active-ageing/locations (the Next.js
  flight payload embeds per-centre `position`; no geocoding needed).
  Names match the calendar-page anchor text exactly.
- `context/community.py` (NEW, stdlib-only, key-free, mirrors ura.py):
  fetches the calendar landing page, parses the
  `assets.ntuchealth.sg/ae/<centre>-<Mon>-<YYYY>.pdf` anchors →
  `[{name, month, pdf_url}]` (dedup by name), disk-cached 24 h at
  `config.CACHE_DIR/ntuc_ageing_v1.json`. Payload carries `count`,
  `calendar_months`, `caveat`; the caveat dynamically detects the known
  NTUC site quirk where 'Bukit Batok West' reuses the Bedok-North PDF
  (confirmed live 2026-09-10).
- `context/snapshot.py`: new `active_ageing` block (fetched after the URA
  block, `time.sleep(2)`); `_enrich_ageing()` adds `km` (1 dp, haversine
  to the NTUC-published centre position — exact, unlike the URA
  street-heuristic) + `near` (≤ 2 km) per centre and `near_count`; a
  live centre missing from the config table renders without a km (still
  listed, never `near`). Failure → `data_gaps: "active_ageing: ..."`,
  never a crash.
- `static/index.html`: Planning tile RENAMED `Planning 90d` →
  **`New health facilities (3 mo)`** (value now "N approvals"), sub
  "k near (~10 km)" → "k near (~2 km)"; NEW 6th tile **`Active Ageing
  (2 km)`** (value "N centre(s)", sub = names + km within 2 km, or
  "none within ~2 km"; warn tile on gap); NEW raw-context card
  **`Active Ageing (NTUC Health, island-wide)`** — all centres
  nearest-first, within-2 km bolded, each with a link to the current
  month's calendar PDF + the caveat line; NEW chip "Active Ageing: N
  within 2 km". Drive-by fix: the raw-context "Nearest polyclinics"
  list read `nearest_services.pyclinics` (typo) and never rendered —
  now `polyclinics`.
- `context/prompts.py`: Local context gains an NTUC line — N centres
  island-wide (calendar month), near ones named with km, "NON-clinical",
  calendars answered from the Community resources corpus section, never
  as clinical services. URA line "~10 km" → "~2 km".
- `context/brief.py`: drive-by fix — `_project()` read
  `ns.get("pyclinics")` (typo) so the brief prompt's polyclinics were
  always `[]`; now `polyclinics`. AAP is deliberately NOT in the brief
  projection (brief stays clinical; the system-prompt source whitelist
  has no NTUC — see Outstanding).
- `context/__main__.py`: PLANNING near-count "~10 km" → "~2 km"; new
  AGEING CLI section (count, calendar month, all centres nearest-first,
  `*within 2 km*` marked, unmapped flagged).
- Docs: `docs/signal-research.md` §19; `CLAUDE.md` (6 tiles, AAP note).
- Gates (all 0 paid credits; `node --check` on the page JS; py_compile
  all touched modules; server restarted for live checks):
  - Live unit check (one real NTUC fetch): 27 centres parsed, month
    "Sep 2026", live-name set == config set (0 unmapped, 0 orphans);
    near_count = 3 at the test clinic (Jurong Central Plaza 0.8,
    Boon Lay 1.1, Taman Jurong 1.2 km); next-closest Gek Poh 2.2 and
    Pioneer 2.3 km (correctly excluded from the 2 km band).
  - Full `python -m context` build + `GET /api/context` live: see
    /tmp/ctx_build_p9.log and docs/signal-research.md §19.

## Done this session (continued — misleading-tile fix)

User flagged the **New health facilities (3 mo)** tile as "utterly
misleading": the sub-line mixed a distance filter ("1 near (~2 km)")
with island-wide category counts and gave no idea WHERE the 5 senior
care were. Verified against the live cache: the 1 near row is 182 Jalan
Jurong Kechil (child care, ~1.2 km) — ZERO senior care within 2 km; the
3 listed senior-care rows are Tampines (~25 km) + 2 unmapped streets
(Joo Seng Rd, Ang Mo Kio Ave 3), and 2 of the 5 senior-care rows are
not visible in the UI at all (50-row list cap vs 67 counted).
- `context/snapshot.py::_enrich_ura`: now also emits
  `nearest_km_by_category` — min `approx_km` per category over MAPPED
  rows only (a category with no mapped rows gets no entry; a distance
  is never invented).
- `static/index.html` tile: each category bit gains `(nearest ~X km)`
  (clinic/medical bit = min over Clinic+Medical+Other buckets). Live
  sub-line now: "1 near (~2 km) · 5 senior care (nearest ~25 km) ·
  6 nursing home (nearest ~19.9 km) · 12 child care (nearest 1.2 km) ·
  41 clinic/medical (nearest ~13.4 km) · 3 polyclinic (nearest ~3.5 km)".
- `CLAUDE.md` context-layer bullet lists the new key.
- Gates: `_enrich_ura` run against the live `ura_planning_v2.json`
  cache (assertions: SC nearest = min over mapped SC rows;
  `near_clinic_count` unchanged = 1); `node --check` on the page JS OK.

## Done this session (continued — community programme retrieval fix)

User asked the chat for Active Ageing Centre programmes and got no
specifics. Ingestion was fine (386 Sep-2026 calendar chunks in the
corpus, specific programme names present — verified in-store). Two
retrieval bugs in `rag.py`:
- **`_COMMUNITY_INTENT_RE` boundary bug**: one trailing `\b` on the whole
  group silently disabled every prefix alternative — `activit\b` never
  matched "activity/activities", `communit\b` never matched
  "community/communities". The commonest community phrasings fell
  through to protocol steering, the 386 calendar chunks were crowded
  out of the k=12 top, the model saw an EMPTY Community resources
  section and answered "I don't know". Fix: per-alternative boundaries
  (full-word alternatives keep `\b`, prefix alternatives don't) + added
  AACC/AAC, ageing/aging centre, calendar, leisure. Verified: 11/11
  community phrasings match, 5/5 clinical phrasings don't
  (`/tmp/cp_rag_regex_check.py`).
- **Skipping steering was necessary but not sufficient**:
  "What does the AACC near our clinic provide?" embeds closer to the
  primarycarepages partner page + protocol chunks — all 12 slots
  non-calendar, model answered the generic AAC role (ABC/2Ss) with zero
  programme names. Fix: `build_chain` now runs a second retrieval
  scoped to `COMMUNITY_SITES` (`$in` filter, k=4) for community-intent
  questions and merges up to 4 calendar chunks (deduped by content,
  total capped at 16).
- Gates: `py_compile` OK; server restarted (now PID 27565) and
  e2e-verified live — "What does the AACC near our clinic provide?"
  now retrieves NTUC calendar chunks (Kampung Admiralty, Telok
  Blangah 2026-09) and the answer names specific programmes (Kpop,
  Qigong, Pilates, Zumba, walking football, ukulele, 3D printing
  workshop, TCM Mobile Clinic). Clinical regression check:
  "How do I manage hypertension in an 80-year-old?" still retrieves
  ACG + protocol chunks with steering active, no community chunks.
- Known soft spot: ~186 of the 386 calendar chunks contain PDF
  cover-page/design-template noise (Pantone swatches, "Housekeeping")
  from the NTUC template pages — the model tolerates it, but a cleaner
  parse (skip template pages in `ingest_community_refresh`) is a
  candidate improvement at the next monthly refresh.

## Outstanding

1. **Uncommitted work (user's call — commits left to the user):**
   Phases 7–9 + provider toggle were committed and PUSHED as `b4b0a7a`
   (2026-09-10). Now uncommitted: the per-category nearest tile fix —
   `context/snapshot.py`, `static/index.html`, `CLAUDE.md` — plus the
   community-programme retrieval fix — `rag.py` (this session).
   Suggested commits: `feat: per-category nearest distance on New health
   facilities tile (nearest_km_by_category)` and `fix: community-intent
   gate + calendar quota for NTUC programme questions (rag.py)`.
2. **Provider does not survive restart**: runtime state is in-memory;
   boot default is `openrouter`. Server restarted this session; if it
   came back on `openrouter`, re-set local with `POST /api/provider`
   (free, config-only) or pin `CHAT_PROVIDER=local` in `.env`.
3. **AAP is dashboard/prompt-only, not in the clinician brief**: the
   brief's source whitelist ("NEA | data.gov.sg | CDA WIDB | URA |
   derived") has no NTUC and the projection omits `active_ageing`. If
   the brief should mention nearby Active Ageing centres, add "NTUC" to
   the whitelist + a small projection block (the data is already in the
   snapshot).
4. **Calendars track the published month** (Sep 2026 at write time).
   NTUC updates the page around the start of each month; the next
   `POST /api/community/refresh` pulls the new month and replaces the
   old (no history retained — by design). The Phase 9 card reads the
   SAME landing page, so it always reflects the published month.
5. **URA street-hint coverage is partial by design** (19/50 listed rows
   mapped at Phase 8 build time; the 2 km band is stricter, so expect
   fewer near rows). Consequence: `near_clinic_count` AND
   `nearest_km_by_category` are LOWER BOUNDS — unmapped addresses
   silently drop out of both, and distances are to the area CENTROID,
   not the facility (±km either way). Extend `config.STREET_AREA_HINTS`
   when a recurring unmapped street matters.
6. **NTUC centre table is static** (verified 2026-09-10). A new centre
   appears in the card with "(no coordinates on file)" and no km — add
   it to `config.ACTIVE_AGING_CENTRES` (positions are in the locations
   page flight payload).
7. **Tile counts 67, drill-down lists 50** (demo gotcha): the tile's
   "N approvals" + category counts span ALL 67 in-window URA rows, but
   the raw-context "Planning decisions" list is capped at
   `URA_MAX_ITEMS = 50` with no "showing 50 of 67" note — 2 of the 5
   senior-care rows are invisible in the UI. If the user drills down
   and counts 3 senior care vs the tile's 5, that's why. Fix options:
   raise the cap or annotate the card.
8. **"67 approvals" headline + hidden caveat**: "permission ≠ opened
   facility" renders on the tile only when the category list is empty
   (basically never) — the caveat lives in the drill-down. Remaining
   options from the 2026-09-10 critique (per-category nearest shipped):
   restructure the sub-line to separate the distance signal from the
   category signals, and/or rename "approvals" → "permissions".

## Next-session prompt

CP_RAG is a FastAPI RAG assistant for SG primary care (uvicorn,
127.0.0.1:5001, log `/tmp/cp_rag_uvicorn.log`; startup ~2 min while the
context snapshot builds). Corpus: 96 ACG guideline PDFs (~2,659 chunks;
*Clinical guidelines* prompt section, never cited as protocol) +
primarycarepages.sg protocol pages + MOH public guidance + **NTUC Health
Active Ageing Centre programme calendars (26 centres, 386 chunks,
`ntuchealth.sg` source site, `Community resources` prompt section —
non-clinical, refreshed via `POST /api/community/refresh`, replace-only,
idempotent, guarded)** in Chroma (`chroma_db/` — expensive, don't rebuild).
Live local-context signals (NEA air, DENGBURDEN dengue, MOM heat, CDA
disease_week, **URA planning — Phase 8: per-row `category`,
`category_counts`, decision-date-window re-filter with `stale_dropped`,
street-name `district`/`approx_km`/`distance_band`/`near_clinic_count`
+ `nearest_km_by_category` (per-category nearest ~km on the tile),
cache `ura_planning_v2.json`; Phase 9: near band 10 → 2 km, tile renamed
`New health facilities (3 mo)`**) via the key-free `context/` package;
Phase 6 clinician-brief dashboard (Brief default: **6 KPI tiles — Nearest
polyclinic tile removed 2026-09-10 (data kept in prompt + raw context),
Active Ageing (2 km) tile added 2026-09-10** — / Clinical Flight Bag /
Library). **NTUC Active Ageing: `context/community.py` (Phase 9) —
calendar landing page, 24 h cache, 27-centre coordinate table in
config, `active_ageing` snapshot block with km/near/near_count +
raw-context card with per-centre calendar PDF links**; the same NTUC
programmes are ALSO in the RAG corpus as `Community resources` (Phase 7,
386 chunks). Embeddings: `BAAI/bge-m3` via `LOCAL_EMBEDDING_URL`
(free), never switched by the provider toggle. CHAT LLM is
runtime-switchable via header toggle: `openrouter` (DeepSeek, paid; boot
default) or `local` (vLLM `comp9:gpu0-vllm`, Qwen3.8-27B-FP8 at
`http://10.8.0.9:9001/v1` over WireGuard — `10.0.8.9` in old notes is a
typo; free). Local calls disable Qwen3 thinking via
`extra_body={"chat_template_kwargs":{"enable_thinking": false}}` in
`rag.py make_chat_llm()` (`LOCAL_DISABLE_THINKING=0` re-enables).
Retrieval: k=12 with `_steer_query()` protocol-name steering for the
retriever only; community-intent questions (`_COMMUNITY_INTENT_RE`) skip
steering AND get a second retrieval scoped to the NTUC calendar chunks
(up to 4 merged, total capped at 16) so specific programmes survive —
see "Done this session (continued — community programme retrieval fix)".
Current state: HEAD = origin/main = `b4b0a7a` (Phases 7–9 + provider
toggle committed and pushed 2026-09-10); uncommitted = per-category
nearest tile fix (`context/snapshot.py`, `static/index.html`,
`CLAUDE.md`) + community programme retrieval fix (`rag.py`); all
live-verified; tile's known soft spots in Outstanding #5–#8. Server
restarted 2026-09-10 for the user's demo (now PID 27565, uvicorn
127.0.0.1:5001, log /tmp/cp_rag_uvicorn.log; provider is in-memory, so
it is back on `openrouter` after the restart — re-set with
`POST /api/provider` if the demo needs local/free). URA v2 cache and
`ntuc_ageing_v1.json` hold real fetches (2026-09-10). Do next: commit
the tile fix + rag.py fix (user's call), then optionally pin
`CHAT_PROVIDER` in `.env`. Golden rules in `CLAUDE.md` (read it first).
