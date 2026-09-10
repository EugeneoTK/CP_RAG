# HANDOFF — CP_RAG (updated 2026-09-10 — SGDS UI reverted to pre-SGDS design per user preference, committed + pushed)

## Done this session (2026-09-10)
**All work committed** — `ce43834` (fix: brief generation) + `f626a07` (feat: SGDS v3 UI migration + static asset serving), pushed with this docs commit.

### 1. UI overhaul (`static/index.html`)
- Design-system variable set (`:root` tokens; accent `#2563eb`, surface `#fff`, radius 16px cards / 10px tabs); consistent type scale and button styles; header with app title + "Ready" pill.
- Brief tab: grouped control bar; 6 KPI tiles in a balanced 2×3 grid; watch cards with uppercase "Why it matters" / "Action" labels; bordered outlook card; provenance note.
- "Planning 90d" tile → plain English `CLINIC PLANS (URA)` ("74 approvals · healthcare-related, past 90 days · approved, may not be open yet"); detail card note clarifies approvals ≠ open facilities.
- Chat tab renamed → "Clinical Flight Bag" (label-only); grounded empty-state copy ("Ask about chronic-disease care — asthma, diabetes, hypertension, COPD, CKD, and more.").

### 2. Truncated-brief root cause + recovery
Root cause: brief generation hit the provider's default `max_tokens` cap → JSON cut mid-string → "Brief unavailable."
- `rag.py` `generate_brief()` — explicit `max_tokens=4000` (primary fix).
- `rag.py` `_parse_brief()` — tolerates prose around the JSON (extracts the outermost `{...}` span before `json.loads`).
- `rag.py` `_parse_and_guard()` — parse-error raw-text cap 2000 → 8000 chars.
- `static/index.html` — client-side `_salvageWatch()` recovers complete watch items from truncated JSON; amber "AI response was cut off" banner; scrollable mono fallback card + Copy button for unparseable output.

### 3. SGDS v3 component migration (`static/index.html` + `static/sgds-utility.css`)
- Remaining hand-rolled controls replaced with SGDS web components 3.26.1 (CDN, pinned): `sgds-button`, `sgds-input`, `sgds-textarea`, `sgds-tab-group`/`sgds-tab`/`sgds-tab-panel`, `sgds-chip`, `sgds-card`, `sgds-alert`, `sgds-badge`, `sgds-spinner`. Layout shell, hash routing and all JS logic preserved; page CSS now only adds sizing/behavior.
- **SGDS `:host` quirk (root cause of invisible-when-hidden alerts):** SGDS base component styles include `:host { display: block }`, which outranks the UA `[hidden] { display: none }` rule — the `hidden` attribute alone does NOT hide `sgds-alert`. Fix: page rule `sgds-alert[hidden] { display: none }` + JS toggles `hidden` and `show` together (`#ingest-alert`, `#brief-error`, `#brief-prose-head`).
- `#ctx-lat`/`#ctx-lon` inputs 150px; refresh control uses a "Refresh" text label (not `↻`).
- `static/sgds-utility.css`: local Tailwind build with SGDS theme alias tokens — the page depends on it (was untracked; now committed).

### 4. Live-backend static-asset fix (`app.py`)
- Backend served static files only under the `/static/` prefix while `index.html` (served at `/`) references assets root-relative (`/sgds-utility.css`) → 404 on the app; only the plain dev static server worked. Added a catch-all `StaticFiles(directory="static")` mount registered AFTER all routes — the page now renders identically from the backend.
- Inline SVG data-URI favicon — kills the remaining `/favicon.ico` 404 in both deployments (no binary asset).

### 5. Validation (Playwright harness `/tmp/sgds-verify/verify.mjs`, `VERIFY_BASE` overridable)
- Static mode (`node verify.mjs`, port 8137): **17/17** pass, zero real 404s.
- Live-ready mode (`VERIFY_BASE='http://127.0.0.1:5001/' node verify.mjs`): **16/17** — the single "fail" is the API-down assertion (expects Not ready/danger; with the backend up, the badge correctly shows Ready/success). `REAL_ERRORS []`, `CONSOLE_ERRORS []`.
- Harness is mutation-tested (a hidden-rule regression is caught).

### 6. Server restarted
- New server on port 5001 (pid 22814, log `/tmp/cp_rag_server.log`) running the NEW code — brief fixes and catch-all static mount active. Startup ~2 min this time (vs the 6.14 min cold-start figure in Outstanding — worth re-checking at next restart). Brief cache cold after restart by design.

### 7. SGDS UI reverted to pre-SGDS design (user request)
- User reviewed the SGDS UI live and preferred the original look. Reverted: `git checkout ce43834 -- static/index.html` (the pre-SGDS single self-contained 1203-line file — blue `#1a56db` header, Brief/Chat/Library tabs, hand-rolled CSS/JS, no external assets) + `git rm static/sgds-utility.css` (existed only for the SGDS build).
- Only SGDS-era carry-over kept: the inline SVG data-URI favicon (invisible; kills the `/favicon.ico` 404) — recolored its tile from `#2563eb` to the design blue `#1a56db` to match the header.
- `app.py` untouched: the `@app.get("/")` `FileResponse` route takes precedence, so the catch-all `StaticFiles` mount from `f626a07` stays — harmless and future-proofs any root-relative asset.
- Note: the "Clinical Flight Bag" tab rename was part of `f626a07`, so the restored UI shows the original **"Chat"** label — expected, not a regression.
- Validated with a fresh harness `/tmp/sgds-verify/verify_revert.mjs` (puppeteer-core; `VERIFY_BASE` overridable, `VERIFY_LIVE=1` for backend mode): **23/23** static-mode (incl. expected API-down degradation: Offline badge + brief-error banner, only-/api/ 404s) and **25/25** live on 5001 (Ready badge, 6 KPI tiles, real context chips, tab switching, zero 404s / console / page errors). Screenshots: `/tmp/sgds-verify/revert-*.png`.
- The SGDS commit `f626a07` remains in git history if the design is ever wanted back.

## Outstanding
1. **Regenerate the brief — 1 paid call, user's call** — to confirm the truncation fix holds (max_tokens=4000) and the (restored) brief tiles/cards render real content.
2. (carry-over) Upload progress indicator for `POST /api/ingest-pdf` (stretch).
3. (carry-over) URA planning detail card: `decision_type` values not yet mapped to plain language.
4. (carry-over) WIDB: per-region dengue case counts (weekly) missing from the fetch.
5. (carry-over) CDA `disease_week` is a single national bulletin, not clinic-specific.
6. (carry-over) No alerting/escalation tiers — signals surface only in brief/chat context.
7. (carry-over) data.gov.sg is the sole external data source — no retry/circuit breaker; one 5xx degrades the snapshot until the 24 h TTL.
8. (carry-over) Cold start was 6.14 min (reliability debt: ~4.6 min embedding 2,716 ACE chunks); this restart took ~2 min — re-check at next restart before concluding it's fixed.
9. (carry-over) `context/` has no tests (stretch); 2,716-chunk ACE corpus completeness unverified (optional).

Lower-priority carry-overs (unchanged): live PDF upload unexercised recently (`POST /api/ingest-pdf` verified in Phase 5); primarycarepages refresh needs a full `ingest()` rebuild (URL rewrite defeats append skip); postcode geocoding is district-centroid approx (OneMap DNS-blocked) — `--lat/--lon` exact; `rag.py` ingest dedupe is check-then-act with no lock (concurrent duplicates can double-embed — wasted credits, not corruption); 1 scanned ACE PDF uningested (rehab appendix, needs OCR); ACE repo drift check: `venv/bin/python scripts/ace_guidelines.py --list` (free), `--ingest` idempotent; k=12 retrieval heuristic — revisit as corpus grows; WIDB parse layout-fragile (3-day cache may serve stale parse — `python -m context --no-cache` force-refetches); catchment geocoding for per-clinic URA planning signals.



## Next-session prompt
Project: CP_RAG at `/Users/ugeneo/Documents/Project Codes/CP_RAG` — a Singapore GP clinic's AI research assistant: FastAPI + LangChain RAG over primary-care chronic-care protocol pages, 96 ACG clinical-guideline PDFs (2,975 chunks, routed to the *Clinical guidelines* prompt section, never cited as protocol content), MOH public guidance, and live local-context signals (NEA air, DENGBURDEN dengue clusters, MOM heat, CDA `disease_week`, URA planning `catchment_change` — all key-free via the `context/` package), plus the Phase 6 clinician-brief dashboard with hash-routed Brief (default) / Clinical Flight Bag (chat) / Library tabs. LLM `deepseek/deepseek-v4-flash-0731` + `text-embedding-ada-002` via OpenRouter (`.env`, git-ignored — never commit). Golden rules in `CLAUDE.md`: credits are a budget (prefer `/api/status` over chat probes; one brief Generate = 1 paid call), Python 3.9, `chroma_db/` is expensive — don't rebuild. Read `CLAUDE.md` first; phase notes in `docs/signal-research.md` §9/§12–16.

State (2026-09-10): All 2026-09-10 work committed + pushed: `ce43834` (fix: brief generation — max_tokens=4000, prose-tolerant JSON parse, raw cap 2000→8000) + `f626a07` (feat: SGDS v3 UI migration + static-asset serving fixes) + this docs commit; HEAD = origin/main. Server on port 5001 (pid 22814, log `/tmp/cp_rag_server.log`) is running the NEW code — startup ~2 min this time. SGDS-migrated UI was subsequently REVERTED at the user's request (they preferred the original look): `static/index.html` restored from `ce43834`, `static/sgds-utility.css` deleted; only the inline data-URI favicon was ported over (recolored to `#1a56db`). The catch-all `StaticFiles` mount in `app.py` stays. UI re-validated with the fresh harness (`/tmp/sgds-verify/verify_revert.mjs`, `VERIFY_BASE` overridable, `VERIFY_LIVE=1` for backend): 23/23 static-mode, 25/25 live-ready-mode, zero 404s, zero console errors; live run renders the 6 KPI tiles + context chips from real data. The SGDS commit `f626a07` remains in history if ever wanted back. Test clinic: Lakeside Family Medicine Clinic, 518A Jurong West Street 52 (1.3454017, 103.7188383, PC 641518). Brief cache cold after restart by design; Generate = 1 paid call, user's call.

Do next: 1) if the user approves the paid call, regenerate the brief to confirm the truncation fix holds and the SGDS tiles/cards render real content; 2) remaining backlog is the numbered Outstanding list above (brief regen, upload progress indicator, URA `decision_type` mapping, WIDB dengue case counts, cold-start re-check, …). Standing checks after any change: py_compile, `/api/health`, `/api/status`, `/api/context` (0 LLM credits), `/api/brief` (paid — do NOT generate unless asked).
