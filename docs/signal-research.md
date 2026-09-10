# Signal Research — Dynamic "Flight Information" Layer for GP Context POC

Research date: 2026-09-08. Entity-neutral OGP POC scope: population-level, clinic-catchment
context signals built from open government data (data.gov.sg + MOH/CDA), no patient-level data.

Full dataset inventory swept: **4,616 datasets** (data.gov.sg v2 API, 0 failed pages),
filtered into signal families, then top candidates verified end-to-end (metadata + sample rows + live fetch).

---

## 1. Verified LIVE signals — usable with NO API key

| # | Signal | Source | Format / cadence | Verified state (2026-09-08) | Access |
|---|--------|--------|------------------|------------------------------|--------|
| S1 | **Dengue clusters** | NEA via data.gov.sg | GEOJSON, weekly | 13 active cluster polygons; block-level `LOCALITY` (e.g. "Admirality Lk (Blk 493)"), `CASE_SIZE` 2–258, updated **2026-09-04** | `poll-download` → signed S3 URL |
| S2 | **High Aedes population areas** | NEA via data.gov.sg | GEOJSON, weekly | 135 street-level polygons (CO area + street names), updated **2026-08-27** | `poll-download` → signed S3 URL |
| S3 | **PM2.5 (live)** | NEA | Real-time API, hourly | Live regional + national readings (µg/m³), 5 regions | `GET https://api-open.data.gov.sg/v2/real-time/api/pm25` — **no key needed** (works anonymous) |
| S4 | **PSI (live)** | NEA | Real-time API, hourly | Live regional + national readings | `GET https://api-open.data.gov.sg/v2/real-time/api/psi` — **no key needed** |
| S5 | 1-hr PM2.5 history | NEA via data.gov.sg | CSV, hourly since 2014 | Updated 2026-08-12 | `list-rows` / `datastore_search` |
| S6 | 24-hr PSI history | NEA via data.gov.sg | CSV, hourly since 2014 | Updated 2026-08-12 | `list-rows` / `datastore_search` |
| S7 | WIDB weekly PDFs (all notifiable diseases) | CDA (moh.gov.sg) | PDF, weekly (Sundays) | 2026 archive live: EW 1–33+; 2025 archive EW 35–53 | MOH page + CDA page; PDFs on `isomer-user-content.by.gov.sg` |
| S8 | Hospital admissions / public-sector outpatient, monthly | SINGSTAT via data.gov.sg | CSV, monthly | Updated 2026-08-09 (wide-format monthly columns) | `list-rows` / `datastore_search` |
| S9 | Vaccination service locations | MOH via data.gov.sg | GEOJSON | Polyclines (2025-11-13), JTVC (2026-04-18) | `poll-download` |
| S10 | **2-hr weather forecast, town-level** | NEA MSS | Real-time API, ~every 2 hr | **51 towns/subzones** each with forecast + `area_metadata` lat/long; updated 2026-09-08 10:06 SGT | `GET .../real-time/api/two-hr-forecast` — no key needed |
| S11 | 24-hr weather forecast | NEA MSS | Real-time API, several×/day | Today's high/low (35/25°C), wind, valid period; updated 09:31 SGT | `GET .../real-time/api/twenty-four-hr-forecast` |
| S12 | 4-day outlook | NEA MSS | Real-time API, daily | Per-day high/low, rain probability, wind, humidity | `GET .../real-time/api/four-day-outlook` |
| S13 | **WBGT observations** | NEA MSS | Real-time API, 15-min cadence | 15 stations with lat/long, `wbgt` °C + `heatStress` (Low/Moderate/High); e.g. 28.4°C Changi, Low, 10:15 SGT | `GET .../real-time/api/weather?api=wbgt` |
| S14 | Flood alerts | PUB | Real-time API, event-driven | Feed live; 25 records, latest 09:38 SGT (empty readings = no active alert) | `GET .../real-time/api/weather/flood-alerts` |


### WIDB locations (the last missing piece — FOUND)
- MOH landing: `https://www.moh.gov.sg/others/resources-and-statistics/infectious-disease-statistics-2025-weekly-infectious-diseases-bulletin/`
- Official CDA 2026 archive: `https://www.cda.gov.sg/resources/weekly-infectious-diseases-bulletin-2026/`
- MOH page is plain WordPress/Isomer HTML — **curl works with a browser UA** (earlier block was transient/UA-related). No scraping block in practice.
- Format is **PDF only** (e.g. `Weekly Infectious Disease Bulletin EW 49.pdf`). No CSV/API feed. The old machine-readable WIDB CSV on data.gov.sg is dead (see §3).
- MOH sitemap: `https://www.moh.gov.sg/sitemap.xml` (~2 MB, useful for finding other pages).

## 2. Real-time API keys — CORRECTED 2026-09-08 (earlier "key-gated" reading was wrong)

With a valid `x-api-key` (project key verified: 200 on `api-production` dataset APIs), and after
finding the true endpoint slugs (see below), **all real-time weather/heat/flood APIs work — the
key is OPTIONAL; without it they work anonymously exactly like `pm25`/`psi`** (per their own
OpenAPI specs: `"x-api-key … optional, for higher rate limits"`).

**Why we were stuck:** the `api-open` gateway returns the same 403
`{"message":"Missing Authentication Token"}` (or a SigV4-looking `Credential/Signature` error)
for *any unknown path* — the slugs are not `weather24hr`-style, they are spelled out, and they are
only discoverable from each dataset page's embedded "Download API Specs" (OpenAPI JSON) on
`data.gov.sg/datasets/{id}/view`:

| Real-time endpoint (base `https://api-open.data.gov.sg/v2/real-time/api`) | Dataset id | Live-verified 2026-09-08 |
|---|---|---|
| `/pm25`, `/psi` | — | 200 (earlier) |
| `/twenty-four-hr-forecast` | `d_ce2eb1e307bda31993c533285834ef2b` | 200, today's 35/25°C |
| `/two-hr-forecast` | `d_3f9e064e25005b0e42969944ccaf2e7a` | 200, 51 towns, 10:06 SGT |
| `/four-day-outlook` | `d_f131f6e343bf8168e4057a04c4326a0a` | 200, day-by-day |
| `/weather?api=wbgt` (`api` param **required**) | `d_87884af1f85d702d4f74c6af13b4853d` | 200, 15 stations + `heatStress` |
| `/weather/flood-alerts` | `d_f1404e08587ce555b9ea3f565e2eb9a3` | 200, event feed live |

Notes:
- There is **no** `heatadvisory` endpoint/dataset — heat signal = WBGT `heatStress` field.
- `traffic` slug still unknown and low clinical relevance — dropped.
- The key's real value (since 2025-12-31 platform-wide rate limits): **higher rate limits +
  priority support + maintenance notifications**. Use it; it goes in `.env` as `DGS_API_KEY`.
- All real-time endpoints accept `?date=YYYY-MM-DD` (full-day backfill) + `paginationToken`.

Historical forecast CSVs (2-hr / 24-hr / 4-day, 2016–2024) remain available via list-rows if backtests are needed.
## 3. Stale / superseded — do NOT use for live context

| Dataset | data.gov.sg status |
|---------|-------------------|
| Weekly Infectious Disease Bulletin **Cases** (CSV) | `coverageEnd: 2022-12-31` — retired; structure is (epi_week, disease, no._of_cases) — 20,070 rows |
| Weekly Dengue & DHF cases (CSV) | last row **2018-W53** |
| Avg Daily Polyclinic Attendances for Selected Diseases (CSV) | last row **2022-W52** |
| Avg daily hospitalised/ICU by Epi-week (CSV) | 2023 only |
| Top 10 Conditions of Hospitalisation (CSV) | annual, to 2024 |
| Regional Dengue (Cases) GEOJSON ×5, Aedes Breeding Habitats ×5 (2024) | superseded by live S1/S2 |

These are fine as **historical baselines** only.

## 4. data.gov.sg v2 API surface (working reference)

```
BASE_LIST  = https://api-production.data.gov.sg/v2/public/api
BASE_OPEN  = https://api-open.data.gov.sg

# Inventory (no search param exists — sweep all 462 pages, 10/page)
GET {BASE_LIST}/datasets?page=N

# Per dataset
GET {BASE_LIST}/datasets/{datasetId}/metadata     # name, format, lastUpdatedAt, coverage, columnMetadata
GET {BASE_LIST}/datasets/{datasetId}/list-rows?limit=N          # CSV/API datasets → JSON rows
GET {BASE_OPEN}/v1/public/api/datasets/{datasetId}/poll-download # non-CSV (GEOJSON/KML/PDF) → signed S3 URL
# (initiate-download only needed for filtered CSV extracts)

# Legacy CKAN-style search (still alive, supports sort/filter; rate-limits at ~429 — pace it)
GET https://data.gov.sg/api/action/datastore_search?resource_id={datasetId}&limit=N&sort="col desc"

# Real-time (base {BASE_OPEN}/v2/real-time/api) — key OPTIONAL; slugs spelled out, see §2
GET .../pm25[?date=YYYY-MM-DD]              # anonymous OK
GET .../psi[?date=YYYY-MM-DD]               # anonymous OK
GET .../twenty-four-hr-forecast             # 200, national 24-hr forecast
GET .../two-hr-forecast                     # 200, per-town (51 areas) + area_metadata
GET .../four-day-outlook                    # 200, day-by-day outlook
GET .../weather?api=wbgt                    # 200, 15 stations, heatStress Low/Mod/High
GET .../weather/flood-alerts                # 200, PUB flood alert event feed
```

- Old CKAN `data.gov.sg/api/action/package_search` is **gone** (404); `datastore_search` survives.
- Guide: `guide.data.gov.sg` — every page also available as Markdown (append `.md`), index at `/llms.txt`.
- License: **Singapore Open Data License** — commercial + personal use permitted.
- Rate limits: documented per-key on api-open; list-rows/datastore_search unauthenticated is lenient but does 429.


## 5. Gaps & risks

1. **WIDB is PDF-only.** The richest live multi-disease weekly signal (dengue, HFMD, chikungunya,
   influenza, etc., national counts + rates) needs a PDF parser (pdfplumber/pypdf) to extract the
   tables. Weekly cadence, published ~Monday. Feasible; ~1 day of work for a stable schema.
2. **Air quality is 5-region granularity** (N/S/E/W/Central) — coarse for a clinic-catchment claim;
   fine for "this week's environmental context" framing.
3. **Dengue clusters are national polygons**, no per-PLA counts. Catchment relevance requires a
   spatial join: clinic point (or catchment postcode) vs cluster polygon (point-in-polygon, shapely).
   13 polygons makes this cheap.
4. **No facility-level load signals** (A&E waits, ED attendances) in the open inventory — out of scope.
5. **~~Weather/heat/flood need a free API key~~ — RESOLVED 2026-09-08:** all work anonymously;
   the key (DGS_API_KEY) only buys higher rate limits. WBGT `heatStress` is the heat signal.
6. MOH site intermittently challenges non-browser UAs; use a real browser UA header, and prefer
   data.gov.sg structured files over MOH HTML wherever possible.
7. **Gateway footgun (api-open):** unknown real-time paths return 403 "Missing Authentication
   Token" — indistinguishable from a missing key. Always confirm slugs from the dataset page's
   embedded OpenAPI spec before debugging auth.

## 6. Candidate shortlist for the POC (key-free core)

| Family | Signals | Why |
|--------|---------|-----|
| Vector-borne / infectious | S1 dengue clusters, S2 aedes areas, S7 WIDB PDF | Singapore's defining GP seasonality (dengue season, HFMD, flu) |
| Environmental | S3 PM2.5, S4 PSI, S13 WBGT/heat stress, S10–S12 weather, S14 flood | Respiratory + heat-related presentations; rain drives dengue vector breeding |
| System context | S8 monthly admissions (national trend) | Low-frequency backdrop, keeps it population-level |
| Service availability | S9 vaccination locations | "Which nearby polyclinics run the current campaign" |

## 7. Minimal context snapshot schema (proposed)

```json
{
  "snapshot_date": "2026-09-08",
  "epi_week": "2026-W36",
  "clinic": {
    "name": null,
    "postcode": "369905",
    "region": "east",
    "lat": 1.345, "lng": 103.918
  },
  "dengue": {
    "active_clusters_national": 13,
    "clusters_in_region": 3,
    "nearest_cluster": {"locality": "...", "cases": 42, "distance_km": 1.2},
    "in_high_aedes_area": true
  },
  "air": {
    "pm25_now": {"national": 12, "region": 15},
    "psi_now": {"national": 40, "region": 48},
    "pm25_7d_avg": 14.2
  },
  "weather": {
    "today": {"high_c": 35, "low_c": 25, "text": "Partly Cloudy (Day)"},
    "tomorrow": {"high_c": 34, "low_c": 26, "text": "Thundery Showers"},
    "town_forecast_2hr": "Partly Cloudy (Day)",
    "wbgt": {"station": "Upper Changi Road North", "value_c": 28.4, "heat_stress": "Low"},
    "flood_alert_active": false
  },
  "disease_week": {
    "source": "WIDB EW 36 (published 2026-09-07)",
    "dengue_cases_national": 187,
    "notable": ["HFMD cluster in ...", "Influenza A rising"]
  },
  "system": {"hospital_admissions_3m_trend": "up 2%"}
}
```

## 8. Proposed POC feature set (for discussion)

- **F1 — Context builder** (no LLM cost): Python module that fetches S1–S4 (+S7, S8) and emits the
  snapshot schema for a given clinic point. Reusable CLI + FastAPI endpoint.
- **F2 — "Flight info" panel**: strip in the existing CP_RAG chat UI showing the week's context
  (dengue nearby? air quality? heat?) with data-as-of stamps and source links.
- **F3 — RAG integration (the differentiator)**: prepend the snapshot (JSON → compact prose) to the
  RAG chain's system context so protocol answers are framed by population context. No new corpus,
  no extra embeddings.
- **F4 (stretch)** — mini map: clinic marker + active dengue cluster polygons + aedes areas.
- **F5 — Weather/heat/flood (verified live, key optional)**: town-level 2-hr forecast, WBGT
  heat-stress, 4-day rain outlook, active flood alerts — now core, not stretch (see §2).
- **F6 (stretch, key verified)** — URA planning decisions: "new polyclinic / clinic / senior-care
  approved near your catchment" + new residential units approved in region (see §10).


## 9. Implementation plan sketch

| Phase | Deliverable | Cost |
|-------|-------------|------|
| 0 — Spike | `context/` package: fetchers for S1–S4 + S10–S14 (all key-free), snapshot JSON for one test postcode, `python -m context` CLI | 0 API credits |
| 1 — Panel | F2 UI strip on `static/index.html` fed by `GET /api/context` — **DONE 2026-09-08** (lat/lon or postcode params, in-memory 15-min cache keyed on 3-dp coords, `X-Snapshot-Cache` header, strip with chips + detail cards + location switcher + 15-min auto-refresh) | 0 credits |
| 2 — RAG | F3 context injection into `rag.py` chain prompt | 1–2 test chats |
| 3 — WIDB | PDF table parser → `disease_week` block in snapshot | 0 credits |

(Phases 0, 2, 5 as above; **Phase 3 — WIDB: DONE 2026-09-08** — `context/widb.py`: archive crawl → walk-back PDF fetch → pypdf parser → `disease_week` block; WIDB line in *Local context*; CLI `DISEASE` section; UI chip + card. Build notes in §14.)
| 4 — URA | F6 planning-decision catchment block (`URA_ACCESS_KEY` in .env) — **DONE + live-verified 2026-09-08** (2,363 rows / 74 healthcare-related in the 90-day window, §15) (`context/ura.py`: daily token → `Planning_Decision&last_dnload_date=<today−90d>`; healthcare keyword filter on `submission_desc`; 24 h disk cache; `catchment_change` snapshot block = island-wide healthcare-related written permissions, latest 20, each with `decision_type` — NOT opened facilities; *Local context* prompt line; CLI `PLANNING` section; UI chip + card; no key → `ura:` data gap). Sketch deviation: `new_resi_units_approved_region` dropped (no units field in the service; regional aggregation needs OneMap geocoding — deferred, §15) | 0 credits |
| 5 — ACE guidelines | Clinical-guidelines PDFs in the RAG corpus: `pypdf` extraction, append-only `ingest_pdf()` with hash dedupe, `POST /api/ingest-pdf` + `GET /api/pdfs`, 4th prompt section "Clinical guidelines", UI upload bar, optional `scripts/ace_guidelines.py` sitemap crawler for all 29 ACE ACGs. Plan: `docs/superpowers/plans/2026-09-08-ace-guidelines-pdf.md` — **DONE 2026-09-08** (manual-upload path + full seed: 96 guideline PDFs / 2,975 guideline chunks in the store; 1 scanned appendix uningested) | ~1–2 credits (upload path) / a few cents (full ACE seed) |
| 6 — Clinician brief + Library | On-demand one-page clinician brief (headline / watch list / outlook) from the live snapshot + 6 KPI tiles + structured-only Protocol spotlight; three hash-routed tabs (Brief default / Chat / Library); `GET /api/library` corpus inventory + Library UI (PDF grid, protocol collapse card, MOH section). Plan: `docs/superpowers/plans/2026-09-09-clinician-brief-library.md`; spec: `docs/superpowers/specs/2026-09-09-clinician-brief-library-design.md` — **DONE 2026-09-09** (`context/brief.py` prompt projection with `protocol_links` structurally excluded; `rag.py` `generate_brief()` + parse + provenance drop guard; `app.py` two-phase brief API with the full guard order + `GET /api/library`; UI tabs / tiles / renderer / Library view; build notes in §16) | exactly 1 paid call planned — **2 consumed** (snapshot-rotation probe deviation, §16) |

## 10. URA e-Services API — `eservice.ura.gov.sg` (access key VERIFIED 2026-09-08)

Docs: `https://eservice.ura.gov.sg/maps/api/` (Swagger-style reference).

Auth: `AccessKey` header → `GET /uraDataService/insertNewToken/v1` returns a **daily** token →
all data calls: `GET /uraDataService/invokeUraDS/v1?service=<name>` with `AccessKey` + `Token` headers.

| Service | Data | Cadence | POC fit |
|---------|------|---------|---------|
| **Planning_Decision** | URA written permission decisions (granted/rejected): address, submission_desc, decision_date, appl_type, lot no | **Daily**; query by `year` or `last_dnload_date=dd/mm/yyyy` (max 1 year back) | **YES — the one worth using** |
| Car_Park_Availability | URA car park free lots | every 3–5 min | no (not clinical) |
| Car_Park_Details / Season_Car_Park_Details | car park list + rates | slow | no |
| PMI_Resi_Transaction / Rental / Rental_Median / Developer_Sales / Pipeline | private residential property market | weekly/monthly | weak (socioeconomic proxy only) |
| EAU_Appr_Resi_Use | approved residential uses | slow | weak |

**Live-verified with the project access key** (key stored outside repo; when integrated it goes in
`.env` as `URA_ACCESS_KEY` and `.env.example` — never in code):
- token fetch → 200 OK
- `Planning_Decision&last_dnload_date=01/08/2026` → 200 OK, **973 rows**, current through
  early Sep 2026 (e.g. new 3-storey child care centre, Holland Road; HDB strata subdivisions;
  commercial use changes).

### How it fits the POC
`Planning_Decision` is the only clinically-adjacent service. Use: poll weekly, filter
`submission_desc` for health/elderly/childcare keywords (POLYCLINIC, CLINIC, MEDICAL, NURSING
HOME, CHILD CARE, SENIOR) and count new residential units per region → snapshot block:
```json
"catchment_change": {
  "healthcare_approvals_90d": [{"address": "49B HOLLAND ROAD", "what": "child care centre", "date": "2026-02-03"}],
  "new_resi_units_approved_region": 1800
}
```
Caveats: (1) a planning decision is a *permission*, not an opened facility; (2) rows carry street
addresses, no postal code/region — mapping to a clinic catchment needs geocoding (OneMap
Geocoding API is free) or address→region text matching; (3) ~1k rows/week — trivial to store.

**v1 scope ruling (2026-09-08, Phase 4 build).** The sketch's
`new_resi_units_approved_region` is not derivable from this service: rows
carry no unit counts, and `address` is a street address with no
postcode/region, so regional aggregation requires geocoding (OneMap's free
API needs its own key) or a street→region table — both deferred. v1 ships
`healthcare_decisions_90d` (island-wide; every `decision_type`, each
labelled — the sketch's `healthcare_approvals_90d` refined, since the
service returns multiple decision types, observable only live: in the
2026-09-08 window `Written Permission`/`Authorized Work`/`Corrigendum`,
no refusals — §15). Catchment
mapping (caveat 2) is the named follow-up if per-clinic distance filtering
is wanted.

Sequencing keeps every phase shippable and credit-free until Phase 2.

## 11. Phase 0 build notes (completed 2026-09-08)

Delivered: `context/` package (stdlib only, Python 3.9, **no new dependencies**) +
`.env.example` (`DGS_API_KEY` optional, `URA_ACCESS_KEY` placeholder).

```
context/
  config.py     endpoints, dataset ids, thresholds, test clinic, region partition, cache dir
  http.py       UA + 429 backoff (anonymous rate limit is tight: ~8 fast calls → 429)
  fetchers.py   pm25/psi, 24-hr + 2-hr + 4-day forecast, WBGT, flood alerts, GEOJSON poll-download
  geo.py        haversine, ray-cast point-in-polygon, nearest-of-N (no shapely needed)
  linkage.py    signal → protocol map (the differentiator; see below)
  snapshot.py   assembles the §7 schema + protocol_links; failures land in data_gaps
  __main__.py   CLI: venv/bin/python -m context [--lat --lon | --postcode] [--json] [--no-cache]
```

Verified live (0 API credits): test clinic = **Woodlands Polycline** (103.7752, 1.4309,
PC 738579 — real point from the Vaccination_Polyclinics GEOJSON `d_b22489c7dc4065b6e7e45f177fdb33be`).
First run emitted a complete snapshot, `data_gaps: []`, 4 active links: PM2.5 29 (S),
WBGT 31.9 °C Moderate, thundery showers Wed–Fri, and a **dengue cluster 2.7 km from the clinic**.

**2026-09-09 (test clinic swap, historical note):** the test clinic is now
**Lakeside Family Medicine Clinic**, 518A Jurong West Street 52, PC 641518
(103.7188, 1.3454 — real GP-clinic point, OSM node 8083778017; same-day swaps to
Pioneer Polyclinic and DA Clinic @ Taman Jurong were superseded — first because a
*polyclinic* is not the app's GP persona, then by user choice between the two GP
finalists). It generates the richest live brief: west-region PM2.5 56 (island peak
75, central — haze day) **and** a **60-case dengue cluster** (Ho Ching Rd,
upd 2026-09-03) **1.1 km** away + a 2-case cluster 0.3 km away → 5 active protocol
links, `data_gaps: []` (verified 0 LLM credits). The Woodlands run above is the
2026-09-08 historical record.

Implementation decisions / corrections:
- **GEOJSON path** is `GET api-open.../v1/public/api/datasets/{id}/poll-download` (v1, key-free)
  → signed S3 URL. The v2 production subscription endpoint is 404. Cache: system temp dir,
  12 h TTL — never in the repo.
- **Dengue cluster props**: `LOCALITY`, `CASE_SIZE`, `FMEL_UPD_D` (yyyymmddHHMMSS).
  Aedes props: `DESCRIPTION` ("CO77 - street list"). Polygons are `[lon, lat]`.
- **PM2.5/PSI** `data.items[-1].readings` carries the 5 regions only — no `national` key in
  the live payload (the proposed §7 schema assumed one); snapshot reports `peak` + `peak_region`.
- **2-hr forecast** `area_metadata` currently lists **47 towns** (not 51). Town matched by
  nearest `label_location` to the clinic.
- **WBGT** `readings[].wbgt` is a **string**; `location` lat/lon are strings — cast before use.
- **Flood alerts**: `records[].item.readings` empty = no alert (type "observation"); non-empty
  readings = an actual alert entry.
- **No geocoder on this network**: `api.onemap.sg` / `api.onemap.gov.sg` fail DNS. Clinic point
  comes from `--lat/--lon` (exact) or `--postcode` (first-digit district centroid, coarse).
- NEA 5-region label is a **coarse lat/lon partition** (documented in config.py), and the
  full 5-region table is always in the snapshot so nothing hides behind the approximation.
- App launch for the standard check is `venv/bin/uvicorn app:app --port 5001` —
  `python app.py` only builds the FastAPI app (no run block), which looks like a hang.

### How signals link to care protocols (the Phase 0 answer)

Corpus verified: primarycarepages.sg `/Healthier-SG/Care-Protocols/Chronic` — **17 protocols**
(Allergic Rhinitis, Asthma, BPH, Chronic Hep B, CKD, COPD, Diabetes, GAD, Gout, Hypertension,
IHD, Lipid Disorders, MDD, Multimorbidity DM+HTN+Hyperlipidaemia, Osteoarthritis,
Pre-Diabetes, Stable Stroke). `linkage.py` curates a signal→protocol map over exactly these.

Three mechanisms — context never rewrites protocol content, it *frames* it:

| Mechanism | What changes | Phase |
|-----------|--------------|-------|
| **A. Interpretation framing** | The RAG system prompt gains a `context_brief` (prose). A question like "68 y/o COPD, worsening breathlessness" is then answered with the live trigger context in mind (e.g. regional PM2.5 29 µg/m³ → environmental trigger, review inhaler adherence + action plan). | Phase 2 ✅ (2026-09-08, §13) |
| **B. Retrieval steering** | Active protocol names (e.g. "COPD", "heat") are appended to the retrieval query so the vector search surfaces matching chunks — no re-embedding. | Phase 2 ✅ (2026-09-08, §13) |
| **C. Population counselling** | Visit-level nudges independent of the patient's question (e.g. do-the-mozzie-wipeout while a cluster is active nearby; hydration advice in heat). | Phase 1 UI + Phase 2 ✅ (2026-09-08, §13) |

Example linkage (today's live run, Woodlands test clinic):

| Active signal | Protocols engaged | Clinical hook |
|---------------|-------------------|---------------|
| PM2.5 peak 29 µg/m³ (S) | Asthma, COPD, IHD, HTN | environmental trigger → inhaler technique/adherence, action plan, outdoor-exertion advice |
| WBGT 31.9 °C, Moderate | DM, CKD, HTN, Gout, Multimorbidity | dehydration: hypoglycaemia, prerenal AKI (review ACEi/ARB/diuretics), electrolytes, gout flares |
| Thundery showers Wed–Fri | HTN, DM, CKD | access disruption → chronic-tier medication collection before discharge |
| Dengue cluster 2.7 km (2 cases) | DM, CKD, HTN, IHD, Stroke | febrile patient → NS1/CRP; sick-day rules: stop SGLT2 while acutely unwell (CKD + multimorbidity protocols), NSAID caution on antiplatelets/anticoagulants → paracetamol |

Thresholds (config.py, conservative): PM2.5 ≥ 15 moderate / ≥ 25 high; PSI ≥ 101 unhealthy /
≥ 201 very unhealthy; heat = WBGT ≥ 31 °C or heatStress ≥ Moderate; rain = "thunder/shower"
in 4-day outlook or active flood alert; dengue-near = cluster centroid ≤ 3 km; aedes =
point-in-polygon. Each is one constant — easy to tune per clinical feedback.


## 12. F2 provenance audit (2026-09-08)

Before Phase 2 injects context into the RAG prompt, every linkage rule was
audited against the actual corpus text — direct grep over all 3,785 indexed
chunks (`chroma_db/chroma.sqlite3`, table `embedding_fulltext_search_content`,
per-chunk `source` metadata from `embedding_metadata`). Zero API credits.

Result: each rule in `context/linkage.py` carries a `basis` tag +
`basis_note`, surfaced in `protocol_links.active[]`, in a `basis_legend`
inside `protocol_links`, and as a provenance badge on the F2 strip.

| Signal | Linked protocols | Basis | Corpus evidence (grep of 3,785 chunks) |
|--------|------------------|-------|------------------------------------------|
| humidity | Allergic Rhinitis, Asthma | partial | Rhinitis protocol names "house dust mite, pets, rodents, cockroaches, indoor moulds, cigarette smoke" as common allergens (8 chunks). Asthma protocol has "trigger avoidance"/"look for triggers" but never names mites/moulds. |
| heat (WBGT) | DM, CKD, HTN, Gout, Multimorbidity | partial | In corpus: dehydration as a gout flare trigger; CKD hydration advice (euglycaemic DKA risk); "Sick-day advice: Stop SGLT2 inhibitors … while acutely unwell" (CKD + multimorbidity). "Heat"/"WBGT" itself: 0 mentions. |
| dengue_cluster | DM, CKD, HTN, IHD, Stroke | partial | In corpus: SGLT2 sick-day rules (CKD + multimorbidity). Dengue: 0 mentions in any of the 17 protocols (all "dengue/fever" hits are non-protocol pages: GPFirst, NUHS marketing, yellow-fever vaccination). NS1 differential + NSAID avoidance on anticoagulants = standard practice. |
| pm25 | Asthma, COPD, IHD, HTN | partial | **Corrected 2026-09-08:** 10 chunks mention "haze" — Asthma + COPD protocols each say "Advise on haze precautions when appropriate" (single line, no thresholds); PHPC/Haze Subsidy Scheme pages list six haze-related conditions. The original audit word list (pollut/PM2.5/air quality/smog) missed "haze". PM2.5/PSI levels and the exacerbation association: 0 hits in protocol text. |
| psi | Asthma, COPD, IHD | derived | 0 hits for PSI; the "haze precautions" line carries no PSI threshold. PSI-band → activity advice now citable from the MOH haze page (public guidance, in corpus since Phase 2). |
| rain | HTN, DM, CKD | derived | Only "flood" hits are the site's legal DDoS disclaimer. Logistics/access argument. |
| aedes_area | — | none | Population counselling only; no clinical claim. |

**Rule for Phase 2 (F3).** The RAG prompt gets two labelled sections —
*Protocol content* (only `corpus`/`partial` material, per `basis_note`) and
*Local context* (all live signals, never attributable to the protocols).
`derived` links must never be presented as protocol content. The F2 strip
renders a provenance badge per link with the audit note on hover.

Wording fix (2026-09-08): the `dengue_cluster` clinical focus previously read
"hold metformin/SGLT2 inhibitors during dehydration or vomiting"; the corpus
only supports the SGLT2 sick-day rule, so it now reads "stop SGLT2 inhibitors
while acutely unwell (CKD + multimorbidity protocols)".

## 13. Phase 2 build notes (completed 2026-09-08)

Phase 2 delivered F3 (context-injected RAG) + corpus expansion, all verified
live on the free model with 1 append-embed (~4 chunks) + 2 chat calls.

**1. Provenance renderer — `context/prompts.py`** (new). `format_live_context`
splits the snapshot into: *basis notes* (only `corpus`/`partial` links, each
with its `basis_note`) and *local context* (ALL active signals, labelled "not
protocol content", plus `population_counselling` lines). `derived`/`none`
never reach the protocol section — enforced in the renderer, not the prompt
text.

**2. RAG prompt — `rag.py` `build_chain` / `_prepare_sections`.** Three
labelled sections: *Protocol content* (retrieved chunks, grouped by
`source_site` — pre-Phase-2 chunks without the metadata group fine) + basis
notes; *Public health guidance* (moh.gov.sg chunks only, "public guidance,
not clinic protocol content"); *Local context* (live snapshot: signals,
clinical framing, counselling, clinic + as-of). No snapshot → local section
says "unavailable", chain still works.

**3. Retrieval steering (mechanism B).** `_steer_query` appends
`[related chronic-care protocols: ...]` from active links to the query before
vector search. No re-embedding.

**4. Multi-source corpus — `rag.py` `SOURCES`.**
`primarycarepages.sg` (depth 2) + `moh.gov.sg/others/haze/` (depth 0 — the
site is huge; the haze page is static public guidance). Every chunk gets
`source_site` metadata. `context/prompts.py` + `linkage.py` `pm25`/`psi`
basis notes now cite the MOH page as the citable government source.

**5. MOH extractor — `_isomer_text`.** gov.sg Isomer pages need a
browser-like User-Agent and aggressive boilerplate stripping: `nav`/`footer`/
`header`, the "Back to top" button column, `<details>/<summary>` link boxes
("Other pages in this section", "Related sites", "Useful links"). Dry-run:
~3,049 clean chars, no boilerplate leaks.

**6. Append-only ingest — `ingest_append()` + `POST /api/ingest/append`.**
Skips sites the store already covers, **matched by host** (not URL — see
pitfall below); within a new source, skips chunks whose normalized URL
(lowercase scheme/host/path, no trailing slash, no query/fragment) is stored;
refuses to embed >500 chunks (credit guard vs OpenRouter's 300k-token
per-request limit). The one append of Phase 2: 4 MOH chunks
(3,785 → 3,789), primarycarepages untouched. Full `ingest()` still rebuilds
everything (fresh directory only).

**Pitfall discovered — primarycarepages rewrote its URL structure.**
The site now redirects the old paths (`/healthier-sg/care-protocols/
chronic-care-protocols/...`) to new canonical URLs (`/healthier-sg/
care-protocols/chronic/...`) with case changes, so a re-crawl produces URLs
that match none of the 77 stored source URLs. Per-URL dedupe therefore
treated the whole corpus as "new" (would have re-embedded ~332k tokens —
billed 400 at the 300k limit). Host-level skip makes append idempotent and
credit-safe; refreshing an existing site's content requires a full rebuild.

**7. App integration — `app.py`.** Clinic point via `CONTEXT_LAT` /
`CONTEXT_LON` / `CONTEXT_NAME` (defaults to `context/config.py` TEST_CLINIC).
Snapshot provider is the same 15-min in-memory cache as `GET /api/context`;
a cold cache degrades the prompt (local section "unavailable") and schedules
a background build on the captured app loop — the chat path never blocks.
`/api/status` reports `{"ready", "context": "building"|"cached (Ns old)"}`.

**Verification (2 free-model chats, `nvidia/nemotron-3.5-lightning:free`).**
(1) Haze/asthma: retrieved the asthma protocol AND the MOH haze page;
answer separated *Protocol content* (protocol's own "advise on haze
precautions" line) from *Local context* ("not protocol content… comes from
local population-level signals") from *MOH public guidance* (PSI-band
advice quoted accurately). (2) Heat/DM+CKD: respected the `partial` basis
note — explicitly refused to present ACEi/ARB review or heat/WBGT as
protocol-mandated ("would over-claim beyond the basis notes").

**Audit correction (2026-09-08, post-build).** Chat (1) surfaced that the
Asthma/COPD protocols *do* mention haze ("Advise on haze precautions when
appropriate") — the §12 audit grep word list missed "haze" (10 chunks).
`pm25` basis corrected `derived` → `partial` with a corpus-exact note;
`psi` stays `derived` (no PSI threshold anywhere in protocol text).

## 14. Phase 3 build notes — WIDB (completed 2026-09-08)

Phase 3 delivered the CDA Weekly Infectious Disease Bulletin (`disease_week`)
as a population-level signal, verified live (0 API credits — key-free PDF).

**1. Fetch — `context/widb.py`.** `list_bulletins()` crawls the CDA archive
pages (2026 then 2025, `config.WIDB_ARCHIVE_URL`) and collects every
`EW NN` PDF link on `isomer-user-content.by.gov.sg`. `fetch_latest()`
walks the list newest→oldest, downloads with a browser User-Agent (CDA
serves S3 which 403s the default python UA — `http.get_bytes` now takes
`extra_headers`), and parses the **first PDF that yields a payload**, so a
broken newest file (observed live: `EW 34` intermittently 403s from S3)
falls back to `EW 33` without erroring the snapshot. PDF URLs are
percent-encoded. Disk-cached `widb_latest.json`, TTL 3 days
(`config.WIDB_CACHE_TTL_SECONDS`) — WIDB is weekly, so a 3-day TTL
survives the Sunday publish gap and any archive blip.

**2. Parse — `parse_widb()` (pypdf, already a Phase-5 dependency).**
Verified against EW 33, 32 and 1. Page 1: the master disease table —
per-disease rows `<name> <week> <prev_week> <median same-week 2021–25>`
plus `cum`/`cum_prev` for notifiable diseases (5-number rows; ARI/other
diseases have 3, no cumulative — column mapping verified arithmetically:
EW32 cum + EW33 week == EW33 cum for dengue). All pages are flattened
whitespace-normalized and the key narratives are pulled by regex:
influenza type distribution + ILI positivity, COVID-19 ARI positivity,
top ARI pathogens (adult/paediatric), ARI polyclinic attendances (page 2);
dengue notifications / hospital admissions / serotypes (page 4).
`epi_week` comes from the PDF's `EPIDEMIOLOGICAL WEEK N` header line; the
walk-back ordering uses the `EW NN` in the archive filename.
**Parser pitfalls (all hit and fixed):** (a) the disease row regex
originally allowed digits in names — that backtracked across multi-digit
cells and dropped rows; names now exclude digits. (b) the bulletin wraps
"…in E-\nweek 33" across a line break — the ARI-attendances regex allows
optional whitespace after `E-`. (c) a name-only line whose numbers land on
the following line (e.g. `Mpox#`) is matched by a two-line fallback.

**3. Snapshot — `context/snapshot.py`.** `disease_week` block (national
counts + `dengue` + `ari` + `source`). A fetch failure appends
`widb: <reason>` to `data_gaps` and omits the block — never crashes the
rest of the snapshot.

**4. Prompt — `context/prompts.py`.** One `disease_week` line in *Local
context* (dengue week vs prev vs median, admissions, ILI/ARI/COVID %,
top subtype, HFMD). Labelled national counts, "not protocol content".

**5. CLI + UI.** `python -m context` prints a `DISEASE` section;
`static/index.html` adds a WIDB chip to the strip + an "Infectious diseases
(WIDB, national)" detail card. `node --check` on the page script OK.

**Verification (0 credits).** EW33/EW32/EW1 parse spot-checked by hand
against the PDFs; live `fetch_latest()` → EW34; cache hit on second call;
walk-back + all-fail paths exercised with a monkeypatched `get_bytes`
(deterministic, `/tmp/widb_fallback_test.py`); `/api/context` returns
`disease_week` with `data_gaps: []`; `/api/status` → `ready:true`.

## 15. Phase 4 build notes (URA planning decisions, 2026-09-08)

`context/ura.py` (stdlib, Phase 3 pattern): `get_token()` —
`insertNewToken/v1` with `AccessKey` header, daily token from
`{"Result": ...}`; `fetch_rows()` — `invokeUraDS/v1?service=Planning_Decision&last_dnload_date=dd/mm/yyyy`
(today − 90 d; the API max lookback is 1 year and `year=` is all-year —
the 90-day window keeps the payload small, ~2–3k rows) with `AccessKey` +
`Token` headers; `filter_healthcare()` — case-insensitive keyword match on
`submission_desc` (POLYCLINIC, CLINIC, MEDICAL, NURSING HOME, CHILD CARE,
SENIOR), `delete_ind == 'Yes'` rows dropped, newest `decision_date`
(dd/mm/yyyy → ISO) first; `fetch_catchment()` — 24 h disk cache
(`$TMPDIR/cp_rag_context_cache/ura_planning.json`; data cadence is daily
and the token is daily, so one refetch/day is enough) and the
`(payload, error)` contract. No `URA_ACCESS_KEY` in `.env` →
`ura: URA_ACCESS_KEY not set in .env (Phase 4 signal disabled)` in
`data_gaps` — the signal degrades exactly like every other source, and
the prompt/UI say nothing about planning.

Payload (`catchment_change`): `source`, `window`, `rows_scanned`,
`healthcare_decisions_90d_count` (full count), `healthcare_decisions_90d`
(latest 20: `address/what/date/decision_type/decision_no`), `caveat`.
Provenance: *Local context* only — the prompt line, CLI header, and UI
card all state a written permission is NOT an opened facility.

Verified (monkeypatched transport, 0 URA network): filter/sort/delete-skip,
cache-hit no-refetch, no-key gap, token-HTTP-failure and bad-Status error
paths (`/tmp/verify_ura_task1.py`); prompt line renders with the caveat,
absent block adds nothing (`/tmp/verify_ura_task2.py`); no-key CLI shows
the gap and no `PLANNING` section; seeded-cache CLI shows the section;
live `/api/context` → `ura:` gap only, no crash.

**Live verification (2026-09-08, `URA_ACCESS_KEY` set, plan Appendix A):**
fresh fetch → window 2026-06-10 to 2026-09-08, **2,363 rows scanned,
74 healthcare-related decisions** (≈3 % of all decisions; keyword overlap:
CLINIC 45, MEDICAL 36, CHILD CARE 13, SENIOR 8, POLYCLINIC 4, NURSING HOME 6
— rows can match several). **Observed `decision_type` values across all 74:
`Written Permission` (67), `Authorized Work` (6), `Corrigendum` (1)** — no
refusal-type rows in this window (the code never branches on
`decision_type`, it only displays it, so nothing to change). Note: the API
filters by record *created/modified* date, not decision date — a few
healthcare rows carry `decision_date`s earlier than the window start
(oldest 2026-02-03; e.g. late corrigenda). CLI `PLANNING` section and
`/api/context` `catchment_change` both render the real rows
(`data_gaps: []`); negative check (key commented out) → `ura:` gap + no
`PLANNING` section, key restored. UI: `/` → 200, card markup present.

Follow-ups (not built): catchment geocoding of `address` (OneMap free key
or street→region table) for per-clinic distance; residential-unit counts
(URA's separate `Private_Residential_Properties` services, if the signal
warrants them); rejected-vs-approved split chip in the UI.

## 16. Phase 6 build notes (clinician brief dashboard + Library, 2026-09-09)

Spec: `docs/superpowers/specs/2026-09-09-clinician-brief-library-design.md`
(approved v2); plan: `docs/superpowers/plans/2026-09-09-clinician-brief-library.md`;
SDD ledger: `.superpowers/sdd/2026-09-09-clinician-brief-library/ledger.md`
(local-only, gitignored).

**What was built.** The chat page is now three hash-routed tabs —
`#/brief` (default), `#/chat`, `#/library` (bogus hashes fall back to brief).
The Brief tab is a clinician dashboard: 6 KPI tiles from the live snapshot
(Air / Weather / Dengue / WIDB / Planning / Nearest polyclinic, each with
gap and degraded states per the spec's D10), a Generate/Regenerate CTA, and
the generated brief (headline, up-to-6 watch cards with sources, Outlook,
provenance-drop footer). The Protocol spotlight is rendered 100% from the
structured `snapshot.protocol_links.active` — never from LLM text. The old
context strip survives verbatim in the Brief view as the collapsed
"Raw context data" drill-down (its DOM nodes are never removed — the
15-min auto-refresh intervals depend on them). The Library tab lists the
corpus: totals line, guideline-PDF card grid (title, chunks, source link,
doc-hash prefix), care-protocol pages, and MOH public guidance; the PDF
upload bar moved here from Chat.

**`context/brief.py`** (new, stdlib-only, pure): builds the brief prompt
from the snapshot with a *projection* — `protocol_links` is structurally
absent from the LLM payload (asserted in dry runs). The LLM returns strict
JSON `{headline, watch[], outlook}`.

**`rag.py`**: `generate_brief()` (one chat-model call, no retrieval, no
embeddings), `_parse_brief()` (fence-strip, strict JSON, type coercion,
watch list clipped to 6), `_apply_provenance_guard()` — server-side drop of
any watch item mentioning `protocol`, `guideline`, or a known protocol name
(`provenance_drops` counted in the response).

**`app.py`**: brief cache state (in-memory, keyed on 3-dp clinic coords,
tied to the snapshot's `generated_at`; restart = cold) +
`GET /api/brief` (free, never builds) + `POST /api/brief/generate` —
**the only LLM path outside `/api/chat`**, guard order per spec §7.2:
invalid point → 400; `rag_chain is None` → 503; fresh cache & no force →
cached (0 credits); cached error → 502 (90-s failure sentinel); in-flight
build → await the shared future (D5 stampede guard); force inside 60 s →
429 + `Retry-After` (D6 cooldown); otherwise thread-pool build. 15-min
success TTL. Also `GET /api/library` (Task 1: `rag.py` `list_corpus()` —
503 on `rag_chain is None`, executor scan, 500 on scan failure).

**`static/index.html`**: tabs + hash routing; KPI tile renderer; brief
renderer with degraded states and a parse-error fallback that shows the raw
LLM text; `generateBrief()` — the UI's only brief POST (double-click guard,
staged progress text, button flips to "Regenerate (costs 1 LLM call)" only
when a fresh brief exists); `loadLibrary()`/`renderLibrary()` with the
>25-URL sprawl collapse rule (spec §9).

**Verification (gates).**
- Library API (Task 1): 0 credits; totals 6,764 chunks / 96 guideline PDFs
  (2,975 chunks, matches Phase 5) / 28 protocol pages / 1
  public-guidance page. Pre-check: `web_sources` = primarycarepages.sg 77
  URLs (3,785 chunks; legacy chunks have no `source_site` — site derived
  from URL host) + moh.gov.sg 1 URL / 4 chunks; `/care-protocols/` URLs =
  28 > 25 → Library renders the collapsed card + "49 other pages"
  footnote (sitemap/nav/crawl artifacts — ~22 real protocol pages, the
  depth-2 crawl also pulled preventive + administrative protocols).
- Brief backend (Task 2): dry runs green (prompt projection leaks no
  protocol material; parse/guard mocks — valid / malformed / violating /
  fence-strip / 9→6 clip / coercion). Free endpoints green (stale/none,
  400 bad point). **Paid record: exactly one verification brief** —
  `POST /api/brief/generate` → 200, `deepseek/deepseek-v4-flash-0731`,
  valid JSON, headline + 5 sourced watch items, `provenance_drops: 0`,
  outlook names the PM2.5 gap (~91 s wall = cold snapshot + LLM).
  Forced regenerate → **429 + `Retry-After: 59`** (0 credits);
  no-force after fresh → `cached: true` (0 credits).
- **Credit deviation (recorded):** the plan budgeted 1 paid call; **2 were
  consumed.** The boot-time context warm-build completed mid-verification,
  rotating the 15-min snapshot cache; a no-force POST issued as a "cache
  probe" (status probes should use the FREE `GET /api/brief`) then saw the
  rotated snapshot → correct stale detection → paid rebuild. Cache /
  invalidation behaviour worked as designed; the probe was the deviation.
  No further paid calls were made in the phase.
- UI (Tasks 3/4): `node --check` on the extracted page script; headless DOM
  harnesses against the live server — Brief: 6 tiles, headline, 6 watch
  cards, outlook, spotlight, tab round trip incl. a manual
  `refreshContext()` tick from the Chat tab, bogus hash → brief, zero
  runtime errors; Library (direct reload at `#/library`): 13/13 checks
  (totals vs API, 96 PDF cards + 1 collapse card, "28 pages — collapsed",
  "49 other pages" footnote, MOH row with raw URL + derived title,
  round trip), zero runtime errors. Note: the harness fetch stub always
  issues GET, so a harness Generate click 405s at the server and can never
  trigger a paid build; the 0-credit cached path was proven server-side
  via curl.
- Standing checks (Task 5): server boot, `/api/status` ready, `/` 200,
  `GET /api/brief` no-crash, `GET /api/library` totals, `node --check`,
  `venv/bin/python -m context` exit 0, `git status` clean.

**Cost model (spec §8.3, verbatim):** one brief = at most 2 provider
round-trips (snapshot context fetches are key-free; the only paid step is
the single chat-model call), 0 embeddings, 0 retrieval; success cached 15
min; failure sentinel 90 s; force-regenerate cooldown 60 s.

**Known boundaries:** live PDF upload was NOT exercised this phase
(embedding spend); brief + snapshot caches are lost on server restart (by
design); the depth-2 crawl's URL sprawl is visible in the Library
(collapsed by design).

## 17. Phase 7 build notes — NTUC community calendars (2026-09-10)

**What:** the 26 NTUC Health Active Ageing Centre programme calendars
(centre-specific PDFs linked from the NTUC landing page) are ingested into
the main Chroma collection as a fifth, explicitly non-clinical source.
Prompts carry a `Community resources` section; community answers must be
labelled community activities and may never be cited as protocol or
guideline content.

**Design (decisions):**
- `rag.COMMUNITY_SITES = {"ntuchealth.sg"}`, parallel to the existing
  `PUBLIC_GUIDANCE_SITES` / `GUIDELINE_SITES`; `_prepare_sections()` routes
  on `source_site` metadata into a fifth standalone prompt section,
  `Community resources`, tagged "not clinical content".
- `fetch_community_calendars()` parses centre PDF links from the NTUC landing
  page (no hardcoded centre list — the page is the source of truth).
- `ingest_community_refresh()` is replace-only, scoped to
  `source_site == ntuchealth.sg`: add-then-delete (a broken month never wipes
  existing data), idempotent via `doc_hash` (identical PDFs no-op),
  structural guards (zero PDFs, <100 or >600 chunks all fail closed,
  store untouched).
- `app.py POST /api/community/refresh`: 409 while a run is in flight,
  429 within 60 s of the last success, 502 on structural failure (store
  untouched), 200 with `{"status": "updated"|"up-to-date", "centres",
  "ok", "failed", "chunks_added", "chunks_removed", "calendar_months"}`;
  the chat chain is rebuilt only when the store actually changed.
- Deliberately NOT in `SOURCES`: calendars are not a fetchable corpus
  source — they have their own endpoint, not the general refresh.
- `list_pdfs()` still excludes them (Library PDF list unchanged, 96
  guideline PDFs); `list_corpus()` gains a `community` bucket: centres,
  months, `total_chunks`, `refreshed_at`. `refreshed_at` is written per
  chunk at ingest time and surfaced in the Library bar.

**Retrieval fix (steering gate):** the first community probe returned an
empty `Community resources` section. Cause: `_steer_query()` appends the
active protocol names for the retriever, which biases the embedding toward
clinical chunks — the raw question ranked the Redhill calendar #1, but the
steered query returned zero community chunks in the top 12. Fix:
`_COMMUNITY_INTENT_RE` (ntuc, active-ageing, communit-, programme, activit-,
workshop, class, volunteer, day care, dinner, digital skills, senior,
older people/adults, social activity) makes community-intent questions skip
the steering suffix. Clinical questions are unchanged.

**Verified.** 2026-09-10, 4th session:
- First refresh: 26 centres, 386 chunks, 0 failures, all month=2026-09
  (~15 s, embed-only spend).
- Second refresh: `up-to-date`, 0 chunks added/removed (~10 s, zero
  embedding calls).
- Live NTUC page lists exactly 26 calendar PDFs (matches ingest).
- `/api/library` shows the community bucket with `refreshed_at`;
  `/api/pdfs` still lists the 96 guideline PDFs.
- Community probe (local vLLM, free): "What programmes are available for
  older people at the Redhill Active Ageing Centre this month?" answered
  from the Redhill 2026-09 calendar with named programmes (Dragon Boat,
  Piloxing, Walking Football, Fab Lab workshops, haircuts, pedicure);
  sources = Redhill / Bukit Panjang / Nanyang calendars.
- Clinical negative probe: asthma inhaler technique answered from the
  asthma protocol + ACG guideline; no calendar cited.
- Guards: 409 in-flight, 429 within 60 s cooldown confirmed.

**Known boundaries:** the calendar month is whatever NTUC publishes now
(2026-09 at write time); an October page update changes `calendar_months`
on the next refresh. The intent regex is deliberately narrow — a clinical
question that merely mentions "exercise"/"activities" (e.g. cardiac rehab)
keeps its steering. Refresh replaces the whole centre set from the current
live page; a month's calendars are not retained historically.

## 18. Phase 8 build notes — URA planning zoom (2026-09-10)

User feedback on the Phase 4/6 tile: a bare "N healthcare-related decisions"
is not actionable for a GP; wanted senior-care / nursing-home visibility, a
boundary, and the (static) Nearest polyclinic tile removed.

- **Category buckets** (`context/ura.py`): keyword priority Nursing home >
  Senior care > Child care > Polyclinic > Clinic > Medical > Other;
  `category_counts` in the payload (over all in-window rows, not just the
  listed cap). VETERINARY rows now DROPPED (previously matched CLINIC).
- **Honest 90-day window**: `window_decisions()` re-filters to
  decision_date inside the window (the API's `last_dnload_date` filters on
  record created/modified date — Phase 4 observed decision dates months old).
  Dropped count surfaced as `stale_dropped`.
- **Boundary** (user's choice): street-name → area heuristic, OFFLINE —
  `config.STREET_AREA_HINTS` (first token match, ~55 tokens, 8 area groups),
  enriched in `snapshot.py::_enrich_ura` with `district` / `approx_km` /
  `distance_band` (near ≤ 10 km / mid ≤ 25 / far > 25 to the area CENTROID —
  rough). Unmapped addresses get no fields. Live coverage at build time:
  24/67 in-window rows mapped (19/50 listed rows after the cap; 31 listed
  rows unmapped). Ruled out: OneMap geocoding (API hosts fail DNS on this
  network, §15–§16); Nominatim (reachable, but a new external dependency —
  rejected).
- **Tile** (Brief): "N decisions" sub-line now "k near (~10 km) · a senior
  care · b nursing home · c child care · d clinic/medical · e polyclinic".
  **Card** (raw context): grouped by category, each row tagged
  `[district · ~km]`. **Nearest polyclinic tile REMOVED** (static data, no
  signal); data kept in the RAG prompt Local context + raw-context list.
- **Prompt/brief/CLI**: category split + near-clinic count added to the URA
  Local-context line, brief `planning` projection, and CLI PLANNING.
- **Cache**: `ura_planning_v2.json` (schema bump forces one refetch).
  `URA_MAX_ITEMS` 20 → 50 (near-count + card coverage).
- **Known limitations**: centroid distance ≠ facility distance;
  `near_clinic_count` covers the listed cap (50) only; unmapped addresses:
  31 of 50 listed rows (no street-name hint match); the headline count now
  excludes vet rows and stale-dated rows (in the live 2026-09-10 window:
  4 vet rows and 4 stale-dated rows dropped — 71 → 67 in-window).
- **Live verification** (2026-09-10, window 2026-06-12 → 2026-09-10,
  0 paid credits; all gates ran on seeded caches, one real URA fetch for
  these numbers): 2,361 API rows scanned; 67 in-window decisions —
  5 senior care, 6 nursing home, 12 child care, 3 polyclinic, 38 clinic,
  3 medical; `near_clinic_count` = 2 (bands of the 19 mapped listed rows:
  2 near / 14 mid / 3 far). CLI PLANNING header shows the split + "2 within
  ~10 km", rows tagged `(category) [district ~km]`. `/api/context` live:
  `category_counts` + `near_clinic_count` + per-row `district`/`approx_km`
  present, `data_gaps` empty. Prompt line renders "(5 senior care, 6
  nursing home, 12 child care, 3 polyclinic, 38 clinic, 3 medical), 2
  within ~10 km of the clinic". Brief `planning` projection carries
  `category_counts` + `near_clinic_count`. Tile sub-line renders
  "2 near (~10 km) · 5 senior care · 6 nursing home · 12 child care ·
  41 clinic/medical · 3 polyclinic" (41 = 38 clinic + 3 medical).



## 19. Phase 9 build notes — 2 km catchment + Active Ageing card (2026-09-10)

User request: the 90-day planning view should use a **2 km radius**, its
title should be **layperson-friendly**, and the dashboard should gain an
**Active Ageing Programmes card for NTUC centres within 2 km**.

- **URA near band**: `config.URA_NEAR_KM` 10 → **2.0** — the "near" band
  of the planning view is now the walkable catchment (mid ≤ 25 km and
  far > 25 km unchanged). Rationale: a GP's patient catchment is
  ~1–2 km; the 10 km "near" band made the near-count meaningless
  (everything island-wide was "nearish"). Tile renamed `Planning 90d`
  → **`New health facilities (3 mo)`**; sub-line "k near (~10 km)" →
  "k near (~2 km)"; same change in the prompt line and CLI.
- **NTUC Active Ageing centre locations** (no geocoding needed):
  `ntuchealth.sg/active-ageing/locations` is a Next.js page whose flight
  payload embeds each centre's exact `position` (lat/lon) keyed by
  `name`. Verified live 2026-09-10: 27 centres, all matching the
  calendar-page anchor text exactly. Stored statically in
  `config.ACTIVE_AGING_CENTRES` (27×(name, lat, lon)) — updating is a
  one-line-per-centre change if NTUC adds/renames a centre.
- **NTUC calendar PDF source**: `ntuchealth.sg/active-ageing`
  (landing page) lists, per centre, a monthly programme-calendar PDF at
  `assets.ntuchealth.sg/ae/<centre>-<Mon>-<YYYY>.pdf`, keyed by centre
  name in the anchor text. The PDFs rotate monthly (Sep 2026 at write
  time) and are the SAME source as the Phase 7 RAG corpus — the card
  just links to them; no new corpus content.
- **Browser UA required**: the site is Akamai-fronted; the default
  python UA is 403. `config.NTUC_UA` is a Chrome/124 UA (same
  workaround class as WIDB's `http.py`).
- **`context/community.py`** (new, stdlib, key-free): fetches the
  landing page, regex-parses the anchors (name, month, pdf_url),
  dedupes by centre name, disk-caches 24 h at
  `config.CACHE_DIR/ntuc_ageing_v1.json`, returns
  `{count, calendar_months, centres[], caveat}`. The caveat DYNAMICALLY
  detects shared-PDF quirks: if one pdf_url serves >1 centre name, it
  names them (2026-09-10 live: 'Bukit Batok West' reuses the
  Bedok-North PDF — an NTUC site bug, not ours).
- **Snapshot**: `active_ageing` block = fetched centres +
  `snapshot._enrich_ageing()` (km to 1 dp via haversine to the
  NTUC-published centre position — exact, unlike the URA
  street-heuristic — `near` ≤ `ACTIVE_AGING_NEAR_KM` = 2.0, plus
  `near_count`). A live centre missing from the config table renders
  without `km` (listed, never `near`). Failure → `data_gaps:
  "active_ageing: ..."`, never a crash.
- **Frontend**: new KPI tile **`Active Ageing (2 km)`** (value "N
  centre(s)", sub = the within-2 km names + km); new raw-context card
  **`Active Ageing (NTUC Health, island-wide)`** — all centres
  nearest-first, within-2 km bolded, each row a link to the current
  month's calendar PDF + the caveat line; chip "Active Ageing: N within
  2 km". Drive-by fix: the raw-context "Nearest polyclinics" list read
  `nearest_services.pyclinics` (typo) and silently never rendered —
  now `polyclinics` (same typo class as the `brief.py` `pyclinics`
  projection fix).
- **Prompt**: Local context gains an NTUC line — N centres island-wide
  (calendar month), the near ones named with km, explicitly
  **NON-clinical**, calendar questions answered from the Community
  resources corpus section, never as clinical services.
- **Out of scope (deliberate)**: the clinician brief (brief.py) does
  NOT project `active_ageing` — its source whitelist
  ("NEA | data.gov.sg | CDA WIDB | URA | derived") has no NTUC and the
  brief stays clinical. If wanted: add "NTUC" to the whitelist + a
  small projection block (data already in the snapshot).
- **Live verification** (2026-09-10, 0 paid credits): one real NTUC
  fetch → 27 centres, month "Sep 2026", live-name set == config set
  (0 unmapped, 0 orphans); at the test clinic `near_count` = 3 (Jurong
  Central Plaza 0.8, Boon Lay 1.1, Taman Jurong 1.2 km; next-closest
  Gek Poh 2.2, Pioneer 2.3 — correctly outside 2 km); shared-PDF caveat
  rendered for Bukit Batok West. `node --check` on the page JS;
  py_compile on all touched modules.

