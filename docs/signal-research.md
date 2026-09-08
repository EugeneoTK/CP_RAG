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
| 4 — URA (optional) | F6 planning-decision catchment block (`URA_ACCESS_KEY` in .env) | 0 credits |
| 5 — ACE guidelines | Clinical-guidelines PDFs in the RAG corpus: `pypdf` extraction, append-only `ingest_pdf()` with hash dedupe, `POST /api/ingest-pdf` + `GET /api/pdfs`, 4th prompt section "Clinical guidelines", UI upload bar, optional `scripts/ace_guidelines.py` sitemap crawler for all 29 ACE ACGs. Plan: `docs/superpowers/plans/2026-09-08-ace-guidelines-pdf.md` — **DONE 2026-09-08** (manual-upload path + full seed: 96 guideline PDFs / 2,975 guideline chunks in the store; 1 scanned appendix uningested) | ~1–2 credits (upload path) / a few cents (full ACE seed) |

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

