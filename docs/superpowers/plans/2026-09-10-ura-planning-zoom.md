# URA Planning Zoom (Phase 8) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the "Planning 90d" brief tile and the URA planning card useful to a GP: facility-type categories (senior care / nursing home / child care / clinic), a street-name-to-district heuristic with rough distance bands, an honest 90-day decision-date filter, veterinary-noise removal, and removal of the non-signal "Nearest polyclinic" KPI tile.

**Architecture:** `context/ura.py` classifies each parsed decision (keyword priority, veterinary excluded) and re-filters to decisions actually dated inside the 90-day window (the URA API window is record *modified* date, not decision date — observed decision dates up to 5 months old). `context/snapshot.py` enriches each listed row with an area label + rough km from a street-name hint table (offline; no geocoder is reachable on this network — OneMap API hosts fail DNS, verified 2026-09-10). UI tile/card, RAG prompt line, brief projection, and CLI print the new fields. No new dependencies, no new keys.

**Tech Stack:** Python 3 stdlib only (`context/` package), one static HTML file (no framework, no build step), existing `context/geo.py::haversine_km`, URA e-Services API (`URA_ACCESS_KEY` already in gitignored `.env`, 0 credits).

**Spec:** none on disk — requirements from the 2026-09-10 session: "75 decisions makes no sense for the doctor"; show "approved Senior care centre / nursing home in last 90 days"; boundary via street-name-to-district heuristic (user's choice over island-wide-only and Nominatim geocoding); the Nearest polyclinic card is not useful (remove the tile, keep the data in the prompt + raw-context list).

## Global Constraints

- `context/` package is **stdlib-only** — no new imports beyond what `context/` already uses.
- A failing source never aborts the snapshot build — it lands in `data_gaps`.
- **No `git commit` in this plan.** The working tree already carries uncommitted Phase 7 WIP in the same files (`static/index.html`, `CLAUDE.md`, `docs/signal-research.md`, `HANDOFF.md`, plus `rag.py`, `app.py`, `.env.example`); prior sessions deliberately leave commits to the user (HANDOFF Outstanding #1). Each task ends with a diff-review step instead. Suggested user commit when they get to it: `feat: URA planning zoom — categories, street-area heuristic, polyclinic tile removed (Phase 8)`.
- 0 paid API credits and 0 paid LLM calls in every gate: seeded caches, local vLLM only, or no network.
- UI gate: `sed -n '/<script>/,/<\/script>/p' static/index.html | sed '1d;$d' > /tmp/ctx_page.js && node --check /tmp/ctx_page.js && echo "JS OK"`.
- The URA line stays *Local context* in the RAG prompt — never protocol content; the "permission ≠ opened facility" caveat survives everywhere it rendered before.
- URA disk cache file bumps `ura_planning.json` → `ura_planning_v2.json` (payload schema changed; forces exactly one refetch; 0 credits).
- `URA_NEAR_KM = 10.0` — the "near clinic" band, measured against a coarse area centroid (documented as rough everywhere rendered).
- Cache dir: `$TMPDIR/cp_rag_context_cache/`. Server: uvicorn 127.0.0.1:5001, log `/tmp/cp_rag_server.log`; `/api/context` has a 15-min in-memory cache — restart the server after `context/` changes before live checks.
- `venv/bin/python` is the interpreter for all Python gates.

### Task 1: Categories, vet exclusion, decision-date window (`context/config.py` + `context/ura.py`)

**Files:**
- Modify: `context/config.py` — URA block (lines 32–40)
- Modify: `context/ura.py` (whole file, 154 lines)

**Interfaces:**
- Produces (Tasks 2–4 consume these exact names): payload keys `category_counts` (dict of category → int, over ALL in-window decisions, not just the listed cap) and `stale_dropped` (int); each row in `healthcare_decisions_90d` gains `"category"` ∈ {`Senior care`, `Nursing home`, `Child care`, `Polyclinic`, `Clinic`, `Medical`, `Other`}; function `ura.window_decisions(decisions, window_days, today=None) -> (in_window_list, stale_count)`; cache file `ura_planning_v2.json`; config names `URA_NEAR_KM`, `STREET_AREA_HINTS`, `URA_MAX_ITEMS` (now 50).
- Consumes: nothing new.

- [ ] **Step 1: Run the failing check**

```bash
venv/bin/python -c "from context import ura; ura.window_decisions([], 90)"
```

Expected: `AttributeError: module 'context.ura' has no attribute 'window_decisions'`

- [ ] **Step 2: `context/config.py` — replace the URA block (lines 32–40).** `URA_BASE`, `URA_WINDOW_DAYS`, `URA_CACHE_TTL_SECONDS`, `URA_HEALTHCARE_KEYWORDS` keep their current values; `URA_MAX_ITEMS` becomes `50` (comment: `# snapshot cap on the decision list (Phase 8: 20->50 so category + near counts cover more rows)`). ADD after `URA_HEALTHCARE_KEYWORDS`:

```python
# Phase 8: "near clinic" band for the street-area heuristic (rough: distance
# to an area centroid, not the facility).
URA_NEAR_KM = 10.0
# Phase 8: street-name -> (area label, approx centroid). First hint whose token
# appears in the UPPERCASE address wins. Coarse, documented heuristic — no
# geocoder is reachable on this network (docs/signal-research.md §15–§16).
STREET_AREA_HINTS = (
    # Central (Orchard / Tanjong Pagar / Chinatown / River Valley / Kitchener)
    ("ORCHARD", "Central", 1.3042, 103.8318),
    ("CATHAY", "Central", 1.3042, 103.8318),
    ("STAMFORD", "Central", 1.3042, 103.8318),
    ("RAFFLES", "Central", 1.2910, 103.8520),
    ("RIVER VALLEY", "Central", 1.2962, 103.8405),
    ("KITCHENER", "Central", 1.3098, 103.8521),
    ("NORTH BRIDGE", "Central", 1.2910, 103.8510),
    ("BRAS BASAH", "Central", 1.2975, 103.8505),
    ("PLAYFAIR", "Central", 1.2930, 103.8440),
    ("TANJONG PAGAR", "Central", 1.2760, 103.8440),
    ("TANJONG KATONG", "Central", 1.3080, 103.8540),
    ("MAXWELL", "Central", 1.2800, 103.8460),
    ("OUTRAM", "Central", 1.2800, 103.8440),
    ("CHINATOWN", "Central", 1.2840, 103.8450),
    ("PECK SIAH", "Central", 1.2790, 103.8430),
    ("BENCOOLEN", "Central", 1.2870, 103.8400),
    # Bukit Merah / one-north
    ("BUKIT MERAH", "Bukit Merah / one-north", 1.3010, 103.7950),
    ("ONE-NORTH", "Bukit Merah / one-north", 1.3000, 103.7900),
    ("KALLAWAY", "Bukit Merah / one-north", 1.3160, 103.7930),
    # Bedok / Changi
    ("BEDOK", "Bedok / Changi", 1.3250, 103.9300),
    ("CHANGI", "Bedok / Changi", 1.3450, 103.9550),
    ("TAN TONG", "Bedok / Changi", 1.3310, 103.9450),
    ("MARSILING", "Bedok / Changi", 1.3250, 103.9520),
    ("CHAI CHEE", "Bedok / Changi", 1.3260, 103.9190),
    ("JOO CHIAT", "Bedok / Changi", 1.3140, 103.8950),
    # Queenstown / Redhill
    ("QUEENSTOWN", "Queenstown / Redhill", 1.2960, 103.8150),
    ("REDHILL", "Queenstown / Redhill", 1.3020, 103.8130),
    ("TAMAY", "Queenstown / Redhill", 1.2920, 103.8170),
    ("HOLLIS", "Queenstown / Redhill", 1.3050, 103.8150),
    # West (Clementi / Jurong / Bukit Batok)
    ("BUKIT BATOK", "West (Clementi / Jurong)", 1.3500, 103.7500),
    ("CLEMENTI", "West (Clementi / Jurong)", 1.3140, 103.7650),
    ("JURONG", "West (Clementi / Jurong)", 1.3400, 103.7100),
    ("BOON LAY", "West (Clementi / Jurong)", 1.3380, 103.7070),
    ("TENNYSON", "West (Clementi / Jurong)", 1.3220, 103.7700),
    ("GARDEN CITY", "West (Clementi / Jurong)", 1.3180, 103.7720),
    ("CHOA CHU KANG", "West (Clementi / Jurong)", 1.3870, 103.7460),
    # North (Woodlands / Yishun / Hougang)
    ("WOODLANDS", "North (Woodlands / Yishun)", 1.4380, 103.7860),
    ("YISHUN", "North (Woodlands / Yishun)", 1.4290, 103.8360),
    ("HOUGANG", "North (Woodlands / Yishun)", 1.3710, 103.8920),
    ("KUNINGAL", "North (Woodlands / Yishun)", 1.3560, 103.9060),
    ("TAI THONG", "North (Woodlands / Yishun)", 1.3730, 103.8950),
    ("SEMAKA", "North (Woodlands / Yishun)", 1.3670, 103.9010),
    # East (Sengkang / Punggol / Tampines)
    ("TAMPINES", "East (Sengkang / Punggol / Tampines)", 1.3530, 103.9440),
    ("SIMEI", "East (Sengkang / Punggol / Tampines)", 1.3460, 103.9560),
    ("SIMPANG", "East (Sengkang / Punggol / Tampines)", 1.4000, 103.9200),
    ("SENGKANG", "East (Sengkang / Punggol / Tampines)", 1.3960, 103.9600),
    ("PUNGGOL", "East (Sengkang / Punggol / Tampines)", 1.4000, 103.9100),
    ("LOMONDA", "East (Sengkang / Punggol / Tampines)", 1.3900, 103.9100),
    ("CHANDIGARTH", "East (Sengkang / Punggol / Tampines)", 1.3850, 103.9050),
)
```

- [ ] **Step 3: `context/ura.py` — replace the module docstring's "Scope (v1, island-wide)" paragraph (lines 14–20) with:**

```
Scope: rows carry street addresses with NO postcode/region and no geocoder is
reachable on this network (OneMap API hosts fail DNS, §15–§16), so the block
lists healthcare-related decisions island-wide. Phase 8 (2026-09-10): each row
gets a `category` (keyword priority: Nursing home > Senior care > Child care >
Polyclinic > Clinic > Medical > Other); VETERINARY rows are dropped (a vet
clinic is not a healthcare decision for a GP audience); rows are re-filtered
to decision_date inside the window (the API window is record MODIFIED date,
not decision date); snapshot.py adds a street-name area label + rough km band
(`STREET_AREA_HINTS`, coarse centroid distance). Labelled "permission is NOT
an opened facility" everywhere it is rendered.
```

- [ ] **Step 4: `context/ura.py` — after `_fmt_date`, add the category rules + window function; replace `_parse_row` and `filter_healthcare` with:**

```python
# First match wins.
_CATEGORY_RULES = (
    ("NURSING HOME", "Nursing home"),
    ("SENIOR", "Senior care"),
    ("CHILD CARE", "Child care"),
    ("CHILDCARE", "Child care"),
    ("POLYCLINIC", "Polyclinic"),
    ("CLINIC", "Clinic"),
    ("MEDICAL", "Medical"),
)


def _category(desc):
    """Facility-type label for a submission_desc (first rule that matches)."""
    d = (desc or "").upper()
    for key, cat in _CATEGORY_RULES:
        if key in d:
            return cat
    return None


def _parse_row(row):
    return {
        "address": (row.get("address") or "").strip(),
        "what": re.sub(r"\s+", " ", row.get("submission_desc") or "").strip(),
        "date": _fmt_date(row.get("decision_date")),
        "decision_type": (row.get("decision_type") or "").strip(),
        "decision_no": row.get("decision_no") or None,
        "category": _category(row.get("submission_desc")) or "Other",
    }


def filter_healthcare(rows):
    """Raw Planning_Decision rows -> parsed healthcare-related decisions,
    newest decision_date first. Deleted records (delete_ind == 'Yes'),
    veterinary rows, and non-healthcare rows are dropped."""
    out = []
    for r in rows or []:
        if (r.get("delete_ind") or "").strip() == "Yes":
            continue
        desc = r.get("submission_desc") or ""
        if "VETERINARY" in desc.upper():
            continue
        if _match_healthcare(desc):
            out.append(_parse_row(r))
    out.sort(key=lambda x: x["date"] or "", reverse=True)
    return out


def window_decisions(decisions, window_days, today=None):
    """Keep decisions whose decision_date falls in [today-window_days, today].
    Unparseable dates are kept (best effort, documented). Returns
    (in_window_list, stale_count)."""
    today = today or datetime.date.today()
    since = today - datetime.timedelta(days=window_days)
    kept, stale = [], 0
    for a in decisions:
        try:
            d = datetime.date.fromisoformat(a["date"])
        except (TypeError, ValueError):
            kept.append(a)
            continue
        if since <= d <= today:
            kept.append(a)
        else:
            stale += 1
    return kept, stale
```

- [ ] **Step 5: `context/ura.py` — in `fetch_catchment`, replace the block from `decisions = filter_healthcare(rows)` through the `payload = {…}` literal with:**

```python
    decisions = filter_healthcare(rows)
    in_window, stale = window_decisions(decisions, config.URA_WINDOW_DAYS)
    today = datetime.date.today()
    since = today - datetime.timedelta(days=config.URA_WINDOW_DAYS)
    counts = {}
    for a in in_window:
        counts[a["category"]] = counts.get(a["category"], 0) + 1
    payload = {
        "source": "URA Planning_Decision (written permissions), %d-day window"
                  % config.URA_WINDOW_DAYS,
        "window": "%s to %s" % (since.isoformat(), today.isoformat()),
        "rows_scanned": len(rows),
        "healthcare_decisions_90d_count": len(in_window),
        "healthcare_decisions_90d": in_window[:config.URA_MAX_ITEMS],
        "category_counts": counts,
        "stale_dropped": stale,
        "caveat": ("URA written permission = planning approval, not an opened "
                   "facility; island-wide; district labels are a street-name "
                   "heuristic (coarse, some addresses unmapped)"),
    }
```

Also in `_cache_path()`: `ura_planning.json` → `ura_planning_v2.json`.

- [ ] **Step 6: Run the passing checks** (must print OK):

```bash
venv/bin/python - <<'PY'
from context import ura
assert ura._category("PROPOSED CHANGE OF USE TO SENIOR CARE CENTRE") == "Senior care"
assert ura._category("PROPOSED CHANGE OF USE TO NURSING HOME") == "Nursing home"
assert ura._category("CHANGE OF USE TO MEDICAL CLINIC") == "Clinic"
assert ura._category("VETERINARY CLINIC") == "Clinic"
assert ura._category("SOMETHING ELSE MEDICAL") == "Medical"
rows = [
    {"submission_desc": "CONTINUED USE AS VETERINARY CLINIC", "delete_ind": "No",
     "decision_date": "01/09/2026", "address": "1 X RD", "decision_type": "Written Permission"},
    {"submission_desc": "PROPOSED CHANGE OF USE TO SENIOR CARE", "delete_ind": "No",
     "decision_date": "01/09/2026", "address": "21 TAMPINES ST 1", "decision_type": "Written Permission"},
    {"submission_desc": "CHANGE OF USE TO MEDICAL CLINIC", "delete_ind": "No",
     "decision_date": "01/01/2026", "address": "304 ORCHARD RD", "decision_type": "Written Permission"},
]
got = ura.filter_healthcare(rows)
assert len(got) == 2 and [g["category"] for g in got] == ["Senior care", "Clinic"], got
in_w, stale = ura.window_decisions(got, 90)
assert [a["address"] for a in in_w] == ["21 TAMPINES ST 1"] and stale == 1, (in_w, stale)
print("OK")
PY
```

(With today = 2026-09-10: the 2026-01-01 decision is stale; the veterinary row is dropped by `filter_healthcare`.)

- [ ] **Step 7: Seeded-cache CLI gate** — seed the v2 cache and confirm the CLI renders it with NO refetch (0 credits):

```bash
CACHE="$TMPDIR/cp_rag_context_cache"
cat > "$CACHE/ura_planning_v2.json" <<'JSON'
{"source": "seed", "window": "2026-06-12 to 2026-09-10", "rows_scanned": 2,
 "healthcare_decisions_90d_count": 2,
 "healthcare_decisions_90d": [
  {"address": "21 TAMPINES ST 1", "what": "PROPOSED CHANGE OF USE TO SENIOR CARE CENTRE",
   "date": "2026-09-01", "decision_type": "Written Permission", "decision_no": "P1", "category": "Senior care"},
  {"address": "304 ORCHARD ROAD", "what": "PROPOSED CHANGE OF USE TO MEDICAL CLINIC",
   "date": "2026-08-20", "decision_type": "Written Permission", "decision_no": "P2", "category": "Clinic"}],
 "category_counts": {"Senior care": 1, "Clinic": 1}, "stale_dropped": 0, "caveat": "seed"}
JSON
stat -f %m "$CACHE/ura_planning_v2.json" > /tmp/ura_mtime_before
venv/bin/python -m context | sed -n '/PLANNING/,/^$/p'
test "$(stat -f %m "$CACHE/ura_planning_v2.json")" = "$(cat /tmp/ura_mtime_before)" && echo "no refetch OK"
```

Expected: `PLANNING` section prints both rows (the CLI per-row format is still the old one — Task 4 updates it), then `no refetch OK`.

- [ ] **Step 8: Diff review** — `git --no-pager diff context/config.py context/ura.py` — sanity-check nothing outside the URA block changed. NO COMMIT (Global Constraints).

---

### Task 2: Street-name district enrichment in `context/snapshot.py`

**Files:**
- Modify: `context/snapshot.py` (helpers after `_nearest_polyclinics`, ~line 65; the URA block at lines 185–191)

**Interfaces:**
- Consumes: `config.STREET_AREA_HINTS`, `config.URA_NEAR_KM`, `geo.haversine_km` (already imported), Task 1 payload (rows with `address`, `category`).
- Produces (Tasks 3–4 consume): each row in `snap["catchment_change"]["healthcare_decisions_90d"]` gains `district` (str, only when mapped), `approx_km` (float, 1 dp), `distance_band` (`near` ≤ URA_NEAR_KM / `mid` ≤ 25 / `far` > 25, only when mapped); the block gains `near_clinic_count` (int, listed rows with band `near`). Unmapped rows gain none of the three fields.

- [ ] **Step 1: Run the failing check**

```bash
venv/bin/python -c "from context import snapshot; snapshot._area_for_address('ORCHARD RD')"
```

Expected: `AttributeError: module 'context.snapshot' has no attribute '_area_for_address'`

- [ ] **Step 2: `context/snapshot.py` — add after `_nearest_polyclinics` (line ~65):**

```python
def _area_for_address(address):
    """Street-name heuristic -> (label, lat, lon) or None. First hint whose
    token appears in the uppercase address wins. Coarse — see
    config.STREET_AREA_HINTS docstring."""
    a = (address or "").upper()
    for token, label, lat, lon in config.STREET_AREA_HINTS:
        if token in a:
            return label, lat, lon
    return None


def _enrich_ura(cc, lat, lon):
    """Phase 8: add district / approx_km / distance_band to each listed
    decision (distance to the AREA centroid — rough, documented) and set
    cc["near_clinic_count"]. Mutates and returns cc."""
    near = 0
    for a in cc.get("healthcare_decisions_90d") or []:
        hit = _area_for_address(a.get("address"))
        if not hit:
            continue
        label, alat, alon = hit
        km = round(haversine_km(lat, lon, alat, alon), 1)
        a["district"] = label
        a["approx_km"] = km
        a["distance_band"] = ("near" if km <= config.URA_NEAR_KM
                              else "mid" if km <= 25 else "far")
        if km <= config.URA_NEAR_KM:
            near += 1
    cc["near_clinic_count"] = near
    return cc
```

- [ ] **Step 3: `context/snapshot.py` — in `build_snapshot`, replace line 191** `snap["catchment_change"] = cc` **with:**

```python
        snap["catchment_change"] = _enrich_ura(cc, lat, lon)
```

- [ ] **Step 4: Run the passing checks** (must print OK):

```bash
venv/bin/python - <<'PY'
from context import snapshot, config
assert snapshot._area_for_address("21 Blk 1A TAMPINES AVENUE 1") == \
    ("East (Sengkang / Punggol / Tampines)", 1.3530, 103.9440)
assert snapshot._area_for_address("50 BUKIT BATOK WEST AVENUE 3")[0] == "West (Clementi / Jurong)"
assert snapshot._area_for_address("1 MYSTERY CLOSE") is None
cc = {"healthcare_decisions_90d": [
        {"address": "288 RIVER VALLEY ROAD"},
        {"address": "21 TAMPINES ST 1"},
        {"address": "1 MYSTERY CLOSE"}],
      "near_clinic_count": None}
snapshot._enrich_ura(cc, config.TEST_CLINIC["lat"], config.TEST_CLINIC["lon"])
r0, r1, r2 = cc["healthcare_decisions_90d"]
assert r0["district"] == "Central" and r0["distance_band"] in ("near", "mid", "far")
assert r1["district"].startswith("East") and r1["approx_km"] == round(r1["approx_km"], 1)
assert "district" not in r2
assert cc["near_clinic_count"] in (0, 1)
print("OK")
PY
```

- [ ] **Step 5: Seeded-cache CLI + coverage gate** (reuses the seed from Task 1 Step 7; re-seed if missing):

```bash
venv/bin/python -m context | sed -n '/PLANNING/,/^$/p'
venv/bin/python - <<'PY'
import json, os, tempfile
from context import snapshot
d = json.load(open(os.path.join(tempfile.gettempdir(), "cp_rag_context_cache", "ura_planning_v2.json")))
rows = d["healthcare_decisions_90d"]
mapped = sum(1 for r in rows if snapshot._area_for_address(r["address"]))
print("seed coverage: %d/%d mapped" % (mapped, len(rows)))
PY
```

Expected: the 2 seeded rows render (CLI per-row format unchanged until Task 4); seed coverage `2/2 mapped`.

Then, once the 24 h v2 cache holds a REAL fetch (first run after the seed expires or is deleted), measure live coverage and record the number in Task 4 Step 5:

```bash
venv/bin/python - <<'PY'
import json, os, tempfile
from context import snapshot
d = json.load(open(os.path.join(tempfile.gettempdir(), "cp_rag_context_cache", "ura_planning_v2.json")))
rows = d["healthcare_decisions_90d"]
mapped = sum(1 for r in rows if snapshot._area_for_address(r["address"]))
print("live coverage: %d/%d listed rows mapped" % (mapped, len(rows)))
PY
```

- [ ] **Step 6: Diff review** — `git --no-pager diff context/snapshot.py`. NO COMMIT.

---

### Task 3: UI — category sub-line, grouped card, polyclinic tile removed (`static/index.html`)

**Files:**
- Modify: `static/index.html` — brief tiles (~lines 1139, 1190–1203); context-strip Planning card (~lines 1069–1077)

**Interfaces:**
- Consumes: Task 1+2 snapshot fields (`cc.category_counts`, `cc.near_clinic_count`, row `category`/`district`/`approx_km`).
- Produces: rendered UI only — no code interface. The "Nearest polyclinics" raw-context list (lines ~1084–1088) and the polyclinic chip in the context strip are UNCHANGED.

- [ ] **Step 1: Run the failing check** (old tile code present, new absent):

```bash
grep -n "Nearest polyclinic" static/index.html   # expect the tile lines ~1200-1202
grep -n "category_counts" static/index.html       # expect: no match
```

- [ ] **Step 2: Replace the Planning 90d tile** (lines ~1190–1196, from `// Planning 90d — informational` through the `else tiles += tileHtml('Planning 90d', '', 'no signal', ...)` line) **with:**

```js
    // Planning 90d — informational; sub: category breakdown + near count
    // ("~10 km" = config.URA_NEAR_KM; district heuristic is rough)
    const pGap = gap(['ura']);
    if (pGap) tiles += tileHtml('Planning 90d', 'warn', 'unavailable', escapeHtml(pGap));
    else if (cc.healthcare_decisions_90d_count != null) {
      const k = cc.category_counts || {};
      const bits = [];
      if (k['Senior care']) bits.push(k['Senior care'] + ' senior care');
      if (k['Nursing home']) bits.push(k['Nursing home'] + ' nursing home');
      if (k['Child care']) bits.push(k['Child care'] + ' child care');
      const cm = (k['Clinic'] || 0) + (k['Medical'] || 0) + (k['Other'] || 0);
      if (cm) bits.push(cm + ' clinic/medical');
      if (k['Polyclinic']) bits.push(k['Polyclinic'] + ' polyclinic');
      const near = cc.near_clinic_count ? cc.near_clinic_count + ' near (~10 km) · ' : '';
      tiles += tileHtml('Planning 90d', '', cc.healthcare_decisions_90d_count + ' decisions',
        near + (bits.join(' · ') || 'permission ≠ opened facility'));
    } else tiles += tileHtml('Planning 90d', '', 'no signal', 'URA unavailable');
```

- [ ] **Step 3: Remove the Nearest polyclinic tile** (lines ~1197–1202, from `// Nearest polyclinic — neutral: name + km` through `else tiles += tileHtml('Nearest polyclinic', '', 'no data', '');`) — delete the whole block and replace it with a comment:

```js
    // Nearest polyclinic tile removed 2026-09-10 (Phase 8): static data, no
    // signal. The data stays in the RAG prompt (Local context) and the
    // "Nearest polyclinics" list in the raw-context drill-down below.
```

Also in the tile function's locals (line ~1139), remove `ns = snap.nearest_services || {}` from the `const` declaration (it is used only by the removed tile).

- [ ] **Step 4: Replace the context-strip Planning card** (lines ~1069–1077, from `let uraHtml = ...` through the `} else uraHtml += ...` line) **with:**

```js
    let uraHtml = '<h3>Planning decisions (URA, island-wide)</h3>';
    if (cc.healthcare_decisions_90d_count != null) {
      uraHtml += `<div class="row dim">${escapeHtml(cc.window || '')} — ${cc.rows_scanned != null ? cc.rows_scanned : '?'} written permissions scanned · district = street-name heuristic (coarse)</div>`;
      const hcd = cc.healthcare_decisions_90d || [];
      if (hcd.length) {
        const order = ['Senior care', 'Nursing home', 'Child care', 'Polyclinic', 'Clinic', 'Medical', 'Other'];
        for (const cat of order) {
          const grp = hcd.filter(a => (a.category || 'Other') === cat);
          if (!grp.length) continue;
          uraHtml += `<div class="row" style="font-weight:600">${cat} (${grp.length})</div>`;
          uraHtml += grp.map(a =>
            `<div class="row">• ${escapeHtml(a.date || '?')}: ${escapeHtml(a.address || '?')}${a.district ? ' <span class="dim">[' + escapeHtml(a.district) + ' · ~' + a.approx_km + ' km]</span>' : ''} — ${escapeHtml((a.what || '?').slice(0, 90))}${a.decision_type ? ' <span class="dim">[' + escapeHtml(a.decision_type) + ']</span>' : ''}</div>`).join('');
        }
      } else uraHtml += '<div class="row dim">• no healthcare-related decisions in window</div>';
      uraHtml += `<div class="row dim">${escapeHtml(cc.caveat || 'planning permission ≠ opened facility')}</div>`;
    } else uraHtml += '<div class="row dim">unavailable (see data gaps)</div>';
```

- [ ] **Step 5: Gates** — JS syntax + live page:

```bash
sed -n '/<script>/,/<\/script>/p' static/index.html | sed '1d;$d' > /tmp/ctx_page.js && node --check /tmp/ctx_page.js && echo "JS OK"
grep -c "Nearest polyclinic" static/index.html    # expect 0 (tile gone; raw list header is "Nearest polyclinics" — if this greps 1, it must be that raw-list <h3>, verify by eye)
```

Then restart the server (see Global Constraints) and verify live with the seeded cache still in place:

```bash
curl -s -o /dev/null -w '%{http_code}\n' localhost:5001/          # expect 200
curl -s localhost:5001/api/context | venv/bin/python -c "import json,sys; cc=json.load(sys.stdin)['catchment_change']; print(cc['category_counts'], cc['near_clinic_count'], [ (r['address'], r.get('district'), r.get('approx_km')) for r in cc['healthcare_decisions_90d'] ])"
```

Expected: `{'Senior care': 1, 'Clinic': 1}` and `near_clinic_count` = **0** (from the Woodlands test clinic, both seed rows land `mid`: Central ≈13.3 km, Tampines ≈24.9 km — both > 10 km and ≤ 25 km), and both rows carry `district` + `approx_km`. If the near count differs, re-derive from `haversine_km` before "fixing" anything — the assertion is on the mechanic, not the number.

- [ ] **Step 6: Diff review** — `git --no-pager diff static/index.html` — confirm ONLY the tile block, the tile locals line, and the uraHtml card changed (the file also carries uncommitted Phase 7 WIP — review carefully that the Phase 7 hunks are untouched). NO COMMIT.

---

### Task 4: RAG prompt line, brief projection, CLI output, docs

**Files:**
- Modify: `context/prompts.py` — URA block (lines 55–72)
- Modify: `context/brief.py` — `_project` planning block (lines 78–82)
- Modify: `context/__main__.py` — PLANNING section (lines 118–128)
- Modify: `docs/signal-research.md` — add `## 18. Phase 8 build notes — URA planning zoom (2026-09-10)` after §17
- Modify: `CLAUDE.md` — context-layer bullet (line ~22) + Phase 6 bullet (line ~21)
- Modify: `HANDOFF.md` — full rewrite (repo convention)

**Interfaces:**
- Consumes: all Task 1–3 fields (`category_counts`, `near_clinic_count`, row `category`/`district`/`approx_km`/`distance_band`).
- Produces: no new code interface; docs record the live verification numbers.

- [ ] **Step 1: Run the failing check** (entry point is `prompts.format_live_context(snapshot)` → returns `(basis_notes, local_context, clinic, as_of)`):

```bash
venv/bin/python - <<'PY'
import json, os, tempfile
from context import prompts
d = json.load(open(os.path.join(tempfile.gettempdir(), "cp_rag_context_cache", "ura_planning_v2.json")))
snap = {"meta": {"clinic": {"name": "seed", "lat": 1.345, "lon": 103.718}},
        "catchment_change": d, "data_gaps": []}
basis, local, clinic, as_of = prompts.format_live_context(snap)
assert "1 senior care" in local and "1 clinic" in local, local[-800:]
print("prompt line OK")
PY
```

First run must FAIL (current URA line has no category counts).

- [ ] **Step 2: `context/prompts.py` — replace the URA block (lines 55–72) with:**

```python
    # URA (Phase 4/8): island-wide planning-decision signal — baseline context,
    # same treatment as WIDB; a written permission is explicitly NOT an
    # opened facility, so the line says so. Phase 8 adds the category split
    # and the (rough) near-clinic count.
    cc = snapshot.get("catchment_change") or {}
    if cc.get("healthcare_decisions_90d_count"):
        examples = "; ".join(
            "%s %s — %s%s" % (
                a.get("date") or "?", a.get("address") or "?",
                (a.get("what") or "?")[:80],
                (" [" + a["decision_type"] + "]") if a.get("decision_type") else "")
            for a in (cc.get("healthcare_decisions_90d") or [])[:3])
        k = cc.get("category_counts") or {}
        bits = ", ".join("%d %s" % (k[c], c.lower())
                         for c in ("Senior care", "Nursing home", "Child care",
                                   "Polyclinic", "Clinic", "Medical", "Other")
                         if k.get(c))
        extra = (" (%s)" % bits) if bits else ""
        if cc.get("near_clinic_count"):
            extra += ", %d within ~10 km of the clinic" % cc["near_clinic_count"]
        local_lines = local_lines + [
            "- URA planning decisions %s (island-wide written permissions, "
            "NOT protocol content): %d healthcare-related%s, e.g. %s. A written "
            "permission is NOT an opened facility — never present one as an "
            "existing service." % (
                cc.get("window") or "last 90 days",
                cc["healthcare_decisions_90d_count"], extra, examples)]
```

- [ ] **Step 3: `context/brief.py` — replace the `planning` dict in `_project` (lines 78–82) with:**

```python
        "planning": {"window": cc.get("window"),
                     "healthcare_decisions_90d_count":
                         cc.get("healthcare_decisions_90d_count"),
                     "category_counts": cc.get("category_counts"),
                     "near_clinic_count": cc.get("near_clinic_count"),
                     "recent_decisions":
                         _clip_dicts(cc.get("healthcare_decisions_90d"), 3)},
```

(rows already carry `district`/`approx_km`/`distance_band` after Task 2 — they flow through `_clip_dicts` unchanged.)

- [ ] **Step 4: `context/__main__.py` — replace the PLANNING section (lines 118–128) with:**

```python
    cc = snap.get("catchment_change") or {}
    if cc.get("healthcare_decisions_90d_count") is not None:
        counts = cc.get("category_counts") or {}
        count_str = ", ".join("%d %s" % (counts[k], k.lower())
                              for k in ("Senior care", "Nursing home", "Child care",
                                        "Polyclinic", "Clinic", "Medical", "Other")
                              if counts.get(k))
        print("PLANNING URA written permissions %s: %d healthcare-related (of %d rows)%s%s"
              % (cc.get("window") or "?", cc["healthcare_decisions_90d_count"],
                 cc.get("rows_scanned", 0),
                 (": " + count_str) if count_str else "",
                 ("; %d within ~10 km" % cc["near_clinic_count"])
                 if cc.get("near_clinic_count") else ""))
        for a in cc.get("healthcare_decisions_90d") or []:
            loc = " [%s ~%s km]" % (a["district"], a["approx_km"]) if a.get("district") else ""
            cat = (" (%s)" % a["category"]) if a.get("category") else ""
            print("         - %s: %s — %s [%s]%s%s"
                  % (a.get("date") or "?", a.get("address") or "?",
                     (a.get("what") or "?")[:70], a.get("decision_type") or "?", cat, loc))
        if not cc.get("healthcare_decisions_90d"):
            print("         - none in window")
```

- [ ] **Step 5: Gates** — prompt line + CLI on the seeded cache (0 credits):

```bash
venv/bin/python - <<'PY'
import json, os, tempfile
from context import prompts, brief
d = json.load(open(os.path.join(tempfile.gettempdir(), "cp_rag_context_cache", "ura_planning_v2.json")))
snap = {"meta": {"clinic": {"name": "seed", "lat": 1.345, "lon": 103.718}},
        "catchment_change": d, "data_gaps": []}
basis, local, clinic, as_of = prompts.format_live_context(snap)
assert "1 senior care" in local and "1 clinic" in local, local[-800:]
proj = brief._project(snap, {})
assert proj["planning"]["category_counts"] == {"Senior care": 1, "Clinic": 1}
print("prompt + brief OK")
PY
venv/bin/python -m context | sed -n '/PLANNING/,/^$/p'
```

Expected: prompt line contains `(1 senior care, 1 clinic)`; CLI PLANNING header shows the same counts and each row ends with its `(category)` + `[district ~km]`.

- [ ] **Step 6: Docs.**

1. `docs/signal-research.md` — append after §17:

```markdown
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
  RECORD HERE. Ruled out: OneMap geocoding (API hosts fail DNS on this
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
  RECORD HERE of listed rows; the headline count now excludes vet rows and
  stale-dated rows (RECORD HERE: n vet, m stale dropped in the live window).
- **Live verification**: RECORD HERE (tile/card/prompt/brief observations,
  0 paid credits; seeded-cache gates throughout).
```

Fill every `RECORD HERE` with the measured values before finishing the task.

2. `CLAUDE.md` — two edits:
   - Phase 6 bullet (line ~21): "6 KPI tiles" → "5 KPI tiles (Nearest polyclinic tile removed 2026-09-10 — static, no signal; data kept in prompt + raw context)".
   - Context-layer bullet (line ~22): extend the URA clause to: "`context/ura.py` (Phase 4/8: 90-day window re-filtered on decision_date, veterinary excluded, per-row `category` + `category_counts`) with street-name area enrichment in `snapshot.py` (`STREET_AREA_HINTS` → `district`/`approx_km`/`distance_band`/`near_clinic_count`, rough centroid distance)".
3. `HANDOFF.md` — full rewrite (repo convention): Done = Phase 8 summary (files + gates + live numbers); Outstanding = keep the Phase 7 WIP item (update file list) + "URA v2 cache first real fetch pending if still seeded"; Next-session prompt = current state + suggested user commit `feat: URA planning zoom — categories, street-area heuristic, polyclinic tile removed (Phase 8)`.

- [ ] **Step 7: Diff review** — `git --no-pager diff context/prompts.py context/brief.py context/__main__.py docs/signal-research.md CLAUDE.md HANDOFF.md` + `git status --short`. NO COMMIT.

---

## Self-Review (run by the plan author before execution)

1. **Spec coverage:** category zoom (Tasks 1, 3) · senior-care/nursing-home visibility (Task 1 counts, Task 3 tile/card) · boundary = street→district heuristic (Task 1 table, Task 2 enrichment) · decision-date honesty (Task 1) · vet noise (Task 1) · polyclinic tile removed, data kept (Task 3; polyclinic data stays in the RAG prompt and raw-context list) · docs (Task 4). No gaps.
2. **Placeholder scan:** the only `RECORD HERE`s are in Task 4 Step 6 §18 — intentional: live-only numbers (coverage, dropped counts, verification observations) that cannot be known at plan time; the step requires filling them before the task is done.
3. **Type consistency:** `category_counts` (dict), `stale_dropped` (int), row `category` (str), `district`/`approx_km`/`distance_band` (str/float/str, absent when unmapped), `near_clinic_count` (int) — identical names across Tasks 1–4 (payload → snapshot → UI/prompt/brief/CLI). Category string set {Senior care, Nursing home, Child care, Polyclinic, Clinic, Medical, Other} used verbatim in the JS `order` array, the Python `bits`/`count_str` tuples, and `_CATEGORY_RULES`.
4. **Known soft spot (Task 3 Step 5):** with the 2-row seed and the Woodlands test clinic, both rows land `mid` (Central ≈13.3 km, Tampines ≈24.9 km), so expected `near_clinic_count` = **0**; if a different value appears, re-derive from `haversine_km` before changing code.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-10-ura-planning-zoom.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

