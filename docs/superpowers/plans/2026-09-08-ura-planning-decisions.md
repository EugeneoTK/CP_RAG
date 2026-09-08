# Phase 4 — URA Planning-Decision Catchment Block Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the URA `Planning_Decision` service as the `catchment_change` live-context signal: healthcare-related written permissions (last 90 days) appear in the snapshot, the RAG prompt's *Local context* section, the CLI, and the UI strip — following the Phase 3 (WIDB) pattern exactly.

**Architecture:** New `context/ura.py` module (stdlib-only, like the rest of `context/`): AccessKey → daily token → `invokeUraDS/v1?service=Planning_Decision&last_dnload_date=dd/mm/yyyy` → keyword-filter `submission_desc` for healthcare/elderly/childcare uses → 24 h disk cache → `fetch_catchment()` returns `(payload, error)` like every other fetcher. `snapshot.py` adds the `catchment_change` block (failure → `data_gaps`, never a crash); `prompts.py` adds one *Local context* line (never protocol content); `__main__.py` prints a `PLANNING` section; `static/index.html` gets a chip + detail card.

**Tech Stack:** Python 3.9 stdlib (`urllib` via `context/http.py`), no new dependencies, no embedding/LLM calls — **0 API credits**. Server: existing FastAPI on port 5001.

**Spec:** `docs/signal-research.md` §9 (phase table row 4) + §10 (URA e-Services API research, access key verified live 2026-09-08). Pattern reference: Phase 3 WIDB (`context/widb.py`, build notes §14).

## Global Constraints

- **Python 3.9 only** (system `/usr/bin/python3`; venv at repo root). No 3.10+ syntax.
- **`context/` stays stdlib-only** — no new dependencies, nothing in `requirements.txt`.
- **0 API credits** for this phase: no embedding/LLM calls; never touch `chroma_db/`; prefer `curl /api/status` (free) over chat probes.
- **A failing source never crashes the snapshot** — errors land in `data_gaps` as a short `ura: ...` string.
- **`URA_ACCESS_KEY`**: read from `.env` (machine-parsed, one `KEY=value` per line) with `os.environ` fallback, exactly like `DGS_API_KEY` in `config.py`. Never in code or logs; `.env` is gitignored. `.env.example` already carries the `URA_ACCESS_KEY=` placeholder — no change needed.
- **Provenance rule**: URA data is *Local context* — a written permission is **not an opened facility** and is never presented as protocol content or an existing service. This caveat is encoded in the payload, the prompt line, the CLI header, and the UI card.
- **Verification convention (CLAUDE.md: "No test suite exists")**: no pytest. Per-task verification = ad-hoc monkeypatched runs (Phase 3 pattern) + live endpoints (`/api/status`, `/api/context`, `python -m context`).
- **Commits**: one commit per task (SDD pattern, cf. Phase 5 git history). The repo rule "one focused change per session, verify, stop" is satisfied per-task by the per-task review gates.
- **Deliberate scope deviations from the §10 sketch** (documented in Task 4):
  1. Sketch's `healthcare_approvals_90d` → **`healthcare_decisions_90d`**: the service returns granted AND rejected decisions (`decision_type` field; exact values only observable live — Task 4 records them). Both are kept, each with its `decision_type` shown.
  2. Sketch's `new_resi_units_approved_region` → **dropped**: `Planning_Decision` rows carry no unit counts, and rows have street addresses only (no postcode/region), so regional aggregation needs OneMap geocoding (its own free key) — deferred, recorded as follow-up.

---

### Task 1: `context/config.py` URA constants + `context/ura.py` (fetch, filter, cache)

**Files:**
- Modify: `context/config.py` (URA block after the WIDB block ending line 30; `_load_ura_key()` + `URA_ACCESS_KEY` after the `DGS_API_KEY = _load_key()` line 57)
- Create: `context/ura.py`
- Ad-hoc verify script (NOT committed): `/tmp/verify_ura_task1.py`

**Interfaces:**
- Consumes: `context.http.get_json(url, extra_headers=...)` (existing); `config.CACHE_DIR` (existing).
- Produces (Task 2 relies on these exact names):
  - `ura.fetch_catchment(use_cache=True) -> (payload_dict | None, error_str | None)` — never raises.
  - `ura.get_token() -> str` (raises on non-success token response).
  - `ura.fetch_rows(window_days=None) -> (rows_list, None) | (None, error_str)` (raises on transport failure).
  - `ura.filter_healthcare(rows) -> [ {address, what, date, decision_type, decision_no} ]`, newest `date` first.
  - `config.URA_ACCESS_KEY` (str | None; `ura.py` reads it **at call time** so ad-hoc verification can stub it).
  - Payload shape (the `catchment_change` block):
    ```json
    {"source": "...", "window": "YYYY-MM-DD to YYYY-MM-DD", "rows_scanned": 2500,
     "healthcare_decisions_90d_count": 3,
     "healthcare_decisions_90d": [{"address": "49B HOLLAND ROAD",
       "what": "PROPOSED FOR THE ERECTION OF A NEW 3-STOREY CHILD CARE CENTRE",
       "date": "2026-02-03", "decision_type": "Approved", "decision_no": "..."}],
     "caveat": "URA written permission = planning approval, not an opened facility; island-wide (no catchment geocoding yet)"}
    ```

- [ ] **Step 1: Add URA constants to `context/config.py`**

After `WIDB_CACHE_TTL_SECONDS = 3 * 24 * 3600  # weekly publication cadence` (line 30), insert:

```python
# --- URA e-Services (Phase 4 — planning decisions) ---------------------------------
# Auth: AccessKey header -> daily token (insertNewToken/v1) -> data calls carry
# AccessKey + Token headers (invokeUraDS/v1). Docs: eservice.ura.gov.sg/maps/api/.
URA_BASE = "https://eservice.ura.gov.sg/uraDataService"
URA_WINDOW_DAYS = 90             # last_dnload_date window (API max lookback: 1 year)
URA_CACHE_TTL_SECONDS = 24 * 3600  # data cadence is daily; token valid for the day
URA_MAX_ITEMS = 20               # snapshot cap on the decision list (count stays full)
URA_HEALTHCARE_KEYWORDS = ("POLYCLINIC", "CLINIC", "MEDICAL", "NURSING HOME",
                           "CHILD CARE", "SENIOR")
```

After `DGS_API_KEY = _load_key()` (line 57), insert:

```python
def _load_ura_key():
    try:
        env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env):
            for line in open(env):
                line = line.strip()
                if line.startswith("URA_ACCESS_KEY="):
                    return line.split("=", 1)[1].strip() or None
    except OSError:
        pass
    return os.environ.get("URA_ACCESS_KEY") or None


URA_ACCESS_KEY = _load_ura_key()
```

- [ ] **Step 2: Create `context/ura.py`** (complete file):

```python
"""URA planning decisions — the catchment_change block (Phase 4).

URA e-Services `Planning_Decision` service (docs/signal-research.md §10):
written permissions (granted/rejected) with street address, submission
description, decision date, decision type. Auth: `AccessKey` header ->
daily token from `insertNewToken/v1` (`{"Result": <token>}`) -> data call
`invokeUraDS/v1?service=Planning_Decision&last_dnload_date=dd/mm/yyyy`
with `AccessKey` + `Token` headers -> `{"Status": ..., "Result": [rows]}`.
Row fields (per the official docs): dr_id, submission_no, decision_no,
decision_date (dd/mm/yyyy), decision_type, submission_desc, address,
mkts_lotno, delete_ind ('Yes' only when queried by last_dnload_date: the
record was deleted).

Scope (v1, island-wide): rows carry street addresses with NO postcode/
region, and the docs' catchment option (OneMap geocoding) needs its own
key — deferred (build notes §15). So the block lists healthcare-related
decisions nationwide (keyword filter on submission_desc) and is labelled
"not an opened facility, island-wide" everywhere it is rendered. The
§10 sketch's `new_resi_units_approved_region` is NOT derivable (the
service has no units field) — dropped, documented in §15.

Needs URA_ACCESS_KEY in .env (config.URA_ACCESS_KEY, read at call time so
ad-hoc verification can stub it). No key -> (None, "URA_ACCESS_KEY not
set in .env (Phase 4 signal disabled)") -> data_gaps, never a crash.
Disk-cached 24 h (daily cadence) in config.CACHE_DIR.
"""

import datetime
import json
import os
import re
import time

from . import config
from .http import get_json


# --- parsing -----------------------------------------------------------------------


def _match_healthcare(desc):
    """True if submission_desc names a healthcare/elderly/childcare use."""
    d = (desc or "").upper()
    return any(k in d for k in config.URA_HEALTHCARE_KEYWORDS)


def _fmt_date(s):
    """dd/mm/yyyy -> YYYY-MM-DD (best effort; odd values pass through)."""
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", str(s or "").strip())
    if not m:
        return str(s or "").strip() or None
    return "%s-%s-%s" % (m.group(3), m.group(2), m.group(1))


def _parse_row(row):
    return {
        "address": (row.get("address") or "").strip(),
        "what": re.sub(r"\s+", " ", row.get("submission_desc") or "").strip(),
        "date": _fmt_date(row.get("decision_date")),
        "decision_type": (row.get("decision_type") or "").strip(),
        "decision_no": row.get("decision_no") or None,
    }


def filter_healthcare(rows):
    """Raw Planning_Decision rows -> parsed healthcare-related decisions,
    newest decision_date first. Deleted records (delete_ind == 'Yes') and
    non-healthcare rows are dropped."""
    out = []
    for r in rows or []:
        if (r.get("delete_ind") or "").strip() == "Yes":
            continue
        if _match_healthcare(r.get("submission_desc")):
            out.append(_parse_row(r))
    out.sort(key=lambda x: x["date"] or "", reverse=True)
    return out


# --- API -----------------------------------------------------------------------------


def get_token():
    """Daily URA token (raises on non-success / missing token)."""
    doc = get_json(config.URA_BASE + "/insertNewToken/v1",
                   extra_headers={"AccessKey": config.URA_ACCESS_KEY})
    if (doc.get("Status") or "").lower() != "success" or not doc.get("Result"):
        raise ValueError("token fetch rejected: " + json.dumps(doc)[:200])
    return doc["Result"]


def fetch_rows(window_days=None):
    """Planning_Decision rows created/modified/deleted since
    today - window_days -> (rows, error). Raises on transport failure."""
    window_days = window_days or config.URA_WINDOW_DAYS
    since = datetime.date.today() - datetime.timedelta(days=window_days)
    url = "%s/invokeUraDS/v1?service=Planning_Decision&last_dnload_date=%s" % (
        config.URA_BASE, since.strftime("%d/%m/%Y"))
    token = get_token()
    doc = get_json(url, extra_headers={"AccessKey": config.URA_ACCESS_KEY,
                                       "Token": token})
    if (doc.get("Status") or "").lower() != "success":
        return None, "URA error: " + json.dumps(doc)[:200]
    return doc.get("Result") or [], None


# --- snapshot block (cache + public entry point) --------------------------------------


def _cache_path():
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, "ura_planning.json")


def fetch_catchment(use_cache=True):
    """catchment_change payload -> (payload, error). Never raises."""
    if not config.URA_ACCESS_KEY:
        return None, "URA_ACCESS_KEY not set in .env (Phase 4 signal disabled)"
    path = _cache_path()
    if use_cache and os.path.exists(path):
        if time.time() - os.path.getmtime(path) < config.URA_CACHE_TTL_SECONDS:
            try:
                with open(path) as fh:
                    cached = json.load(fh)
                if cached.get("window"):
                    return cached, None
            except (OSError, ValueError):
                pass  # corrupt cache -> refetch

    try:
        rows, err = fetch_rows()
    except Exception as e:  # transport/HTTP failure -> short safe string
        return None, "fetch failed: %s" % e
    if err:
        return None, err

    decisions = filter_healthcare(rows)
    today = datetime.date.today()
    since = today - datetime.timedelta(days=config.URA_WINDOW_DAYS)
    payload = {
        "source": "URA Planning_Decision (written permissions), %d-day window"
                  % config.URA_WINDOW_DAYS,
        "window": "%s to %s" % (since.isoformat(), today.isoformat()),
        "rows_scanned": len(rows),
        "healthcare_decisions_90d_count": len(decisions),
        "healthcare_decisions_90d": decisions[:config.URA_MAX_ITEMS],
        "caveat": ("URA written permission = planning approval, not an opened "
                   "facility; island-wide (no catchment geocoding yet)"),
    }
    try:
        with open(path, "w") as fh:
            json.dump(payload, fh)
    except OSError:
        pass
    return payload, None
```

- [ ] **Step 3: Write the ad-hoc verification script `/tmp/verify_ura_task1.py`** (NOT committed):

```python
"""Task 1 gate: ura.py fetch/filter/cache/failure paths, 0 network, 0 credits."""
import json
import os
import sys

sys.path.insert(0, ".")
import context.config as config
import context.ura as ura

FAKE_TOKEN = {"Result": "test-token", "Status": "Success", "Message": ""}
# 4 rows: 2 healthcare (child care, clinic), 1 deleted non-healthcare,
# 1 healthcare (senior) — mirrors the live sample in docs §10.
FAKE_ROWS = [
    {"dr_id": "1", "submission_no": "S1", "decision_no": "D1",
     "decision_date": "03/02/2026", "decision_type": "Approved",
     "submission_desc": "PROPOSED FOR THE ERECTION OF A NEW 3-STOREY CHILD CARE CENTRE",
     "address": "49B HOLLAND ROAD", "mkts_lotno": "4231A", "delete_ind": "No"},
    {"dr_id": "2", "submission_no": "S2", "decision_no": "D2",
     "decision_date": "20/07/2026", "decision_type": "Rejected",
     "submission_desc": "CHANGE OF USE OF MARGINAL PORTION TO CLINIC",
     "address": "10 TAMPINES STREET 92", "mkts_lotno": "18845B", "delete_ind": "No"},
    {"dr_id": "3", "submission_no": "S3", "decision_no": "D3",
     "decision_date": "01/09/2026", "decision_type": "Approved",
     "submission_desc": "HDB STRATA SUBDIVISION OF FLAT",
     "address": "200 WOOLTHOROUGH ROAD", "mkts_lotno": "11331", "delete_ind": "Yes"},
    {"dr_id": "4", "submission_no": "S4", "decision_no": "D4",
     "decision_date": "30/08/2026", "decision_type": "Approved",
     "submission_desc": "PROPOSED FOR THE PROVISION OF A SENIOR ACTIVITY CENTRE",
     "address": "58 ANG MO KIO AVENUE 10", "mkts_lotno": "9012C", "delete_ind": "No"},
]

calls = []


def fake_get_json(url, timeout=45, retries=4, backoff=5, extra_headers=None):
    calls.append((url, extra_headers))
    assert (extra_headers or {}).get("AccessKey") == "test-key", extra_headers
    if url.endswith("insertNewToken/v1"):
        return FAKE_TOKEN
    assert "service=Planning_Decision" in url and "last_dnload_date=" in url, url
    assert (extra_headers or {}).get("Token") == "test-token", extra_headers
    return {"Status": "Success", "Message": "", "Result": FAKE_ROWS}


config.URA_ACCESS_KEY = "test-key"
ura.get_json = fake_get_json
p = ura._cache_path()
if os.path.exists(p):
    os.remove(p)

# 1. happy path: filter + payload + cache written
payload, err = ura.fetch_catchment(use_cache=True)
assert err is None, err
assert payload["rows_scanned"] == 4, payload
assert payload["healthcare_decisions_90d_count"] == 3, payload
addrs = [d["address"] for d in payload["healthcare_decisions_90d"]]
assert addrs == ["58 ANG MO KIO AVENUE 10", "10 TAMPINES STREET 92",
                 "49B HOLLAND ROAD"], addrs  # newest decision_date first
assert payload["healthcare_decisions_90d"][0]["date"] == "2026-08-30"
assert payload["healthcare_decisions_90d"][1]["decision_type"] == "Rejected"
assert "not an opened facility" in payload["caveat"]
assert " to " in payload["window"], payload["window"]
assert os.path.exists(p), "cache file not written"

# 2. cache hit: no second network round-trip
n = len(calls)
payload2, err2 = ura.fetch_catchment(use_cache=True)
assert err2 is None and payload2 == payload
assert len(calls) == n, "cache miss — re-fetched from the API"

# 3. no key -> clean error, no fetch attempted
config.URA_ACCESS_KEY = None
payload3, err3 = ura.fetch_catchment(use_cache=True)
assert payload3 is None and "URA_ACCESS_KEY not set" in err3, (payload3, err3)

# 4. token transport failure -> error string, no exception
config.URA_ACCESS_KEY = "test-key"
if os.path.exists(p):
    os.remove(p)  # force a refetch past the cache


def boom(url, **kw):
    raise Exception("HTTP Error 401: Unauthorized")


ura.get_json = boom
payload4, err4 = ura.fetch_catchment(use_cache=True)
assert payload4 is None and "fetch failed" in err4, (payload4, err4)

# 5. bad Status on the data call -> error string
def bad_status(url, **kw):
    if url.endswith("insertNewToken/v1"):
        return FAKE_TOKEN
    return {"Status": "Error", "Message": "invalid token", "Result": None}


ura.get_json = bad_status
payload5, err5 = ura.fetch_catchment(use_cache=True)
assert payload5 is None and "URA error" in err5, (payload5, err5)

print("TASK1 OK — filter/sort/delete-skip, payload+window, cache hit, "
      "no-key gap, token-failure and bad-status paths (0 network, 0 credits)")
```

- [ ] **Step 4: Run the verification from the repo root**

Run: `cd "/Users/ugeneo/Documents/Project Codes/CP_RAG" && venv/bin/python /tmp/verify_ura_task1.py`
Expected: prints `TASK1 OK — ...`, no assertion errors. Also confirm a clean Python 3.9 import: `venv/bin/python -c "import context.ura; print('import ok')"`

- [ ] **Step 5: Commit**

```bash
git add context/config.py context/ura.py
git commit -m "feat: Phase 4 — URA planning-decision fetch (context/ura.py): token, 90-day Planning_Decision window, healthcare keyword filter, 24h disk cache"
```

---

### Task 2: Wire `catchment_change` into the snapshot, the RAG prompt, and the CLI

**Files:**
- Modify: `context/snapshot.py` (import line 11; insert URA block after the WIDB block, lines 177–183)
- Modify: `context/prompts.py` (insert URA line after the WIDB line block, lines 45–53)
- Modify: `context/__main__.py` (`--no-cache` help text line 28; insert PLANNING section after the DISEASE block, lines 111–116)
- Ad-hoc verify script (NOT committed): `/tmp/verify_ura_task2.py`

**Interfaces:**
- Consumes: `ura.fetch_catchment(use_cache=True) -> (payload, error)` (Task 1).
- Produces: `snap["catchment_change"]` (present only on success; failure → `"ura: ..."` in `data_gaps`); one extra *Local context* prompt line when `healthcare_decisions_90d_count` is truthy; `PLANNING` CLI section when the block is present.

- [ ] **Step 1: `context/snapshot.py`**

Line 11, change:
```python
from . import config, fetchers, widb
```
to:
```python
from . import config, fetchers, ura, widb
```

After the WIDB block (after `snap["disease_week"] = widb_val`, line 183), insert:
```python
    # --- URA planning decisions (catchment_change, Phase 4; needs URA_ACCESS_KEY) -------
    time.sleep(2)
    cc, cc_err = ura.fetch_catchment(use_cache)
    if cc_err:
        gaps.append("ura: %s" % cc_err)
    else:
        snap["catchment_change"] = cc
```

- [ ] **Step 2: `context/prompts.py`**

After the WIDB block (after `local_lines = signal_lines + [line]`, the last line of the `if dw.get("epi_week"):` block, line 53), insert:
```python
    # URA (Phase 4): island-wide planning-decision signal — baseline context,
    # same treatment as WIDB; a written permission is explicitly NOT an
    # opened facility, so the line says so.
    cc = snapshot.get("catchment_change") or {}
    if cc.get("healthcare_decisions_90d_count"):
        examples = "; ".join(
            "%s %s — %s%s" % (
                a.get("date") or "?", a.get("address") or "?",
                (a.get("what") or "?")[:80],
                (" [" + a["decision_type"] + "]") if a.get("decision_type") else "")
            for a in (cc.get("healthcare_decisions_90d") or [])[:3])
        local_lines = local_lines + [
            "- URA planning decisions %s (island-wide written permissions, "
            "NOT protocol content): %d healthcare-related, e.g. %s. A written "
            "permission is NOT an opened facility — never present one as an "
            "existing service." % (
                cc.get("window") or "last 90 days",
                cc["healthcare_decisions_90d_count"], examples)]
```

- [ ] **Step 3: `context/__main__.py`**

Line 28, change the help text to:
```python
                     help="refetch GEOJSON layers, the WIDB bulletin and the URA planning window")
```

After the DISEASE block (after the `for n in dw.get("notable") or []:` loop, line 116), insert:
```python
    cc = snap.get("catchment_change") or {}
    if cc.get("healthcare_decisions_90d_count") is not None:
        print("PLANNING URA written permissions %s: %d healthcare-related (of %d rows)"
              % (cc.get("window") or "?", cc["healthcare_decisions_90d_count"],
                 cc.get("rows_scanned", 0)))
        for a in cc.get("healthcare_decisions_90d") or []:
            print("         - %s: %s — %s [%s]"
                  % (a.get("date") or "?", a.get("address") or "?",
                     (a.get("what") or "?")[:70], a.get("decision_type") or "?"))
        if not cc.get("healthcare_decisions_90d"):
            print("         - none in window")
```

- [ ] **Step 4: Write the ad-hoc verification script `/tmp/verify_ura_task2.py`** (NOT committed):

```python
"""Task 2 gate: prompt rendering + snapshot wiring, 0 URA network (24h cache)."""
import os
import sys

sys.path.insert(0, ".")
import context.config as config
import context.ura as ura

# Re-seed the disk cache deterministically (fake transport, no network).
FAKE_TOKEN = {"Result": "test-token", "Status": "Success", "Message": ""}
FAKE_ROWS = [
    {"dr_id": "1", "submission_no": "S1", "decision_no": "D1",
     "decision_date": "03/02/2026", "decision_type": "Approved",
     "submission_desc": "PROPOSED FOR THE ERECTION OF A NEW 3-STOREY CHILD CARE CENTRE",
     "address": "49B HOLLAND ROAD", "mkts_lotno": "4231A", "delete_ind": "No"},
    {"dr_id": "4", "submission_no": "S4", "decision_no": "D4",
     "decision_date": "30/08/2026", "decision_type": "Approved",
     "submission_desc": "PROPOSED FOR THE PROVISION OF A SENIOR ACTIVITY CENTRE",
     "address": "58 ANG MO KIO AVENUE 10", "mkts_lotno": "9012C", "delete_ind": "No"},
]


def fake_get_json(url, timeout=45, retries=4, backoff=5, extra_headers=None):
    if url.endswith("insertNewToken/v1"):
        return FAKE_TOKEN
    return {"Status": "Success", "Message": "", "Result": FAKE_ROWS}


config.URA_ACCESS_KEY = "test-key"
ura.get_json = fake_get_json
p = ura._cache_path()
if os.path.exists(p):
    os.remove(p)
payload, err = ura.fetch_catchment(use_cache=True)
assert err is None, err

# 1. prompt: the URA line renders into Local context, with the caveat
snap = {"meta": {"generated_at": "now", "clinic": {"name": "T"}},
        "protocol_links": {"active": []},
        "disease_week": {}, "catchment_change": payload}
from context.prompts import format_live_context
basis, local, clinic, as_of = format_live_context(snap)
ura_lines = [l for l in local.splitlines() if "URA planning decisions" in l]
assert len(ura_lines) == 1, ura_lines
assert "NOT an opened facility" in ura_lines[0]
assert "58 ANG MO KIO AVENUE 10" in ura_lines[0]  # newest example first
assert "[Approved]" in ura_lines[0]

# 2. prompt: no block -> no URA line, nothing else changes
basis2, local2, _, _ = format_live_context(
    {"meta": {"generated_at": "now", "clinic": {"name": "T"}},
     "protocol_links": {"active": []}})
assert "URA planning decisions" not in local2

print("TASK2 OK — prompt line renders with caveat + examples; "
      "absent block adds nothing (0 URA network)")
```

- [ ] **Step 5: Run verification + the CLI gates**

Run (from repo root):
```bash
venv/bin/python /tmp/verify_ura_task2.py
# no-key gate: gap line present, PLANNING section absent, exit 0
venv/bin/python -m context 2>&1 | grep "ura: URA_ACCESS_KEY not set"
test "$(venv/bin/python -m context | grep -c PLANNING)" = 0 && echo "no PLANNING section (expected)"
# populated CLI path: seeded 24h cache + env key -> PLANNING section, 0 URA network
URA_ACCESS_KEY=test venv/bin/python -m context | sed -n '/PLANNING/,+4p'
```
Expected: script prints `TASK2 OK ...`; grep matches the `ura: URA_ACCESS_KEY not set in .env (Phase 4 signal disabled)` data gap; the last command prints `PLANNING URA written permissions ...: 2 healthcare-related (of 2 rows)` followed by the two rows (Ang Mo Kio first). Note: each `python -m context` run builds a FULL snapshot, so the other key-free sources hit the network once (~15–40 s, 0 API credits) — the normal CLI path.

- [ ] **Step 6: Commit**

```bash
git add context/snapshot.py context/prompts.py context/__main__.py
git commit -m "feat: Phase 4 — catchment_change in snapshot, Local-context prompt line and PLANNING CLI section"
```

---

### Task 3: UI — URA chip + "Planning decisions" card in the context strip

**Files:**
- Modify: `static/index.html` (chip list after the WIDB chip, lines 646–649; detail cards after the `widbHtml` block, lines 693–718; `ctxCards` assembly, lines 757–763)

**Interfaces:**
- Consumes: `snap.catchment_change` (Task 2 payload shape: `window`, `rows_scanned`, `healthcare_decisions_90d_count`, `healthcare_decisions_90d[]` with `date/address/what/decision_type`, `caveat`).
- Produces: a `URA 90d` chip (only when the block is present) and a detail card; no server restart needed (static file, served by the running app).

- [ ] **Step 1: Add the chip**

After the WIDB chip block (after line 649, `dwD.median_5yr != null && dwD.week > dwD.median_5yr ? 'warn' : 'info')));`), insert:
```js
    const cc = snap.catchment_change || {};
    if (cc.healthcare_decisions_90d_count != null)
      chips.push(chipHtml('URA 90d', cc.healthcare_decisions_90d_count + ' healthcare-related', 'info'));
```

- [ ] **Step 2: Add the detail card**

After the `widbHtml` block (after its closing `}` at line 718, before `let miscHtml = '<h3>Data quality</h3>';`), insert:
```js
    let uraHtml = '<h3>Planning decisions (URA, island-wide)</h3>';
    if (cc.healthcare_decisions_90d_count != null) {
      uraHtml += `<div class="row dim">${escapeHtml(cc.window || '')} — ${cc.rows_scanned != null ? cc.rows_scanned : '?'} written permissions scanned</div>`;
      const hcd = cc.healthcare_decisions_90d || [];
      uraHtml += hcd.length
        ? hcd.map(a => `<div class="row">• ${escapeHtml(a.date || '?')}: ${escapeHtml(a.address || '?')} — ${escapeHtml((a.what || '?').slice(0, 90))}${a.decision_type ? ' <span class="dim">[' + escapeHtml(a.decision_type) + ']</span>' : ''}</div>`).join('')
        : '<div class="row dim">• no healthcare-related decisions in window</div>';
      uraHtml += `<div class="row dim">${escapeHtml(cc.caveat || 'planning permission ≠ opened facility')}</div>`;
    } else uraHtml += '<div class="row dim">unavailable (see data gaps)</div>';
```

- [ ] **Step 3: Mount the card in `ctxCards`**

Lines 757–763, change:
```js
    ctxCards.innerHTML =
      `<div class="ctx-card ctx-wide">${protosHtml}</div>`
      + `<div class="ctx-card">${airHtml}</div>`
      + `<div class="ctx-card">${weatherHtml}</div>`
      + `<div class="ctx-card">${dengueHtml}</div>`
      + `<div class="ctx-card">${widbHtml}</div>`
      + `<div class="ctx-card">${miscHtml}</div>`;
```
to:
```js
    ctxCards.innerHTML =
      `<div class="ctx-card ctx-wide">${protosHtml}</div>`
      + `<div class="ctx-card">${airHtml}</div>`
      + `<div class="ctx-card">${weatherHtml}</div>`
      + `<div class="ctx-card">${dengueHtml}</div>`
      + `<div class="ctx-card">${widbHtml}</div>`
      + `<div class="ctx-card">${uraHtml}</div>`
      + `<div class="ctx-card">${miscHtml}</div>`;
```

- [ ] **Step 4: Verify**

The static file is served without a restart, but `/api/context` runs the `context` package imported at server start — **restart the 5001 server** so it loads the new `snapshot.py`/`ura.py` (the Phase 5 ledger did the same after code changes):

```bash
# restart the app server on 5001
lsof -ti :5001 | xargs kill 2>/dev/null; sleep 1
cd "/Users/ugeneo/Documents/Project Codes/CP_RAG"
nohup venv/bin/python app.py > /tmp/cp_rag_server.log 2>&1 &
sleep 4 && curl -s localhost:5001/api/status    # ready:true (context may still be "building"/"not-built")
# page script still parses
sed -n '/<script>/,/<\/script>/p' static/index.html | sed '1d;$d' > /tmp/ctx_page.js && node --check /tmp/ctx_page.js && echo "JS OK"
# page + context endpoint still healthy (first call after restart builds a fresh snapshot, ~15–40 s)
curl -s -o /dev/null -w "/ -> %{http_code}\n" localhost:5001/
curl -s --max-time 90 "localhost:5001/api/context" | venv/bin/python -c "import json,sys; s=json.load(sys.stdin); print('catchment_change' in s, [g for g in s['data_gaps'] if g.startswith('ura:')])"
```
Expected: `JS OK`; `/ -> 200`; the last line prints either `True []` (key set / seeded cache) or `False ['ura: URA_ACCESS_KEY not set in .env (Phase 4 signal disabled)']` (no key — the card then shows "unavailable (see data gaps)", matching the WIDB pattern).

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat: Phase 4 — URA planning-decision chip + detail card in the context strip"
```

---

### Task 4: Docs, HANDOFF, and the SDD ledger

**Files:**
- Modify: `docs/signal-research.md` (§9 row 4; §10 "v1 scope" note; new §15)
- Modify: `CLAUDE.md` (context-layer bullet; repo-layout `context/` line)
- Rewrite: `HANDOFF.md` (Phase 4 session-state format, like the Phase 3/5 handoffs)
- Create (local only, untracked like the other ledgers): `.superpowers/sdd/2026-09-08-ura-planning/ledger.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: `docs/signal-research.md` §9 — mark Phase 4 DONE**

Replace the Phase 4 table row (currently `| 4 | F6 planning-decision catchment block (optional) |`) with:
```markdown
| 4 | F6 URA planning-decision catchment block — **DONE 2026-09-08** (`context/ura.py`: `URA_ACCESS_KEY` → daily token → `Planning_Decision&last_dnload_date=<today−90d>`; healthcare keyword filter on `submission_desc`; 24 h disk cache; `catchment_change` snapshot block = island-wide healthcare-related written permissions, latest 20, each with `decision_type` — NOT opened facilities; `Local context` prompt line; CLI `PLANNING` section; UI chip + card; no key → `ura:` data gap). Sketch deviation: `new_resi_units_approved_region` dropped (no units field in the service; regional aggregation needs OneMap geocoding — deferred, §15). Build notes in §15. |
```

- [ ] **Step 2: `docs/signal-research.md` §10 — record the v1 scope ruling**

Append to the end of §10 (after the Caveats list):
```markdown
**v1 scope ruling (2026-09-08, Phase 4 build).** The §10 sketch's
`new_resi_units_approved_region` is not derivable from this service: rows
carry no unit counts, and `address` is a street address with no
postcode/region, so regional aggregation requires geocoding (OneMap's free
API needs its own key) or a street→region table — both deferred. v1 ships
`healthcare_decisions_90d` (island-wide; granted AND rejected, each with
`decision_type` — the §10 sketch's `healthcare_approvals_90d` refined, since
the service returns both and the type is only observable live). Catchment
mapping (caveat 2) is the named follow-up if per-clinic distance filtering
is wanted.
```

- [ ] **Step 3: `docs/signal-research.md` — add §15 build notes** (after §14, end of file):

```markdown
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
(`$TMPDIR/cp_rag_context_cache/ura_planning.json`, daily cadence; the token
is daily, so one refetch/day is enough) and the `(payload, error)` contract.
No `URA_ACCESS_KEY` in `.env` → `ura: URA_ACCESS_KEY not set in .env
(Phase 4 signal disabled)` in `data_gaps` — the signal degrades exactly
like every other source, and the prompt/UI say nothing about planning.

Payload (`catchment_change`): `source`, `window`, `rows_scanned`,
`healthcare_decisions_90d_count` (full count), `healthcare_decisions_90d`
(latest 20: `address/what/date/decision_type/decision_no`), `caveat`.
Provenance: *Local context* only — the prompt line, CLI header, and UI
card all state a written permission is NOT an opened facility.

Verified (monkeypatched transport, 0 URA network): filter/sort/delete-skip,
cache-hit no-refetch, no-key gap, token-HTTP-failure and bad-Status error
paths; prompt line renders with the caveat; no-key CLI shows the gap and
no `PLANNING` section; seeded-cache CLI shows the section. Live
verification pending `URA_ACCESS_KEY` in `.env` (Appendix A steps).

Follow-ups (not built): catchment geocoding of `address` (OneMap free key
or street→region table) for per-clinic distance; residential-unit counts
(URA's separate `Private_Residential_Properties` services, if the signal
warrants them); rejected-vs-approved split chip in the UI.
```

- [ ] **Step 4: `CLAUDE.md`**

Context-layer bullet (the `- context/ ...` line in "What each layer does"): append after `widb.py` in the module list:
```markdown
; `ura.py` (Phase 4, needs `URA_ACCESS_KEY` in .env) fetches the URA daily token + `Planning_Decision` rows (90-day window), filters healthcare-related written permissions into the `catchment_change` snapshot block (island-wide; a permission is not an opened facility), 24 h disk cache
```
Repo layout: change the `context/` line from `(Phase 0–3, ...)` to `(Phase 0–4, ...)`.

- [ ] **Step 5: Rewrite `HANDOFF.md`** (same structure as the current Phase 3 handoff):

- **What this session did (Phase 4 — URA planning-decision catchment signal):** the bullets for `context/ura.py` (token, 90-day window, keyword filter, 24 h cache, no-key degradation), snapshot/prompt/CLI wiring, UI chip + card, docs updates; commit list (the four commits from Tasks 1–4).
- **Current state:** `git log --oneline -6`; "Phase 4 complete (island-wide v1)"; server on 5001; "Phases 0–4 done, 0 API credits"; "The standing checks passed after every task: ... `node --check` on the page script ...; no-key and seeded-cache `python -m context` gates ...; `curl /api/status` ...".
- **Known limitations (unchanged from the Phase 3 list, plus):** "URA v1 is island-wide (rows have street addresses only; catchment geocoding deferred) and lists permissions, not opened facilities — the prompt/UI say so" and "the URA live path is unverified until `URA_ACCESS_KEY` lands in `.env` (plan Appendix A); everything else is verified".
- **Outstanding:** keep the Phase 3 list; add "add `URA_ACCESS_KEY=<key>` to `.env` (machine-parseable line) and run plan Appendix A for the live URA verification"; "catchment geocoding for per-clinic planning signals (OneMap free key)".
- **SDD ledger note:** "SDD ledger for this phase: `.superpowers/sdd/2026-09-08-ura-planning/` (untracked, local only); plan at `docs/superpowers/plans/2026-09-08-ura-planning-decisions.md`".

- [ ] **Step 6: Create the ledger** `.superpowers/sdd/2026-09-08-ura-planning/ledger.md` (local only — do NOT `git add` it):

```markdown
# SDD Ledger — 2026-09-08-ura-planning-decisions

Plan: docs/superpowers/plans/2026-09-08-ura-planning-decisions.md
Task 1: complete (commit <sha>) — context/ura.py + config URA constants; Task1 gate green (filter/sort/delete-skip, cache hit, no-key gap, token-failure, bad-Status).
Task 2: complete (commit <sha>) — catchment_change in snapshot.py, Local-context line in prompts.py, PLANNING section in __main__.py; Task2 gate green (prompt line + caveat, absent-block no-op, no-key CLI gap, seeded-cache PLANNING section).
Task 3: complete (commit <sha>) — URA 90d chip + Planning decisions card; node --check OK, / 200, /api/context healthy.
Task 4: complete (commit <sha>) — signal-research §9/§10/§15, CLAUDE.md, HANDOFF.md rewrite.
Ruling: <any plan deviations, e.g. live decision_type values observed in Appendix A, if run>
```

- [ ] **Step 7: Final standing checks + commit**

```bash
curl -s localhost:5001/api/status            # {"ready":true,"context":"cached (...)" or "building"}
venv/bin/python -m context >/dev/null && echo "CLI exit 0"
grep -rn "URA_ACCESS_KEY=" .env.example      # placeholder still present, untouched
git status --short                           # only the four expected commits ahead of bcfcc44; .superpowers untracked
```

```bash
git add docs/signal-research.md CLAUDE.md HANDOFF.md
git commit -m "docs: Phase 4 build notes (signal-research §9/§10/§15, CLAUDE.md, HANDOFF rewrite)"
```

---

## Appendix A: Live URA verification (requires `URA_ACCESS_KEY` in `.env`)

The key is stored outside the repo (docs §10); the placeholder is already in `.env.example`. Once the user adds `URA_ACCESS_KEY=<key>` as a machine-parseable line in `.env`:

1. Fresh fetch (bypasses cache):
   `venv/bin/python -c "from context.ura import fetch_catchment; import json; p, e = fetch_catchment(use_cache=False); print(e or json.dumps(p, indent=2)[:1500])"`
   Expected: no error; `rows_scanned` in the low thousands; inspect the **actual `decision_type` values** (plan assumed `Approved`/`Rejected` strings — if they differ, record them in the ledger and §15; the filter does not branch on them, so no code change is expected).
2. `venv/bin/python -m context` → `PLANNING` section with real rows; `curl -s localhost:5001/api/context` → `catchment_change` present, no `ura:` gap.
3. UI: context strip shows the `URA 90d` chip and the Planning card with the caveat line.
4. Negative: temporarily unset the key (comment the `.env` line) → `ura:` data gap returns; chip/card degrade to "unavailable (see data gaps)".
5. Record live numbers (rows_scanned, count, decision_type values) in the ledger and §15.







