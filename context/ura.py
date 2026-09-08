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
