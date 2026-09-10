"""NTUC Health Active Ageing programme calendars — the active_ageing block (Phase 9).

Source: the calendar landing page (config.NTUC_CALENDAR_URL) lists one
monthly programme-calendar PDF per Active Ageing Centre
(`assets.ntuchealth.sg/ae/<centre>-<Month>-<YYYY>.pdf`). The same PDFs are
what rag.py's `ingest_community_refresh()` ingests into the "Community
resources" corpus section. Browser UA required (ntuchealth.sg is isomer /
Akamai-fronted; non-browser UAs get 403 — same class of wall as
docs/signal-research.md §14/§18).

Centre coordinates are NOT fetched: config.ACTIVE_AGING_CENTRES holds the
27 (name, lat, lon) positions verified live 2026-09-10 from
ntuchealth.sg/active-ageing/locations (Next.js flight payload). A live
centre missing from that table is still listed, just without a distance.

Known NTUC site quirk (detected dynamically, not hardcoded): some centre's
anchor reuses another centre's PDF (2026-09-10: 'Bukit Batok West' links
the Bedok-North PDF).

Disk-cached 24 h (the centre set changes rarely; the month rotates) in
config.CACHE_DIR. Key-free.
"""

import json
import os
import re
import time

from . import config
from .http import get_bytes

_PDF_ANCHOR = re.compile(
    r'<a [^>]*href="([^"]*assets\.ntuchealth\.sg/ae/[^"]+\.pdf)"[^>]*>(.*?)</a>',
    re.S)
_PDF_MONTH = re.compile(r"-([A-Z][a-z]{2})-(\d{4})\.pdf$")
_TAG = re.compile(r"<[^>]+>")


def _clean_name(raw):
    return re.sub(r"\s+", " ", _TAG.sub(" ", raw)).strip()


def _shared_pdf_note(centres):
    """Names that point at the same calendar PDF (NTUC site quirk)."""
    by_url = {}
    for c in centres:
        by_url.setdefault(c["pdf_url"], []).append(c["name"])
    bits = []
    for names in sorted(by_url.values()):
        if len(names) > 1:
            bits.append(" and ".join("'%s'" % n for n in names) + " share one calendar PDF")
    return "; ".join(bits)


def _parse(html):
    """Landing-page HTML -> ([{name, month, pdf_url}], months, shared-pdf note)."""
    seen, centres = set(), []
    for url, raw in _PDF_ANCHOR.findall(html):
        name = _clean_name(raw)
        if not name or name in seen:
            continue
        seen.add(name)
        m = _PDF_MONTH.search(url)
        month = ("%s %s" % (m.group(1), m.group(2))) if m else None
        centres.append({"name": name, "month": month, "pdf_url": url})
    months = sorted({c["month"] for c in centres if c["month"]})
    return centres, months, _shared_pdf_note(centres)


# --- snapshot block (cache + public entry point) --------------------------------------


def _cache_path():
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, "ntuc_ageing_v1.json")


def fetch_centres(use_cache=True):
    """active_ageing payload -> (payload, error). Never raises."""
    path = _cache_path()
    if use_cache and os.path.exists(path):
        if time.time() - os.path.getmtime(path) < config.NTUC_CACHE_TTL_SECONDS:
            try:
                with open(path) as fh:
                    cached = json.load(fh)
                if cached.get("centres"):
                    return cached, None
            except (OSError, ValueError):
                pass  # corrupt cache -> refetch

    try:
        html = get_bytes(config.NTUC_CALENDAR_URL, extra_headers=config.NTUC_UA).decode("utf-8", "replace")
    except Exception as e:  # transport/HTTP failure -> short safe string
        return None, "fetch failed: %s" % e
    if not html or "ntuchealth" not in html:
        return None, "unexpected response (blocked or site layout changed?)"

    centres, months, note = _parse(html)
    if not centres:
        return None, "no calendar PDF links found (site layout changed?)"

    caveat = ("Community (NON-clinical) exercise/social/digital-skills programmes "
              "for older adults; distances from the clinic point to NTUC-published "
              "centre coordinates; a calendar is a programme list, not a service.")
    if note:
        caveat += " Known NTUC site quirk: %s." % note
    payload = {
        "source": "NTUC Health Active Ageing Centre programme calendars (ntuchealth.sg)",
        "count": len(centres),
        "calendar_months": months,
        "centres": centres,
        "caveat": caveat,
    }
    try:
        with open(path, "w") as fh:
            json.dump(payload, fh)
    except OSError:
        pass
    return payload, None
