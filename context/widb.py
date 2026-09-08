"""WIDB — CDA Weekly Infectious Diseases Bulletin (Phase 3).

The bulletin is PDF-only (no CSV/API feed — docs/signal-research.md §1 S7, §5):
a per-year archive page on cda.gov.sg lists one PDF per epi-week. This module
finds the newest bulletin, downloads it (walking back a few weeks if the
newest file is broken — e.g. the EW34 2026 file returns S3 AccessDenied),
and parses:

  page 1  master disease table — per-disease rows:
              <name>  <week>  <prev_week>  <median same week 2021-25>
                      [<cum current year>  <cum prior year, same period>]
          (5-number rows; the ARI/other-disease group has 3 numbers — no
           cumulative columns). Column mapping verified arithmetically:
           EW32 cum + EW33 week == EW33 cum for dengue.
  page 2  ARI / influenza narrative (type distribution, positivity rates,
          top pathogens, polyclinic attendances)
  page 4  dengue narrative (notifications, hospital admissions, serotypes)

Extraction uses pypdf (already a top-level dependency for the Phase 5
guideline corpus) — no new dependency for the context package. The parsed
bulletin is cached on disk for 3 days (weekly publication cadence).

A failing source never crashes the snapshot: fetch_latest() returns
(payload, error) like the other fetchers.
"""

import datetime
import io
import json
import os
import re
import time
import urllib.parse

from . import config
from .http import get_bytes

# --- table row patterns (page 1) --------------------------------------------------
# Name + 3-5 numbers. The name class deliberately excludes digits: disease
# names in this table are letters/punctuation only, and allowing digits lets
# the engine backtrack into name="Dengue Fever 103" + 4 numbers on some rows.
# Separator after the name is 1+ spaces (3-column rows use a single space).
_ROW = re.compile(
    r"^(?P<name>[A-Z][A-Za-z ,/()#^*.\-]*?)\s{1,}"
    r"(?P<nums>(?:\d{1,6}|NA)(?:\s{2,}(?:\d{1,6}|NA)){2,4})\s*$")
# Name-only line (e.g. "Mpox#") whose numbers land on the following line.
_NAME_ONLY = re.compile(r"^[A-Z][A-Za-z0-9 ,/()#^*.\-]{2,60}$")
_NUMS_ONLY = re.compile(r"^(?P<nums>(?:\d{1,6}|NA)(?:\s{1,3}(?:\d{1,6}|NA)){2,4})$")

# --- header patterns ----------------------------------------------------------------
_VOL = re.compile(r"VOL\.\s*(\d+)\s+NO\.(\d+)\s+(\d{4})")
_WEEK = re.compile(
    r"EPIDEMIOLOGICAL WEEK\s+(\d+)\s+([0-9]{1,2}\s*-\s*[0-9]{1,2}\s+[A-Z][a-z]{2}\s+\d{4})")

# --- narrative patterns (run on whitespace-normalized page text) --------------------
_RE_DENGUE_NOTIF = re.compile(
    r"The number of dengue notifications was (\d+) in E-week (\d+)\.")
_RE_DENGUE_ADMISSIONS = re.compile(
    r"The number of hospital admissions \w+ to (\d+) in E-week (\d+), "
    r"compared to (\d+) in E-week (\d+)\.")
_RE_DENGUE_SEROTYPES = re.compile(
    r"serotyped in ([A-Z][a-z]+) (\d{4}) showed DEN-1, DEN-2, DEN-3 and DEN-4 at\s+"
    r"([\d.]+)%,\s*([\d.]+)%,\s*([\d.]+)% and ([\d.]+)% respectively")
_RE_FLU_TYPES = re.compile(
    r"Of the (\d+) specimens tested positive for influenza in ([A-Z][a-z]+) (\d{4}), "
    r"(\d+) were positive for influenza A\(pH1N1\) \(([\d.]+)%\), (\d+) were positive "
    r"for influenza A\(H3N2\) \(([\d.]+)%\), and (\d+) were positive for "
    r"influenza B \(([\d.]+)%\)")
_RE_ILI_POS = re.compile(
    r"positivity rate for influenza among ILI samples \(n= ?(\d+)\) in the community "
    r"was (\d+)% in E-week (\d+)\.")
_RE_COVID_POS = re.compile(
    r"positivity rate for COVID-19 among ARI samples \(n= ?(\d+)\) in the community "
    r"was (\d+)% in E-week (\d+)\.")
_RE_ADU_TREND = re.compile(r"Adult samples: ([^.]+)\.")
_RE_PAE_TREND = re.compile(r"Paediatric samples: ([^.]+)\.")
_RE_ARI_ATTEND = re.compile(
    r"average daily number of patients seeking treatment in the polyclinics for ARI "
    r"was (\d+) \(([^)]+)\) in E-\s?week (\d+)\.")

# Lines on page 1 that look table-ish but are not disease rows.
_NON_DISEASE = re.compile(
    r"(?i)^(widen? |cumulative|median|polyclinic|week|vol\.|epidemiological|na$)")


def _norm_name(name):
    name = re.sub(r"[#^*]", "", name)
    return re.sub(r"\s+", " ", name).strip()




def _num(tok):
    return None if tok == "NA" else int(tok)


def _parse_disease_rows(page_text):
    """page 1 -> (cases, other): 5-column and 3-column disease rows."""
    cases, other = {}, {}
    lines = [ln.strip() for ln in page_text.splitlines()]
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = _ROW.match(ln)
        if m:
            name = _norm_name(m.group("name"))
            toks = re.split(r"\s{2,}", m.group("nums").strip())
            i += 1
        else:
            nm = _NAME_ONLY.match(ln)
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            mm = _NUMS_ONLY.match(nxt)
            if nm and mm:
                name = _norm_name(nm.group(0))
                toks = mm.group("nums").split()
                i += 2
            else:
                i += 1
                continue
        if _NON_DISEASE.match(name):
            continue
        vals = [_num(t) for t in toks]
        if len(vals) == 5:
            cases[name] = {
                "week": vals[0], "prev_week": vals[1], "median_5yr": vals[2],
                "cum": vals[3], "cum_prev": vals[4],
            }
        elif len(vals) == 3:
            other[name] = {"week": vals[0], "prev_week": vals[1], "median_5yr": vals[2]}
    return cases, other


def _parse_narrative(pages_text):
    """Flatten all pages (whitespace-normalized) and pull the key narratives."""
    flat = re.sub(r"\s+", " ", "\n".join(pages_text))
    out = {"dengue": {}, "influenza": {}, "ari": {}}

    m = _RE_DENGUE_NOTIF.search(flat)
    if m:
        out["dengue"]["notifications"] = int(m.group(1))
    m = _RE_DENGUE_ADMISSIONS.search(flat)
    if m:
        out["dengue"]["hospital_admissions"] = int(m.group(1))
    m = _RE_DENGUE_SEROTYPES.search(flat)
    if m:
        out["dengue"]["serotypes_pct"] = {
            "DEN-1": float(m.group(3)), "DEN-2": float(m.group(4)),
            "DEN-3": float(m.group(5)), "DEN-4": float(m.group(6)),
        }
        out["dengue"]["serotyped_month"] = "%s %s" % (m.group(1), m.group(2))

    m = _RE_FLU_TYPES.search(flat)
    if m:
        out["influenza"]["specimens_positive"] = int(m.group(1))
        out["influenza"]["month"] = "%s %s" % (m.group(2), m.group(3))
        out["influenza"]["types_pct"] = {
            "A(H1N1)pdm09": float(m.group(5)), "A(H3N2)": float(m.group(7)),
            "B": float(m.group(9)),
        }
    m = _RE_ILI_POS.search(flat)
    if m:
        out["influenza"]["ili_positivity_pct"] = int(m.group(2))
        out["influenza"]["ili_samples"] = int(m.group(1))
    m = _RE_COVID_POS.search(flat)
    if m:
        out["ari"]["covid_positivity_pct"] = int(m.group(2))
        out["ari"]["ari_samples"] = int(m.group(1))
    m = _RE_ADU_TREND.search(flat)
    if m:
        out["ari"]["top_pathogens_adult"] = m.group(1).strip()
    m = _RE_PAE_TREND.search(flat)
    if m:
        out["ari"]["top_pathogens_paediatric"] = m.group(1).strip()
    m = _RE_ARI_ATTEND.search(flat)
    if m:
        out["ari"]["polyclinic_attendances"] = int(m.group(1))
        out["ari"]["working_days"] = m.group(2)
    for k in ("dengue", "influenza", "ari"):
        if not out[k]:
            out[k] = None
    return out


def parse_widb(pdf_bytes):
    """Parse a WIDB PDF -> the `disease_week` block (raises on unusable input)."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")

    header = pages[0] if pages else ""
    mv = _VOL.search(header)
    mw = _WEEK.search(header)
    if not mw:
        raise ValueError("no EPIDEMIOLOGICAL WEEK header found")
    year = int(mv.group(3)) if mv else None
    ew = int(mw.group(1))
    date_range = re.sub(r"\s+", " ", mw.group(2)).strip()

    cases, other = _parse_disease_rows(pages[0] if pages else "")
    if "Dengue Fever" not in cases:
        raise ValueError("dengue row missing from table — layout changed?")
    narr = _parse_narrative(pages)

    df = cases["Dengue Fever"]
    dhf = cases.pop("Dengue Haemorrhagic Fever", None)
    dengue = {
        "week": df["week"], "prev_week": df["prev_week"],
        "median_5yr": df["median_5yr"], "cum": df["cum"], "cum_prev": df["cum_prev"],
        "dhf": dhf,
    }
    dengue.update(narr["dengue"] or {})

    notable = []
    if df["week"] is not None:
        line = "%d new dengue fever cases (prev week %s; same-week 5-yr median %s)" % (
            df["week"],
            "?" if df["prev_week"] is None else df["prev_week"],
            "?" if df["median_5yr"] is None else df["median_5yr"])
        extra = []
        if dengue.get("notifications") is not None:
            extra.append("%d notifications" % dengue["notifications"])
        if dengue.get("hospital_admissions") is not None:
            extra.append("%d hospital admissions" % dengue["hospital_admissions"])
        if extra:
            line += " — " + ", ".join(extra)
        notable.append(line)
    flu = narr["influenza"] or {}
    if flu.get("ili_positivity_pct") is not None:
        line = "influenza: ILI positivity %d%%" % flu["ili_positivity_pct"]
        if flu.get("types_pct"):
            top = max(flu["types_pct"].items(), key=lambda kv: kv[1])
            line += " (top subtype %s %g%%)" % (top[0], top[1])
        notable.append(line)
    ari = narr["ari"] or {}
    if ari.get("polyclinic_attendances") is not None:
        a = other.get("Acute Upper Respiratory Infections") or {}
        line = "ARI polyclinic attendances %s/day" % ari["polyclinic_attendances"]
        if a.get("median_5yr") is not None:
            line += " (same-week 5-yr median %s)" % a["median_5yr"]
        if ari.get("covid_positivity_pct") is not None:
            line += "; COVID-19 ARI positivity %d%%" % ari["covid_positivity_pct"]
        notable.append(line)
    hfmd = other.get("Hand, Foot And Mouth Disease")
    if hfmd and hfmd.get("week") is not None:
        notable.append("HFMD %d cases (prev week %s)" % (
            hfmd["week"], "?" if hfmd["prev_week"] is None else hfmd["prev_week"]))

    return {
        "source": "WIDB %s EW %d, CDA weekly infectious diseases bulletin" % (
            year or "?", ew),
        "epi_week": "%s-W%02d" % (year or "?", ew),
        "date_range": date_range,
        "dengue_cases_national": df["week"],
        "dengue": dengue,
        "cases": cases,
        "other": other,
        "influenza": flu or None,
        "ari": ari or None,
        "notable": notable,
    }


# --- archive + fetch (walk-back + disk cache) ----------------------------------------

def list_bulletins(year):
    """CDA archive page for `year` -> {epi_week_int: pdf_url} (raises on failure)."""
    raw = get_bytes(config.WIDB_ARCHIVE_URL.format(year=year),
                    extra_headers=config.WIDB_UA)
    html = raw.decode("utf-8", "replace")
    out = {}
    for m in re.finditer(r'href="([^"]+\.pdf)"', html):
        url = m.group(1)
        ew = re.search(r"EW\s*(\d+)", url)
        if ew:
            out[int(ew.group(1))] = url
    return out


def _cache_path():
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, "widb_latest.json")


def fetch_latest(use_cache=True):
    """Newest parseable WIDB bulletin -> (payload, error)."""
    path = _cache_path()
    if use_cache and os.path.exists(path):
        if time.time() - os.path.getmtime(path) < config.WIDB_CACHE_TTL_SECONDS:
            try:
                with open(path) as fh:
                    cached = json.load(fh)
                if cached.get("epi_week"):
                    return cached, None
            except (OSError, ValueError):
                pass  # corrupt cache -> refetch

    bulletins = {}
    now_year = datetime.date.today().year
    for y in (now_year, now_year - 1):
        try:
            bulletins = list_bulletins(y)
        except Exception:
            continue
        if bulletins:
            break
    if not bulletins:
        return None, "no WIDB PDFs found on the CDA archive pages"

    errors = []
    for ew in sorted(bulletins, reverse=True)[:config.WIDB_MAX_LOOKBACK]:
        # Filenames contain spaces — urllib needs them percent-encoded.
        url = urllib.parse.quote(bulletins[ew], safe=":/%?&=")
        try:
            raw = get_bytes(url, extra_headers=config.WIDB_UA)
        except Exception as e:
            errors.append("EW%d download failed: %s" % (ew, e))
            continue
        if not raw.startswith(b"%PDF"):
            errors.append("EW%d not a PDF (S3 error page)" % ew)
            continue
        try:
            payload = parse_widb(raw)
        except Exception as e:
            errors.append("EW%d parse failed: %s" % (ew, e))
            continue
        payload["url"] = url
        try:
            with open(path, "w") as fh:
                json.dump(payload, fh)
        except OSError:
            pass
        return payload, None
    return None, "; ".join(errors)