#!/usr/bin/env python3
"""ACE clinical-guidelines crawler (Phase 5, optional seeder).

sitemap.xml -> 29 guideline detail pages -> "Download the ACG" PDF links ->
download + ingest into the RAG store via rag.ingest_pdf (dedupes by hash,
so re-runs cost 0 credits for already-ingested PDFs).

Usage:
  venv/bin/python scripts/ace_guidelines.py --list             # dry run
  venv/bin/python scripts/ace_guidelines.py --ingest --limit 2  # try 2 pages first
  venv/bin/python scripts/ace_guidelines.py --ingest           # the full seed

Cost: one embedding batch per NEW pdf (~a few cents for all 29).
"""
import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# Same as app.py: load .env before importing rag (rag.py itself never loads it).
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import rag

BASE = "https://www.ace-hta.gov.sg"
UA = rag._BROWSER_UA
DETAIL_RE = re.compile(
    r"^https://www\.ace-hta\.gov\.sg/healthcare-professionals/"
    r"ace-repository-for-clinical-guidelines/[a-z0-9-]+/?$")


def _get(url):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


def detail_pages():
    """All guideline detail-page URLs from the sitemap.

    (The list page paginates client-side React; the sitemap has all 29.)
    The regex admits only /ace-repository-for-clinical-guidelines/<slug>/ —
    it excludes the index page (no slug) and ...-guidelines-overview/.
    """
    root = ET.fromstring(_get(BASE + "/sitemap.xml"))
    locs = [el.text.strip() for el in root.iter()
            if el.tag == "loc" or el.tag.endswith("}loc")]
    return sorted({u for u in locs if u and DETAIL_RE.match(u)})


def fetch_detail(url):
    """(guideline title, [pdf urls]) from one detail page."""
    soup = BeautifulSoup(_get(url), "html.parser")
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else url.rsplit("/", 2)[-2]
    pdfs = []
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if h.lower().endswith(".pdf") and h not in pdfs:
            pdfs.append(h)
    return title, pdfs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="dry run")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max detail pages")
    args = ap.parse_args()
    if not args.list and not args.ingest:
        ap.error("pass --list or --ingest")

    pages = detail_pages()
    print("detail pages found: %d" % len(pages))
    found = []
    for i, u in enumerate(pages):
        if args.limit and i >= args.limit:
            break
        title, pdfs = fetch_detail(u)
        print("%d. %s" % (i + 1, title))
        for p in pdfs:
            print("     %s" % p)
            found.append((title, p))
        time.sleep(1.0)
    print("PDFs found: %d" % len(found))
    if not args.ingest:
        print("dry run only — re-run with --ingest")
        return

    ingested = skipped = failed = 0
    for title, pdf in found:
        try:
            r = requests.get(pdf, headers={"User-Agent": UA}, timeout=120)
            r.raise_for_status()
            added, was_skipped, _ = rag.ingest_pdf(
                r.content, title, source=pdf, source_site="ace-hta.gov.sg")
        except Exception as e:
            failed += 1
            print("  FAIL %s (%s)" % (pdf, e))
            time.sleep(1.0)
            continue
        if was_skipped:
            skipped += 1
            print("  skip (already in store) %s" % pdf)
        else:
            ingested += 1
            print("  + %d chunks  %s" % (added, title))
        time.sleep(1.0)
    print("done: %d ingested, %d skipped, %d failed" % (ingested, skipped, failed))


if __name__ == "__main__":
    main()
