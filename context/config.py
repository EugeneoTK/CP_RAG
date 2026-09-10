"""Endpoints, dataset ids, thresholds and test points for the context spike.

All values verified live 2026-09-08 (see docs/signal-research.md §1–§2).
"""

import os
import tempfile

# --- real-time API (key optional; x-api-key only raises rate limits) -------------
RT_BASE = "https://api-open.data.gov.sg/v2/real-time/api"
RT_ENDPOINTS = {
    "pm25": RT_BASE + "/pm25",
    "psi": RT_BASE + "/psi",
    "forecast_24hr": RT_BASE + "/twenty-four-hr-forecast",
    "forecast_2hr": RT_BASE + "/two-hr-forecast",
    "outlook_4day": RT_BASE + "/four-day-outlook",
    "wbgt": RT_BASE + "/weather?api=wbgt",
    "flood_alerts": RT_BASE + "/weather/flood-alerts",
}

# --- WIDB (CDA weekly infectious-diseases bulletin, PDF-only, weekly) ------------
# Archive page is per-year; the current year's page is the one that gets updated.
# Browser UA required (isomer-user-content S3 rejects non-browser UAs).
WIDB_ARCHIVE_URL = "https://www.cda.gov.sg/resources/weekly-infectious-diseases-bulletin-{year}/"
WIDB_UA = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
}
WIDB_MAX_LOOKBACK = 5      # walk back at most N epi-weeks to find a parseable PDF
WIDB_CACHE_TTL_SECONDS = 3 * 24 * 3600  # weekly publication cadence

# --- URA e-Services (Phase 4 — planning decisions) ---------------------------------
# Auth: AccessKey header -> daily token (insertNewToken/v1) -> data calls carry
# AccessKey + Token headers (invokeUraDS/v1). Docs: eservice.ura.gov.sg/maps/api/.
URA_BASE = "https://eservice.ura.gov.sg/uraDataService"
URA_WINDOW_DAYS = 90             # last_dnload_date window (API max lookback: 1 year)
URA_CACHE_TTL_SECONDS = 24 * 3600  # data cadence is daily; token valid for the day
URA_MAX_ITEMS = 50               # snapshot cap on the decision list (Phase 8: 20->50 so category + near counts cover more rows)
URA_HEALTHCARE_KEYWORDS = ("POLYCLINIC", "CLINIC", "MEDICAL", "NURSING HOME",
                           "CHILD CARE", "SENIOR")
# "near clinic" band for the street-area heuristic (rough: distance to an area
# centroid, not the facility). Phase 9 (2026-09-10): 10 -> 2 km so the "near"
# count reflects the walkable catchment rather than the whole western island.
URA_NEAR_KM = 2.0

# --- NTUC Health Active Ageing (Phase 9 — community referral signal) -------------
# The LIVE centre set + monthly programme-calendar PDFs come from the calendar
# landing page (one PDF per centre, rotates monthly — the same source
# rag.py's ingest_community_refresh() ingests). Centre coordinates are STATIC:
# verified live 2026-09-10 from https://ntuchealth.sg/active-ageing/locations
# (Next.js flight payload, per-centre `position` fields). Centres rarely move;
# if the live set gains a centre missing here, it renders with no distance
# (extend this table). Names match the calendar-page anchor text exactly.
NTUC_CALENDAR_URL = ("https://ntuchealth.sg/active-ageing/services/"
                     "active-ageing-programme-calendars")
NTUC_UA = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 "
                   "Safari/537.36"),
    "Accept-Language": "en-SG,en;q=0.9",
}
NTUC_CACHE_TTL_SECONDS = 24 * 3600   # centre set changes rarely; month rotates
ACTIVE_AGING_NEAR_KM = 2.0           # "within 2 km" band for the dashboard card
ACTIVE_AGING_CENTRES = (
    ("Bishan", 1.358589, 103.845633),
    ("Boon Lay", 1.346983, 103.708823),
    ("Bukit Batok West", 1.356857, 103.740045),
    ("Bukit Merah", 1.281860, 103.826590),
    ("Bukit Merah Silat", 1.277398, 103.830521),
    ("Bukit Merah View", 1.284263, 103.821965),
    ("Bukit Panjang", 1.377576, 103.772503),
    ("Bedok North", 1.330432, 103.932571),
    ("Gek Poh", 1.347500, 103.699177),
    ("Jurong Central Plaza", 1.349189, 103.725004),
    ("Kampung Admiralty", 1.440064, 103.800795),
    ("Kampung Kembangan", 1.325561, 103.910794),
    ("Lengkok Bahru", 1.288213, 103.814203),
    ("Marsiling", 1.439203, 103.778010),
    ("Marsiling Park", 1.435072, 103.774243),
    ("Mount Faber", 1.274362, 103.808792),
    ("Nanyang", 1.344567, 103.693268),
    ("Pasir Ris", 1.367630, 103.956280),
    ("Pioneer", 1.336961, 103.700276),
    ("Redhill", 1.287395, 103.817332),
    ("Serangoon Central", 1.348158, 103.874808),
    ("Taman Jurong", 1.335681, 103.722648),
    ("Tampines", 1.349719, 103.951372),
    ("Telok Blangah", 1.271528, 103.823027),
    ("Whampoa", 1.327748, 103.861187),
    ("Wisma Geylang Serai", 1.314880, 103.896670),
    ("Woodlands East", 1.439810, 103.803586),
)

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

# --- GEOJSON via v1 poll-download (key-free) -------------------------------------
V1_POLL = "https://api-open.data.gov.sg/v1/public/api/datasets/{did}/poll-download"
GEOJSON_DATASETS = {
    "dengue_clusters": "d_dbfabf16158d1b0e1c420627c0819168",
    "aedes_areas": "d_5d060d8b7838a15e8906fb22c50dbf51",
    "polyclinics": "d_b22489c7dc4065b6e7e45f177fdb33be",
}

# --- cache (kept OUT of the repo) -------------------------------------------------
CACHE_DIR = os.path.join(tempfile.gettempdir(), "cp_rag_context_cache")
GEOJSON_CACHE_TTL_SECONDS = 12 * 3600

# --- optional key (rate limits only; never committed) ------------------------------
def _load_key():
    try:
        env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env):
            for line in open(env):
                line = line.strip()
                if line.startswith("DGS_API_KEY="):
                    return line.split("=", 1)[1].strip() or None
    except OSError:
        pass
    return os.environ.get("DGS_API_KEY") or None

DGS_API_KEY = _load_key()


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

# --- test clinic: real GP clinic point (OSM node 8083778017, verified 2026-09-09) ---
# A private GP clinic, not a polycline — the app persona is a GP chronic-care practice.
# Chosen 2026-09-09 for the richest live brief: west-region PM2.5 56 (island peak 75,
# central — haze day) + a 60-case dengue cluster (Ho Ching Rd, upd 2026-09-03) 1.1 km
# away + a 2-case cluster 0.3 km away (5 active protocol links, data_gaps []).
TEST_CLINIC = {
    "name": "Lakeside Family Medicine Clinic (test clinic)",
    "lat": 1.3454017,
    "lon": 103.7188383,
    "postcode": "641518",
}

# Approximate district centroids by first postcode digit (coarse; use --lat/--lon
# for exact points). Sufficient for POC-level context.
POSTCODE_DISTRICTS = {
    "1": {"label": "Downtown", "lat": 1.2890, "lon": 103.8510},
    "2": {"label": "Outram / Chinatown", "lat": 1.2800, "lon": 103.8440},
    "3": {"label": "Bukit Merah / One-North", "lat": 1.3010, "lon": 103.7950},
    "4": {"label": "Bedok / Joo Chiat", "lat": 1.3210, "lon": 103.9240},
    "5": {"label": "Queenstown / Redhill", "lat": 1.2960, "lon": 103.8150},
    "6": {"label": "Clementi / Jurong West", "lat": 1.3140, "lon": 103.7650},
    "7": {"label": "Woodlands / Yishun", "lat": 1.4300, "lon": 103.7900},
    "8": {"label": "Sengkang / Punggol / Tampines", "lat": 1.4000, "lon": 103.9300},
}

# --- NEA 5-region approximation (deterministic partition; documented as coarse) ----
def nea_region(lat, lon):
    if lat >= 1.40:
        return "north"
    if lat < 1.30:
        return "south"
    if lon < 103.76:
        return "west"
    if lon > 103.90:
        return "east"
    return "central"

# --- signal thresholds (conservative, documented) ----------------------------------
PM25_MODERATE = 15.0   # µg/m³
PM25_HIGH = 25.0
PSI_UNHEALTHY = 101    # SG PSI scale: Good 0-50, Moderate 51-100, Unhealthy 101-200,
PSI_VERY_UNHEALTHY = 201  # Very Unhealthy 201-300, Dangerous 301+
WBGT_HOT = 31.0        # °C — NEA heat-stress watch level
HEAT_STRESS_HIGH = ("Moderate", "High")
DENGUE_NEAR_KM = 3.0
RAINY_WORDS = ("thunder", "shower", "rain", "drizzle")
