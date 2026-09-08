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

# --- test clinic: real polycline point from d_b22489c7 (verified 2026-09-08) -------
TEST_CLINIC = {
    "name": "Woodlands Polycline (test clinic)",
    "lat": 1.4308932241216819,
    "lon": 103.77521291746736,
    "postcode": "738579",
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
