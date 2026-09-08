"""Fetchers for the verified live signals (all key-free; DGS_API_KEY optional).

Each fetcher returns (payload, error) where error is None on success and a short
string on failure — a failing source must never crash the snapshot; it is recorded
in data_gaps instead.
"""

import json
import os
import time

from . import config
from .http import get_bytes, get_json


def _headers():
    h = {}
    if config.DGS_API_KEY:
        h["x-api-key"] = config.DGS_API_KEY
    return h


def fetch_nea_reading(slug_key):
    """pm25 / psi: data.items[-1].readings.{metric} per region."""
    url = config.RT_ENDPOINTS[slug_key]
    doc = get_json(url, extra_headers=_headers())
    data = doc.get("data") or {}
    items = data.get("items") or []
    if not items:
        return None, "no items in response"
    latest = items[-1]
    readings = latest.get("readings") or {}
    metric = None
    for v in readings.values():
        if isinstance(v, dict):
            metric = v
            break
    if metric is None:
        return None, "no readings in latest item"
    return {
        "regions": {k: v for k, v in metric.items() if isinstance(v, (int, float))},
        "national": metric.get("national"),
        "timestamp": latest.get("timestamp"),
        "updated": latest.get("updatedTimestamp"),
    }, None


def fetch_forecast_24hr():
    url = config.RT_ENDPOINTS["forecast_24hr"]
    doc = get_json(url, extra_headers=_headers())
    records = (doc.get("data") or {}).get("records") or []
    if not records:
        return None, "no records"
    rec = records[0]
    gen = rec.get("general") or {}
    return {
        "date": rec.get("date"),
        "updated": rec.get("updatedTimestamp"),
        "high_c": (gen.get("temperature") or {}).get("high"),
        "low_c": (gen.get("temperature") or {}).get("low"),
        "text": (gen.get("forecast") or {}).get("text"),
        "wind": gen.get("wind"),
        "humidity_pct": gen.get("relativeHumidity"),
    }, None


def fetch_outlook_4day():
    url = config.RT_ENDPOINTS["outlook_4day"]
    doc = get_json(url, extra_headers=_headers())
    records = (doc.get("data") or {}).get("records") or []
    if not records:
        return None, "no records"
    days = []
    for f in (records[0].get("forecasts") or [])[:4]:
        days.append({
            "day": f.get("day"),
            "high_c": (f.get("temperature") or {}).get("high"),
            "low_c": (f.get("temperature") or {}).get("low"),
            "text": (f.get("forecast") or {}).get("text"),
            "summary": (f.get("forecast") or {}).get("summary"),
        })
    return {"updated": records[0].get("updatedTimestamp"), "days": days}, None


def fetch_forecast_2hr_town(lat, lon):
    """Two-hour town forecast; town = nearest area_metadata point to clinic."""
    url = config.RT_ENDPOINTS["forecast_2hr"]
    doc = get_json(url, extra_headers=_headers())
    data = doc.get("data") or {}
    meta = data.get("area_metadata") or []
    items = data.get("items") or []
    if not meta or not items:
        return None, "no data"
    towns = [
        {"name": m.get("name"),
         "lat": (m.get("label_location") or {}).get("latitude"),
         "lon": (m.get("label_location") or {}).get("longitude")}
        for m in meta
        if (m.get("label_location") or {}).get("latitude") is not None
    ]
    from .geo import nearest_by_distance
    town, km = nearest_by_distance(lat, lon, towns)
    forecast_text = None
    latest = items[-1]
    for f in (latest.get("forecasts") or []):
        if f.get("area") == town:
            forecast_text = f.get("forecast")
            break
    return {
        "town": town,
        "distance_km": round(km, 1) if km is not None else None,
        "forecast": forecast_text,
        "valid_period": (latest.get("valid_period") or {}).get("text"),
        "updated": latest.get("update_timestamp"),
    }, None


def fetch_wbgt(lat, lon):
    url = config.RT_ENDPOINTS["wbgt"]
    doc = get_json(url, extra_headers=_headers())
    records = (doc.get("data") or {}).get("records") or []
    if not records:
        return None, "no records"
    rec = records[0]
    readings = (rec.get("item") or {}).get("readings") or []
    stations = []
    for r in readings:
        loc = r.get("location") or {}
        try:
            sla = float(loc.get("latitude"))
            slo = float(loc.get("longitude"))
            wval = float(r.get("wbgt"))
        except (TypeError, ValueError):
            continue
        from .geo import haversine_km
        stations.append({
            "name": (r.get("station") or {}).get("name"),
            "wbgt_c": wval,
            "heat_stress": r.get("heatStress"),
            "km": round(haversine_km(lat, lon, sla, slo), 1),
            "lat": sla,
            "lon": slo,
        })
    if not stations:
        return None, "no station readings"
    nearest = min(stations, key=lambda s: s["km"])
    worst_stress = "Low"
    for s in stations:
        if s["heat_stress"] == "High":
            worst_stress = "High"
            break
        if s["heat_stress"] == "Moderate":
            worst_stress = "Moderate"
    return {
        "datetime": rec.get("datetime"),
        "nearest_station": nearest,
        "max_wbgt_c": max(s["wbgt_c"] for s in stations),
        "max_heat_stress": worst_stress,
        "stations_observed": len(stations),
    }, None


def fetch_flood_alerts():
    url = config.RT_ENDPOINTS["flood_alerts"]
    doc = get_json(url, extra_headers=_headers())
    data = doc.get("data") or {}
    records = data.get("records") or []
    active = []
    for rec in records[:20]:
        item = rec.get("item") or {}
        readings = item.get("readings")
        if readings:  # non-empty readings => an actual alert entry
            active.append({
                "datetime": rec.get("datetime"),
                "type": item.get("type"),
                "readings": readings,
            })
    latest = records[0] if records else None
    return {
        "active_alerts": active,
        "latest_record": {
            "datetime": (latest or {}).get("datetime"),
            "type": ((latest or {}).get("item") or {}).get("type"),
        },
    }, None


# --- GEOJSON via v1 poll-download (cached in system temp, never in repo) -----------

def _cache_path(key):
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, key + ".geojson")


def fetch_geojson(key, use_cache=True):
    """key in config.GEOJSON_DATASETS. Returns (features, error)."""
    path = _cache_path(key)
    if use_cache and os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < config.GEOJSON_CACHE_TTL_SECONDS:
            try:
                with open(path) as fh:
                    gj = json.load(fh)
                return gj.get("features", []), None
            except (OSError, ValueError):
                pass  # fall through to refetch
    did = config.GEOJSON_DATASETS[key]
    poll = get_json(config.V1_POLL.format(did=did))
    url = (poll.get("data") or {}).get("url")
    if not url:
        return None, "no download url in poll response: " + json.dumps(poll)[:200]
    raw = get_bytes(url)
    gj = json.loads(raw.decode("utf-8"))
    feats = gj.get("features", [])
    try:
        with open(path, "w") as fh:
            json.dump(gj, fh)
    except OSError:
        pass
    return feats, None

