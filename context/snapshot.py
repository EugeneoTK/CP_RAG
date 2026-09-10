"""Assemble the context snapshot for a clinic point.

Snapshot schema follows docs/signal-research.md §7, extended with
protocol_links (see linkage.py). A failing source never aborts the build —
it lands in data_gaps so the snapshot is always emitted.
"""

import datetime
import time

from . import community, config, fetchers, ura, widb
from .geo import haversine_km, point_in_polygon
from .linkage import PROTOCOLS, build_protocol_links


def _now_sgt():
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")


def _fmt_date(s):
    """FMEL_UPD_D yyyymmddHHMMSS -> YYYY-MM-DD (best effort)."""
    if not s or len(str(s)) < 8:
        return None
    s = str(s)
    return "%s-%s-%s" % (s[0:4], s[4:6], s[6:8])


def _ring_centroid(ring):
    mx = sum(c[0] for c in ring) / len(ring)
    my = sum(c[1] for c in ring) / len(ring)
    return mx, my


def _polygon_km(lat, lon, g):
    if g.get("type") == "Polygon" and (g.get("coordinates") or [[]])[0]:
        mx, my = _ring_centroid(g["coordinates"][0])
        return haversine_km(lat, lon, my, mx)
    if g.get("type") == "MultiPolygon" and g.get("coordinates"):
        poly0 = g["coordinates"][0]
        if poly0 and poly0[0]:
            mx, my = _ring_centroid(poly0[0])
            return haversine_km(lat, lon, my, mx)
    return None


def _nearest_polyclinics(lat, lon):
    feats, err = fetchers.fetch_geojson("polyclinics")
    if err or feats is None:
        return [], err
    out = []
    for f in feats:
        p, g = f.get("properties") or {}, f.get("geometry") or {}
        if g.get("type") != "Point":
            continue
        c = g.get("coordinates") or []
        if len(c) < 2:
            continue
        out.append({
            "name": p.get("NAME"),
            "address": p.get("ADDRESS"),
            "postcode": p.get("POSTALCODE"),
            "km": round(haversine_km(lat, lon, c[1], c[0]), 1),
        })
    out.sort(key=lambda x: x["km"])
    return out[:3], None


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
    cc["near_clinic_count"]. Also cc["nearest_km_by_category"]: min approx_km
    per category over MAPPED rows only — a category whose rows are all
    unmapped gets no entry (unmeasurable, never invented). Mutates and
    returns cc."""
    near = 0
    nearest = {}
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
        cat = a.get("category") or "Other"
        if km < nearest.get(cat, float("inf")):
            nearest[cat] = km
    cc["near_clinic_count"] = near
    cc["nearest_km_by_category"] = nearest
    return cc


def _enrich_ageing(ac, lat, lon):
    """Phase 9: add `km` + `near` per centre (coords from
    config.ACTIVE_AGING_CENTRES; a live centre missing from that table keeps
    no km and is not `near`). Mutates and returns ac."""
    coords = dict((n, (la, lo)) for n, la, lo in config.ACTIVE_AGING_CENTRES)
    near = 0
    for c in ac.get("centres") or []:
        hit = coords.get(c.get("name"))
        if not hit:
            continue
        km = round(haversine_km(lat, lon, hit[0], hit[1]), 1)
        c["km"] = km
        c["near"] = km <= config.ACTIVE_AGING_NEAR_KM
        if c["near"]:
            near += 1
    ac["near_count"] = near
    return ac


def build_snapshot(lat, lon, name, use_cache=True):
    gaps = []
    snap = {
        "meta": {
            "generated_at": _now_sgt(),
            "version": "0.1.0",
            "licence": "Singapore Open Data License (commercial use permitted)",
            "clinic": {"name": name, "lat": lat, "lon": lon,
                       "nea_region_approx": config.nea_region(lat, lon)},
        },
        "air_quality": {},
        "weather": {},
        "dengue": {},
        "protocol_links": {},
        "data_gaps": gaps,
    }

    # --- air quality ---------------------------------------------------------------
    peaks = {}
    for label in ("pm25", "psi"):
        val, err = fetchers.fetch_nea_reading(label)
        time.sleep(2)
        if err:
            gaps.append("%s: %s" % (label, err))
            continue
        regions = val["regions"]
        peak = max(regions.values()) if regions else None
        snap["air_quality"][label] = {
            "regions": regions,
            "peak": peak,
            "peak_region": max(regions, key=regions.get) if regions else None,
            "national": val.get("national"),
            "updated": val.get("updated"),
        }
        peaks[label] = peak

    # --- weather ---------------------------------------------------------------------
    val, err = fetchers.fetch_forecast_24hr()
    time.sleep(2)
    if err:
        gaps.append("forecast_24hr: %s" % err)
    else:
        snap["weather"]["today"] = val

    val, err = fetchers.fetch_outlook_4day()
    time.sleep(2)
    if err:
        gaps.append("outlook_4day: %s" % err)
    else:
        snap["weather"]["outlook_4day"] = val

    val, err = fetchers.fetch_forecast_2hr_town(lat, lon)
    time.sleep(2)
    if err:
        gaps.append("forecast_2hr: %s" % err)
    else:
        snap["weather"]["town_2hr"] = val
        snap["meta"]["clinic"]["town"] = val.get("town")

    val, err = fetchers.fetch_wbgt(lat, lon)
    time.sleep(2)
    if err:
        gaps.append("wbgt: %s" % err)
    else:
        snap["weather"]["wbgt"] = val

    val, err = fetchers.fetch_flood_alerts()
    if err:
        gaps.append("flood_alerts: %s" % err)
    else:
        snap["weather"]["flood"] = val

    # --- dengue / aedes ---------------------------------------------------------------
    feats, err = fetchers.fetch_geojson("dengue_clusters", use_cache)
    if err:
        gaps.append("dengue_clusters: %s" % err)
    else:
        nearby = []
        for f in feats:
            p, g = f.get("properties") or {}, f.get("geometry") or {}
            try:
                size = int(p.get("CASE_SIZE"))
            except (TypeError, ValueError):
                size = 0
            km = _polygon_km(lat, lon, g)
            if km is not None and km <= config.DENGUE_NEAR_KM:
                nearby.append({
                    "locality": p.get("LOCALITY"),
                    "case_size": size,
                    "km": round(km, 1),
                    "updated": _fmt_date(p.get("FMEL_UPD_D")),
                })
        nearby.sort(key=lambda x: x["km"])
        snap["dengue"]["clusters_active_total"] = len(feats)
        snap["dengue"]["nearby_clusters"] = nearby

    aedes_in = False
    feats, err = fetchers.fetch_geojson("aedes_areas", use_cache)
    if err:
        gaps.append("aedes_areas: %s" % err)
    else:
        snap["dengue"]["aedes_areas_total"] = len(feats)
        for f in feats:
            g = f.get("geometry") or {}
            if point_in_polygon(lat, lon, g):
                aedes_in = True
                break
    snap["dengue"]["in_high_aedes_area"] = aedes_in

    # --- WIDB: CDA weekly infectious diseases bulletin (national, Phase 3) ---------------
    time.sleep(2)
    widb_val, widb_err = widb.fetch_latest(use_cache)
    if widb_err:
        gaps.append("widb: %s" % widb_err)
    else:
        snap["disease_week"] = widb_val

    # --- URA planning decisions (catchment_change, Phase 4; needs URA_ACCESS_KEY) -------
    time.sleep(2)
    cc, cc_err = ura.fetch_catchment(use_cache)
    if cc_err:
        gaps.append("ura: %s" % cc_err)
    else:
        snap["catchment_change"] = _enrich_ura(cc, lat, lon)

    # --- NTUC Active Ageing centres (community referral, Phase 9) -----------------------
    time.sleep(2)
    ac, ac_err = community.fetch_centres(use_cache)
    if ac_err:
        gaps.append("active_ageing: %s" % ac_err)
    else:
        snap["active_ageing"] = _enrich_ageing(ac, lat, lon)

    # --- nearest polyclines (service availability) -------------------------------------
    pcs, err = _nearest_polyclinics(lat, lon)
    if err:
        gaps.append("polyclinics: %s" % err)
    else:
        snap["nearest_services"] = {"polyclinics": pcs}

    # --- protocol linkage (the differentiator) ------------------------------------------
    links, brief = build_protocol_links(
        {"pm25_peak": peaks.get("pm25"), "psi_peak": peaks.get("psi")},
        snap["weather"], snap["dengue"])
    snap["protocol_links"] = {
        "corpus": PROTOCOLS,
        "active": links,
        "context_brief": "\n".join(brief),
        "basis_legend": {
            "corpus": "claim is in the protocol text",
            "partial": "mechanism in protocol text, trigger not",
            "derived": "clinical synthesis — not protocol text",
            "none": "counselling only, no clinical claim",
        },
        "usage": (
            "Phase 2 (implemented in rag.py build_chain): the RAG prompt gets "
            "'Protocol content' (only corpus/partial material, per basis_note) "
            "and 'Local context' (all live signals, never attributable to the "
            "protocols). 'derived' links are never presented as protocol "
            "content. Active protocol names are appended to the retrieval "
            "query (mechanism B)."
        ),
    }
    return snap
