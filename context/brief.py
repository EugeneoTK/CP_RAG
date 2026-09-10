"""Brief prompt builder (Phase 6) — stdlib only, pure functions.

Builds the (system, user) prompt pair for the one-shot LLM brief. The user
payload is a *projection* of the snapshot; `protocol_links` is deliberately
excluded — the brief LLM must never see protocol names or basis notes. The
Protocol spotlight UI section is rendered directly from structured
snapshot.protocol_links.active, so basis-tag provenance holds structurally,
not by model compliance. Spec:
docs/superpowers/specs/2026-09-09-clinician-brief-library-design.md §8.1.
"""
import json

_SYSTEM = (
    "You write a one-screen daily clinical briefing for a Singapore "
    "primary-care clinic from live population-level data.\n"
    "Rules:\n"
    "1. The JSON provided is the ONLY data source. Never invent values, "
    "places, or statistics. If a block is absent, that source is unavailable "
    "this cycle.\n"
    "2. Live signals are observations about the area. NEVER attribute them to "
    "clinic protocols, care protocols, or clinical guidelines. You are NOT "
    "given protocol names or guideline content, and you must not use the "
    "words protocol or guideline (in any form) anywhere in your output. "
    "Protocol relevance is rendered separately from structured data.\n"
    "3. A URA written permission is NOT an opened facility — never present "
    "one as an existing service.\n"
    "4. Reflect data_gaps: name the affected area in outlook or the relevant "
    "watch item (\"unavailable this cycle\").\n"
    "5. If nothing is elevated, say so plainly — a calm week is a valid "
    "headline. Do not manufacture urgency.\n"
    "6. Output STRICT JSON only — no markdown fences, no commentary: "
    "{\"headline\": string (max 25 words), \"watch\": [{\"finding\": max 25 "
    "words, \"why_it_matters\": max 30 words, \"action\": max 30 words, "
    "\"source\": one of NEA | data.gov.sg | CDA WIDB | URA | derived}] (max 6 "
    "items, most clinically important first), \"outlook\": string (max 40 "
    "words)}."
)


def _clip_dicts(lst, n):
    return [x for x in (lst or [])[:n] if isinstance(x, dict)]


def _project(snapshot, corpus_stats):
    """Compact projection of the snapshot for the prompt (spec §8.1)."""
    meta = snapshot.get("meta") or {}
    clinic = meta.get("clinic") or {}
    air = snapshot.get("air_quality") or {}
    psi, pm = air.get("psi") or {}, air.get("pm25") or {}
    wx = snapshot.get("weather") or {}
    today = wx.get("today") or {}
    den = snapshot.get("dengue") or {}
    dw = snapshot.get("disease_week") or {}
    d_den, d_flu = dw.get("dengue") or {}, dw.get("influenza") or {}
    cc = snapshot.get("catchment_change") or {}
    ns = snapshot.get("nearest_services") or {}
    payload = {
        "clinic": {k: clinic[k] for k in
                   ("name", "lat", "lon", "nea_region_approx", "town")
                   if clinic.get(k) is not None},
        "generated_at": meta.get("generated_at"),
        "air": {"psi_peak": psi.get("peak"),
                "psi_peak_region": psi.get("peak_region"),
                "pm25_national": pm.get("national")},
        "weather": {"today_date": today.get("date"),
                    "today_high_c": today.get("high_c"),
                    "today_low_c": today.get("low_c"),
                    "flood_alerts": wx.get("flood")},
        "dengue": {"clusters_active_total": den.get("clusters_active_total"),
                   "nearby_clusters": _clip_dicts(den.get("nearby_clusters"), 3),
                   "in_high_aedes_area": den.get("in_high_aedes_area")},
        "widb": {"epi_week": dw.get("epi_week"),
                 "date_range": dw.get("date_range"),
                 "notable": dw.get("notable"),
                 "dengue_week": d_den.get("week"),
                 "dengue_median_5yr": d_den.get("median_5yr"),
                 "flu_ili_positivity_pct": d_flu.get("ili_positivity_pct")},
        "planning": {"window": cc.get("window"),
                     "healthcare_decisions_90d_count":
                         cc.get("healthcare_decisions_90d_count"),
                     "category_counts": cc.get("category_counts"),
                     "near_clinic_count": cc.get("near_clinic_count"),
                     "recent_decisions":
                         _clip_dicts(cc.get("healthcare_decisions_90d"), 3)},
        "polyclinics": _clip_dicts(ns.get("polyclinics"), 3),
        "data_gaps": snapshot.get("data_gaps") or [],
        "corpus": corpus_stats or {},
    }
    # A source that failed (gap) simply omits its block — absent = unavailable.
    return {k: v for k, v in payload.items()
            if not (isinstance(v, dict) and not any(x is not None for x in v.values()))}


def format_brief_prompt(snapshot, corpus_stats):
    """Return (system, user) strings for the one-shot brief LLM call."""
    return _SYSTEM, json.dumps(_project(snapshot, corpus_stats), ensure_ascii=False)