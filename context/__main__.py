"""CLI: build a context snapshot for a clinic point.

Usage:
    venv/bin/python -m context                                   # test clinic
    venv/bin/python -m context --lat 1.430893 --lon 103.775213 --name "My clinic"
    venv/bin/python -m context --postcode 738579                 # district approx
    venv/bin/python -m context --json                            # raw JSON
    venv/bin/python -m context --no-cache                        # ignore GEOJSON cache
"""

import argparse
import json
import sys

from . import config
from .snapshot import build_snapshot


def main(argv=None):
    ap = argparse.ArgumentParser(prog="context", description=__doc__)
    ap.add_argument("--lat", type=float, default=None, help="clinic latitude")
    ap.add_argument("--lon", type=float, default=None, help="clinic longitude")
    ap.add_argument("--postcode", default=None,
                    help="clinic postcode (approximate district centroid)")
    ap.add_argument("--name", default=None, help="clinic name label")
    ap.add_argument("--json", action="store_true", help="emit raw JSON only")
    ap.add_argument("--no-cache", action="store_true",
                    help="refetch GEOJSON layers, the WIDB bulletin and the URA planning window")
    args = ap.parse_args(argv)

    if args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
        name = args.name or "Clinic @ %.4f,%.4f" % (lat, lon)
        pc = None
    elif args.postcode:
        pc = args.postcode.zfill(6)
        d = config.POSTCODE_DISTRICTS.get(pc[0])
        if not d:
            print("error: unknown postcode first digit %r (use --lat/--lon)" % pc[0],
                  file=sys.stderr)
            return 2
        lat, lon = d["lat"], d["lon"]
        name = args.name or ("Postcode %s (~%s)" % (pc, d["label"]))
        pc = pc
    else:
        t = config.TEST_CLINIC
        lat, lon, pc = t["lat"], t["lon"], t["postcode"]
        name = args.name or t["name"]

    snap = build_snapshot(lat, lon, name, use_cache=not args.no_cache)

    if args.json:
        print(json.dumps(snap, indent=2, ensure_ascii=False))
        return 0

    c = snap["meta"]["clinic"]
    print("=" * 74)
    print("CONTEXT SNAPSHOT — %s (%s, NEA region ~%s, town ~%s)"
          % (c["name"], pc or "n/a", c["nea_region_approx"], c.get("town") or "?"))
    print("generated %s" % snap["meta"]["generated_at"])
    print("=" * 74)

    aq = snap["air_quality"]
    if "pm25" in aq:
        p = aq["pm25"]
        print("AIR      PM2.5 peak %.0f ug/m3 (%s)  regions %s  updated %s"
              % (p["peak"], p["peak_region"], aq["pm25"]["regions"],
                 (p["updated"] or "?")[11:16]))
    if "psi" in aq:
        p = aq["psi"]
        print("         PSI   peak %.0f (%s)  updated %s"
              % (p["peak"], p["peak_region"], (p["updated"] or "?")[11:16]))

    w = snap["weather"]
    if "today" in w:
        t = w["today"]
        print("WEATHER  today %s/%s C — %s  (humidity %s)"
              % (t["high_c"], t["low_c"], t["text"],
                 (t.get("humidity_pct") or {}).get("high", "?")))
    if "outlook_4day" in w:
        for d in w["outlook_4day"]["days"]:
            print("         %s: %s/%s C — %s"
                  % (d["day"], d["high_c"], d["low_c"], d["text"]))
    if "town_2hr" in w:
        tv = w["town_2hr"]
        print("         2-hr near %s (%.1f km): %s" % (tv["town"], tv["distance_km"] or 0,
                                                        tv["forecast"]))
    if "wbgt" in w:
        v = w["wbgt"]
        ns = v["nearest_station"]
        print("         WBGT max %.1f C (heat stress %s); nearest %s: %.1f C/%s %.1f km"
              % (v["max_wbgt_c"], v["max_heat_stress"], ns["name"], ns["wbgt_c"],
                 ns["heat_stress"], ns["km"]))
    if "flood" in w:
        f = w["flood"]
        n = len(f["active_alerts"])
        print("         flood: %d active alert(s); latest record %s"
              % (n, f["latest_record"].get("datetime")))

    d = snap["dengue"]
    print("DENGUE   %d active cluster polygons; nearby (<=3km):" % d.get("clusters_active_total", 0))
    if d.get("nearby_clusters"):
        for cc in d["nearby_clusters"]:
            print("         - %s: %d cases, %.1f km, updated %s"
                  % (cc["locality"], cc["case_size"], cc["km"], cc["updated"]))
    else:
        print("         - none within 3 km")
    print("         high-Aedes areas: %d total; clinic inside: %s"
          % (d.get("aedes_areas_total", 0), "YES" if d.get("in_high_aedes_area") else "no"))

    dw = snap.get("disease_week") or {}
    if dw.get("epi_week"):
        print("DISEASE  WIDB %s (%s), CDA weekly bulletin"
              % (dw["epi_week"], dw.get("date_range") or "?"))
        for n in dw.get("notable") or []:
            print("         - " + n)

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

    if "nearest_services" in snap:
        for p in snap["nearest_services"]["polyclinics"]:
            print("SERVICES polyclinic: %s (%s) %.1f km" % (p["name"], p["postcode"], p["km"]))

    pl = snap["protocol_links"]
    print("-" * 74)
    print("PROTOCOL LINKS (corpus: %d protocols; %d active signal(s))"
          % (len(pl["corpus"]), len(pl["active"])))
    print(pl["context_brief"])

    if snap["data_gaps"]:
        print("-" * 74)
        print("DATA GAPS:")
        for g in snap["data_gaps"]:
            print("  - %s" % g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
