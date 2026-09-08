"""Small geo helpers (stdlib only): haversine, point-in-polygon, nearest town.

Coordinate order throughout: [lon, lat] (GeoJSON convention) for rings,
(lat, lon) tuples for points.
"""

import math

EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def _ring_contains(x, y, ring):
    """Ray casting. ring = list of [lon, lat]."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y):
            xint = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xint:
                inside = not inside
        j = i
    return inside


def point_in_polygon(lat, lon, geometry):
    """geometry: GeoJSON geometry dict (Polygon or MultiPolygon)."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        # outer ring only; holes are small/irregular in these datasets
        return _ring_contains(lon, lat, coords[0])
    if gtype == "MultiPolygon":
        return any(_ring_contains(lon, lat, poly[0]) for poly in coords)
    return False


def nearest_by_distance(lat, lon, candidates):
    """candidates: list of dicts with 'name', 'lat', 'lon'.
    Returns (name, km) of the nearest, or (None, None)."""
    best, best_km = None, None
    for c in candidates:
        km = haversine_km(lat, lon, c["lat"], c["lon"])
        if best_km is None or km < best_km:
            best, best_km = c["name"], km
    return best, best_km
