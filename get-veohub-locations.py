# GET UMD BIKE RACK LOCATIONS FROM ARCGIS SERVICES

import requests
import numpy as np
from pathlib import Path
import json
from time import sleep

# <<< PUT YOUR LAYER QUERY ENDPOINTS HERE >>>
# Example shapes:
#   https://<host>/arcgis/rest/services/.../FeatureServer/12/query
#   https://<host>/arcgis/rest/services/.../MapServer/7/query
LAYER_QUERY_URLS = [
    "https://services9.arcgis.com/1rOwFRpAwrxe0rBl/arcgis/rest/services/Transportation/FeatureServer/3/query",   # veo scooters (?)
    # "https://services9.arcgis.com/1rOwFRpAwrxe0rBl/arcgis/rest/services/Transportation/FeatureServer/0/query",   # lATLON of all bike amenities
]

# Common ArcGIS Query params to fetch all features in WGS84 (lat/lon)
BASE_PARAMS = {
    "f": "json",
    "where": "1=1",               # no filter
    "outFields": "*",
    "returnGeometry": "true",
    "outSR": "4326",              # WGS84 lat/lon
    "resultOffset": 0,
    "resultRecordCount": 2000,    # page size (ArcGIS caps ~2000)
}

def fetch_all_points(query_url):
    pts = []
    offset = 0
    while True:
        params = dict(BASE_PARAMS, resultOffset=offset)
        r = requests.get(query_url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()

        features = payload.get("features", [])
        if not features:
            break

        for ft in features:
            geom = ft.get("geometry", {})
            # Points are usually x=lon, y=lat. Some layers use 'points' (multipoint) or 'rings' (polygon).
            if "x" in geom and "y" in geom:            # Point
                pts.append((geom["y"], geom["x"]))
            elif "points" in geom:                      # MultiPoint
                for x, y in geom["points"]:
                    pts.append((y, x))
            elif "rings" in geom:                       # Polygon outline vertices (unlikely for racks)
                for ring in geom["rings"]:
                    for x, y in ring:
                        pts.append((y, x))
            # (You can add 'paths' handler for polylines if ever needed.)

        # Pagination handling
        exceeded = payload.get("exceededTransferLimit", False)
        if not exceeded:
            break
        offset += len(features)
        sleep(0.2)  # be polite

    return pts

# Fetch and merge all layers
all_pts = []
for url in LAYER_QUERY_URLS:
    all_pts.extend(fetch_all_points(url))

# Deduplicate (round to ~7 decimal places ≈ 0.011 m)
dedup = {(round(lat, 7), round(lon, 7)) for (lat, lon) in all_pts}
points_latlon = np.array(sorted(list(dedup)), dtype=float)   # shape (N, 2), columns: [lat, lon]

# Save outputs
Path("outputs").mkdir(exist_ok=True)
np.savetxt("outputs/umd_veohubs_latlon.csv", points_latlon, delimiter=",", header="lat,lon", comments="")
with open("outputs/umd_veohubs_latlon.json", "w") as f:
    json.dump({"points": points_latlon.tolist()}, f, indent=2)

print(f"Collected {points_latlon.shape[0]} unique points.")