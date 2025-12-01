#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# get nearest scooter parking location by comparing nearest 3 points and calculating path (Dijkstra)

import csv
import json
import math
import sys
import time
import heapq
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import rospy
from gps_utils import wait_for_latlon
from get_nearest_kpoints import get_points

# ──────────────────────────────
# CONSTANTS (no CLI flags)
# ──────────────────────────────
MASK_JSON   = Path("/home/rezoom/catkin_ws/src/escooter/scripts/map/fixed_campus_map_2.json")
BBOX        = {"west": -76.956482, "south": 38.980763, "east": -76.933136, "north": 39.002110}
GRID_W, GRID_H = 8704, 10240

SNAP_RADIUS_CELLS = 50
CROP_PAD_CELLS    = 128
SAFETY_TARGET     = 4

# TXT outputs
TXT_OUT_LINES = Path("/home/rezoom/catkin_ws/src/waypoint_nav/outdoor_waypoint_nav/waypoint_files/scooter_path_output(1).txt")  # lat,lon per line
# TXT_OUT_FMT2  = Path("/home/rezoom/catkin_ws/src/waypoint_nav/outdoor_waypoint_nav/waypoint_files/scooter_path_output(2).txt")  # numbered like screenshot

# Optional resampling to produce cleaner waypoint spacing
BASE_INTERVAL_MI = 0.0045
MIN_POINTS       = 10

# ──────────────────────────────
# Geo helpers
# ──────────────────────────────
R_EARTH_M = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R_EARTH_M * math.atan2(math.sqrt(a), math.sqrt(1-a))

def path_length_miles(coords_ll: List[Tuple[float,float]]) -> float:
    d = 0.0
    for i in range(len(coords_ll)-1):
        lon1, lat1 = coords_ll[i]
        lon2, lat2 = coords_ll[i+1]
        d += haversine_m(lat1, lon1, lat2, lon2)
    return d / 1609.34

def miles_to_meters(miles: float) -> float:
    return miles * 1609.344

def lnglat_to_grid(lon: float, lat: float) -> Tuple[int, int]:
    u = (lon - BBOX['west']) / (BBOX['east'] - BBOX['west'])
    v = (BBOX['north'] - lat) / (BBOX['north'] - BBOX['south'])
    x = min(GRID_W-1, max(0, int(math.floor(u * GRID_W))))
    y = min(GRID_H-1, max(0, int(math.floor(v * GRID_H))))
    return x, y

def grid_to_lnglat(x: int, y: int) -> Tuple[float, float]:
    lon = BBOX['west'] + (x + 0.5) / GRID_W * (BBOX['east'] - BBOX['west'])
    lat = BBOX['north'] - (y + 0.5) / GRID_H * (BBOX['north'] - BBOX['south'])
    return lon, lat

# ──────────────────────────────
# Mask loading (0=walkable, 1=obstacle)
# ──────────────────────────────
def load_mask_01(path: Path) -> np.ndarray:
    raw = json.loads(path.read_text())
    H = len(raw); W = len(raw[0])
    m = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        row = raw[y]
        for x in range(W):
            v = row[x]
            m[y, x] = 0 if (v == 1 or v is True) else 1
    return m

# ──────────────────────────────
# Clearance (8-neighborhood) & DIJKSTRA
# ──────────────────────────────
def clearance_8(mask01: np.ndarray) -> np.ndarray:
    H, W = mask01.shape
    INF = 10**9
    dist = np.full((H, W), INF, dtype=np.int32)
    qx = np.empty(H*W, dtype=np.int32)
    qy = np.empty(H*W, dtype=np.int32)
    head = 0; tail = 0
    obs_y, obs_x = np.where(mask01 == 1)
    for yy, xx in zip(obs_y, obs_x):
        dist[yy, xx] = 0
        qx[tail] = xx; qy[tail] = yy; tail += 1
    DX = (-1,0,1,-1,1,-1,0,1); DY = (-1,-1,-1,0,0,1,1,1)
    while head < tail:
        x = int(qx[head]); y = int(qy[head]); head += 1
        nd = dist[y, x] + 1
        for k in range(8):
            nx, ny = x + DX[k], y + DY[k]
            if nx < 0 or nx >= W or ny < 0 or ny >= H: continue
            if dist[ny, nx] > nd:
                dist[ny, nx] = nd
                qx[tail] = nx; qy[tail] = ny; tail += 1
    big = max(H, W)
    dist[(mask01 == 0) & (dist == INF)] = big
    return dist

def dijkstra_8_no_corner(grid01: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    """
    Dijkstra on 8-connected grid with 'no corner cutting' rule.
    grid01[y,x] == 0 → free, 1 → blocked.
    Returns list of (x,y) from start to goal (inclusive), or [] if no path.
    """
    H, W = grid01.shape

    def inb(x, y):
        return 0 <= x < W and 0 <= y < H and grid01[y, x] == 0

    if not inb(*start) or not inb(*goal):
        return []

    MOVES = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
             (1,1,math.sqrt(2)),(1,-1,math.sqrt(2)),
             (-1,1,math.sqrt(2)),(-1,-1,math.sqrt(2))]

    pq: List[Tuple[float, Tuple[int,int]]] = []
    heapq.heappush(pq, (0.0, start))
    cost = {start: 0.0}
    parent: dict[Tuple[int,int], Tuple[int,int]] = {}

    while pq:
        gcost, node = heapq.heappop(pq)
        if node == goal:
            path = [node]
            while node in parent:
                node = parent[node]
                path.append(node)
            return path[::-1]

        if gcost > cost.get(node, float("inf")):
            continue

        x, y = node
        for dx, dy, step_cost in MOVES:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if grid01[ny, nx] != 0:
                continue
            # no corner cutting for diagonals
            if dx != 0 and dy != 0:
                if grid01[y, x + dx] != 0 or grid01[y + dy, x] != 0:
                    continue
            ng = gcost + step_cost
            if ng < cost.get((nx, ny), float("inf")):
                cost[(nx, ny)] = ng
                parent[(nx, ny)] = (x, y)
                heapq.heappush(pq, (ng, (nx, ny)))
    return []

# ──────────────────────────────
# Snap & crop
# ──────────────────────────────
def nearest_walkable(mask01: np.ndarray, x: int, y: int, max_r: int) -> Optional[Tuple[int,int]]:
    H, W = mask01.shape
    if 0 <= x < W and 0 <= y < H and mask01[y, x] == 0:
        return (x, y)
    for r in range(1, max_r+1):
        for dx in range(-r, r+1):
            xx = x + dx
            yy1 = y - r
            yy2 = y + r
            if 0 <= xx < W and 0 <= yy1 < H and mask01[yy1, xx] == 0:
                return (xx, yy1)
            if 0 <= xx < W and 0 <= yy2 < H and mask01[yy2, xx] == 0:
                return (xx, yy2)
        for dy in range(-r+1, r):
            yy = y + dy
            xx1 = x - r
            xx2 = x + r
            if 0 <= xx1 < W and 0 <= yy < H and mask01[yy, xx1] == 0:
                return (xx1, yy)
            if 0 <= xx2 < W and 0 <= yy < H and mask01[yy, xx2] == 0:
                return (xx2, yy)
    return None

# ──────────────────────────────
# Data I/O helpers
# ──────────────────────────────
def load_latlons_headered(path, lat_col="lat", lon_col="lon"):
    pts = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pts.append((float(row[lat_col]), float(row[lon_col])))
    return pts

def calculateDistance(latlon1, latlon2, radius=6371000.0):
    """Great-circle distance between two (lat, lon) pairs in degrees. Returns meters."""
    lat1, lon1 = latlon1
    lat2, lon2 = latlon2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    a = min(1.0, max(0.0, a))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c

# ──────────────────────────────
# Optional resampling (even spacing)
# ──────────────────────────────
def resample_uniform(coords_ll: List[Tuple[float,float]], base_interval_mi: float, min_points: int) -> List[Tuple[float,float]]:
    if not coords_ll:
        return []
    if len(coords_ll) == 1:
        return coords_ll[:]
    cum = [0.0]
    for i in range(1, len(coords_ll)):
        lon1, lat1 = coords_ll[i-1]
        lon2, lat2 = coords_ll[i]
        cum.append(cum[-1] + haversine_m(lat1, lon1, lat2, lon2) / 1609.34)
    total = cum[-1]
    interval = base_interval_mi if base_interval_mi > 0 else 0.001
    est = int(total / interval) + 1
    if est < min_points and min_points > 1:
        interval = max(1e-6, total / (min_points - 1))
    out = []
    t = 0.0
    j = 0
    while t <= total + 1e-9:
        while j < len(cum) - 2 and cum[j+1] < t:
            j += 1
        if j == len(cum) - 1:
            out.append(coords_ll[-1]); break
        seg = cum[j+1] - cum[j]
        if seg <= 1e-12:
            out.append(coords_ll[j+1])
        else:
            r = (t - cum[j]) / seg
            lon = coords_ll[j][0] + (coords_ll[j+1][0] - coords_ll[j][0]) * r
            lat = coords_ll[j][1] + (coords_ll[j+1][1] - coords_ll[j][1]) * r
            out.append((lon, lat))
        t += interval
    if out[-1] != coords_ll[-1]:
        out.append(coords_ll[-1])
    return out

# ──────────────────────────────
# Output function
# ──────────────────────────────
def output_optimal_path(sampled_ll: List[Tuple[float,float]]):
    """
    sampled_ll: list of (lon, lat)

    Prints to terminal as JSON:
      [{"lat":..., "long":...}, ...]
    Saves to (1).txt as:
      lat,lon
    Saves to (2).txt as numbered human-readable lines like:
      1 lat, lon
      2 lat, lon
      ...
    """
    # to JSON (lat/long dicts) for terminal
    dict_list = [{"lat": float(lat), "long": float(lon)} for (lon, lat) in sampled_ll]
    print(json.dumps(dict_list, ensure_ascii=False))

    # ensure output dir
    TXT_OUT_FMT2.parent.mkdir(parents=True, exist_ok=True)

    # (1) classic lines (no index, no space)
    with TXT_OUT_LINES.open("w", encoding="utf-8") as f:
        for lon, lat in sampled_ll:
            f.write(f"{lat:.6f},{lon:.6f}\n")

    # (2) numbered lines like the screenshot (index + space + "lat, lon")
    with TXT_OUT_FMT2.open("w", encoding="utf-8") as f:
        for i, (lon, lat) in enumerate(sampled_ll, start=1):
            f.write(f"{i} {lat:.6f}, {lon:.6f}\n")

# ──────────────────────────────
# Main
# ──────────────────────────────
def main():
    # Load candidate parking points
    veohub_pts = load_latlons_headered("/home/rezoom/catkin_ws/src/escooter/scripts/alan/outputs/umd_veohubs_latlon.csv", lat_col="lat", lon_col="lon")
    bikerack_pts = load_latlons_headered("/home/rezoom/catkin_ws/src/escooter/scripts/alan/outputs/umd_racks_latlon.csv", lat_col="lat", lon_col="lon")
    all_parking_pts = veohub_pts + bikerack_pts

    # Current GNSS fix
    rospy.init_node("nearest_parking_node", anonymous=True)
    start_lat, start_lon, alt = wait_for_latlon("/fix", timeout=5.0)
    print(f"Current location of ReZoom is at lat={start_lat:.8f}, lon={start_lon:.8f}, alt={alt:.2f}")

    # Nearest K by straight-line distance
    nearestPoints = get_points((start_lat, start_lon), all_parking_pts, k=3)
    print("Candidate parking spots in ABSOLUTE straight-line distances (meters):")
    for (lat, lon) in nearestPoints:
        m = calculateDistance((start_lat, start_lon), (lat, lon))
        print(f"  → lat={lat:.6f}, lon={lon:.6f} : {m:.3f} m")

    # Load mask
    mask01 = load_mask_01(MASK_JSON)
    H, W = mask01.shape
    if (H, W) != (GRID_H, GRID_W):
        print(f"[WARN] Mask size {W}x{H} != expected {GRID_W}x{GRID_H}. Proceeding.")

    # Snap start
    sx_raw, sy_raw = lnglat_to_grid(start_lon, start_lat)
    ns = nearest_walkable(mask01, sx_raw, sy_raw, SNAP_RADIUS_CELLS)
    if ns is None:
        print("No walkable start within snap radius.")
        sys.exit(2)
    sx, sy = ns

    # Try each candidate goal → Dijkstra path
    distances = []   # [((lat, lon), meters)]
    results   = []   # [(goal_lat, goal_lon, miles, sampled_ll, used_r)]
    for goal_lat, goal_lon in nearestPoints:
        ex_raw, ey_raw = lnglat_to_grid(goal_lon, goal_lat)
        ne = nearest_walkable(mask01, ex_raw, ey_raw, SNAP_RADIUS_CELLS)
        if ne is None:
            print(f"Skip goal ({goal_lat:.6f},{goal_lon:.6f}): cannot snap to walkable.")
            continue
        ex, ey = ne

        # Crop
        x0 = max(0, min(sx, ex) - CROP_PAD_CELLS)
        x1 = min(W-1, max(sx, ex) + CROP_PAD_CELLS)
        y0 = max(0, min(sy, ey) - CROP_PAD_CELLS)
        y1 = min(H-1, max(sy, ey) + CROP_PAD_CELLS)
        sub = mask01[y0:y1+1, x0:x1+1].copy()

        # Clearance & inflation sweep
        clr = clearance_8(sub)
        path_rel = None
        used_r = None
        for r in range(SAFETY_TARGET, -1, -1):
            walk = (sub == 0) & (clr > r)
            inflated = np.where(walk, 0, 1).astype(np.uint8)
            path = dijkstra_8_no_corner(inflated, (sx-x0, sy-y0), (ex-x0, ey-y0))
            if path:
                path_rel = path
                used_r = r
                break

        if not path_rel:
            print("No path found even with r=0. Check mask connectivity / points.")
            continue

        # To lon/lat & resample
        coords_ll = [grid_to_lnglat(x+x0, y+y0) for (x, y) in path_rel]  # (lon, lat)
        sampled_ll = resample_uniform(coords_ll, BASE_INTERVAL_MI, MIN_POINTS)
        miles = path_length_miles(sampled_ll)
        meters = miles_to_meters(miles)

        distances.append(((goal_lat, goal_lon), meters))
        results.append((goal_lat, goal_lon, miles, sampled_ll, used_r))

    if not distances:
        print("No reachable parking points among the nearest candidates.")
        return

    print("Candidate parking spots in PATH distances (meters):")
    for (lat, lon), m in distances:
        print(f"  → lat={lat:.6f}, lon={lon:.6f} : {m:.3f} m")

    # Choose best by meters
    (best_lat, best_lon), best_meters = min(distances, key=lambda t: t[1])

    # Fetch record for best goal
    best_rec = None
    for rec in results:
        goal_lat, goal_lon, miles, sampled_ll, used_r = rec
        if abs(goal_lat - best_lat) < 1e-9 and abs(goal_lon - best_lon) < 1e-9:
            best_rec = rec
            break
    if best_rec is None:
        print("[ERROR] Internal: best goal not found in results.")
        return

    goal_lat, goal_lon, best_miles, best_sampled_ll, used_r = best_rec
    print(f"\nNearest parking point by PATH: lat={goal_lat:.8f}, lon={goal_lon:.8f}, "
          f"distance={best_meters:.3f} m (r={used_r}, ~{best_miles:.3f} mi)\n")

    # Output files + terminal JSON
    output_optimal_path(best_sampled_ll)

if __name__ == "__main__":
    main()
