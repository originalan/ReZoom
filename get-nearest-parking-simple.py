# get nearest scooter parking location with direct distance calculation

import math
import csv
import rospy
from gps_utils import wait_for_latlon


def load_latlons_headered(path, lat_col="lat", lon_col="lon"):
    pts = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pts.append((float(row[lat_col]), float(row[lon_col])))
    return pts

def calculateDistance(latlon1, latlon2, radius=6371000.0):
    """
    Great-circle distance between two (lat, lon) pairs in degrees.
    Returns meters by default (Earth mean radius = 6,371,000 m).
    """
    lat1, lon1 = latlon1
    lat2, lon2 = latlon2

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    # numerical safety for extreme cases
    a = min(1.0, max(0.0, a))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius * c

# example UMD to DC distance = about 13475 meters
# d = calculateDistance((38.9869, -76.9426), (38.8895, -77.0353))
# print(round(d, 4), "meters")

def find_nearest_parking(current_latlon, parking_latlons):
    nearest_pt = None
    nearest_dist = float("inf")
    for pt in parking_latlons:
        dist = calculateDistance(current_latlon, pt)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_pt = pt
    return nearest_pt, nearest_dist

def main():
    veohub_pts = load_latlons_headered("/home/rezoom/catkin_ws/src/escooter/scripts/alan/outputs/umd_veohubs_latlon.csv", lat_col="lat", lon_col="lon")
    bikerack_pts = load_latlons_headered("/home/rezoom/catkin_ws/src/escooter/scripts/alan/outputs/umd_racks_latlon.csv", lat_col="lat", lon_col="lon")
    all_parking_pts = veohub_pts + bikerack_pts
    
    rospy.init_node("nearest_parking_node", anonymous=True)
    lat, lon, alt = wait_for_latlon("/fix", timeout=5.0)
    print(f"Current location of ReZoom is at lat={lat:.8f}, lon={lon:.8f}, alt={alt:.2f}")

    point, dist = find_nearest_parking((lat, lon), all_parking_pts)
    print(f"Nearest parking point at lat={point[0]:.8f}, lon={point[1]:.8f}, distance={dist:.3f} meters")

if __name__ == "__main__":
    main()