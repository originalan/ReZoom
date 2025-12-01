import numpy as np
import math

"""
Expected formats supported in 123.py (any ONE of these):

1) Variables:
   START  = (lat, lon)       # or [lat, lon] or dict {'lat':..., 'lon':...}
   GOALS  = [(lat,lon), (lat,lon), ...]   # list, at least one

2) Function:
   def get_points():
       return {
         "start": (lat, lon),
         "goals": [(lat,lon), (lat,lon), ...]
       }

3) Single list 'POINTS' where first is start and others are goals:
   POINTS = [(lat, lon), (lat, lon), ...]

4) Stdout (if you `print`):
   a) JSON: {"start":[lat,lon], "goals":[[lat,lon], ...]}
   b) Lines:
      START: lat,lon
      GOAL:  lat,lon
      GOAL:  lat,lon
"""

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

def get_points(current_latlon, points, k=3):
    """
    Returns the closest k points to the current latlon location based on the list of 'points' given
    """
    # fast shortlist by straight-line distance
    scored = [(p, calculateDistance(current_latlon, p)) for p in points]
    scored.sort(key=lambda x: x[1])
    return [p for (p, _) in scored[:k]]

