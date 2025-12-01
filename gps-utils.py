# gps_utils.py
import rospy
from sensor_msgs.msg import NavSatFix

def wait_for_latlon(topic="/fix", timeout=5.0):
    """Block until a NavSatFix arrives, then return (lat, lon, alt)."""
    msg = rospy.wait_for_message(topic, NavSatFix, timeout=timeout)
    return (msg.latitude, msg.longitude, msg.altitude)