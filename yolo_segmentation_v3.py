#!/usr/bin/env python3

# Tuning knobs you'll most often adjust:
#   ~person_distance_threshold
#   ~speed_gain
#   ~person_speed_gain
#   ~perp_gain — how much extra buffer to add per m/s of lateral relative speed.
#   ~max_dist_jump — meters allowed between frames for same person; tighten if IDs switch when people cross.
#   ~max_misses — how many frames a person can disappear (occlusion) before we drop the track.
#   ~filter_alpha — higher = more responsive, lower = smoother speeds. 
#   ~ttc_horizon, ~dca_limit — crossing risk parameters. Nudge ttc_horizon up (e.g. 2.0 → 3.0 s) and/or dca_limit up (0.8 → 1.0 m) to catch more crossers.
#   ~assoc_xy_gate — max lateral distance (m) for associating detections to tracks. Lower to reduce ID switches when people cross.
#   ~min_alert_age — min frames before a track can raise an alert. Increase to reduce false alarms on spurious tracks.
#   ~min_vel_age — min frames before a track shows velocity arrows. Increase to avoid showing arrows for shaky tracks.
#   ~min_cross_speed — min lateral speed (m/s) to consider for crossing risk. lower = more sensitive to slow walkers

#improvements: right now everything is relative to the escooter, not to the world. also, perpendicular detection works on the basis that the scooter is always moving exactly forward, which isn't always the case. 

# simulating scooter velocity:

# rostopic pub /odom nav_msgs/Odometry "
# twist:
#   twist:
#     linear:
#       x: 1.0
#       y: 1.0
#       z: 0.0
# "

import cv2

import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg
from geometry_msgs.msg import PointStamped
from std_msgs.msg import UInt8  

from visualization_msgs.msg import Marker, MarkerArray

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import rospy
import math
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32MultiArray, Float32MultiArray
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import message_filters

from sensor_msgs.msg import CameraInfo

from ultralytics import YOLO  # YOLOv8

DETECTION_DISTANCE = 1.2  # m
VEHICLE_SPEED_GAIN = 0.3   # m per m/s
PERSON_SPEED_GAIN  = 0.3   # m per m/s of person radial speed
FRAME_SKIP = 2             # process every (N+1)th frame
MAX_DISTANCE_JUMP = 1.75   # m (inter-frame distance gate)
MAX_POSSIBLE_THRESHOLD_DISTANCE = 8.0  # m

# Hazard levels (match supervisor)
CLEAR, WARN, SLOW, STOP = 0, 1, 2, 3

# ---------------- Single-person detection structure ----------------
@dataclass
class PersonDet:
    """
    Single person detection from YOLO+depth.

    - (cx, cy): mask centroid in pixel coordinates
    - dist:     robust range from depth (m)
    - mask:     boolean mask in image space (HxW)
    - X, Y:     metric coordinates in camera frame (m)
    - conf:     YOLO confidence score (0-1)
    """
    cx: int
    cy: int
    dist: float
    mask: np.ndarray  # bool HxW
    X: float
    Y: float
    conf: float

@dataclass
class OffenderInfo:
    """
    Info about the track that most violates its threshold
    (i.e., with the most negative dist - thr margin).
    """
    tid: int
    dist: float
    threshold: float
    vrad: Optional[float]   # radial speed (m/s); None if unknown
    t_star: float           # time to closest approach (s)
    d_star: float           # distance at closest approach (m)
    vnorm: float            # |v_xy| (m/s)


@dataclass
class AlertInfo:
    """
    Global alert state for this frame.

    - active:  True if any track violates its threshold.
    - margin:  minimum (dist - thr) across all tracks
               (negative => violation).
    - offender: OffenderInfo for the worst offender, or None if safe.
    """
    active: bool
    margin: float
    offender: Optional[OffenderInfo]

# ---------------- Small per-person track ----------------
class Track:
    __slots__ = ("id","cx","cy","dist","filt_dist","prev_time","prev_for_speed",
                 "radial","misses","xy","prev_xy","vxy","age", "conf")

    def __init__(self, tid:int, cx:int, cy:int, dist:float, tstamp:float, xy:Tuple[float,float], conf:float=1.0):
        self.id = tid
        self.cx, self.cy = cx, cy
        self.dist = dist
        self.filt_dist = dist
        self.prev_for_speed = dist
        self.prev_time = tstamp
        self.radial = None
        self.misses = 0
        self.xy = xy
        self.prev_xy = None
        self.vxy = np.array([0.0, 0.0], dtype=float)
        self.age = 1  # frames this track has been matched
        self.conf = conf

    def update(self, cx:int, cy:int, dist:float, tstamp:float, alpha:float,
               xy:Tuple[float,float], conf:float=1.0, dt_lo:float=0.02, dt_hi:float=0.5):
        self.dist = dist
        self.filt_dist = alpha*dist + (1.0-alpha)*self.filt_dist
        dt = tstamp - self.prev_time if self.prev_time is not None else None
        if dt is not None and (dt_lo < dt < dt_hi):
            self.radial = (self.prev_for_speed - self.filt_dist) / dt  # >0 approaching
            if self.prev_xy is not None:
                dx = xy[0] - self.prev_xy[0]
                dy = xy[1] - self.prev_xy[1]
                self.vxy = np.array([dx/dt, dy/dt], dtype=float)
        else:
            self.radial = None
        self.prev_for_speed = self.filt_dist
        self.prev_time = tstamp
        self.cx, self.cy = cx, cy
        self.prev_xy = self.xy
        self.xy = xy
        self.misses = 0
        self.age += 1  # one more successful association
        self.conf = conf

# ---------------- Main Node ----------------
class YoloPeopleProximityNode:
    def __init__(self):
        self.bridge = CvBridge()

        # ---------------- Params ----------------
        self.base_threshold = rospy.get_param("~person_distance_threshold", DETECTION_DISTANCE)
        self.speed_gain     = rospy.get_param("~speed_gain", VEHICLE_SPEED_GAIN)
        self.person_speed_gain = rospy.get_param("~person_speed_gain", PERSON_SPEED_GAIN)

        self.speed_topic    = rospy.get_param("~speed_topic", "/odom")
        self.frame_skip     = rospy.get_param("~frame_skip", FRAME_SKIP)
        self._frame_count   = 0

        # --- TF for frame transforms ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Camera frame to use for transforms (can also just read from rgb_msg.header.frame_id)
        self.camera_frame = rospy.get_param("~camera_frame", "zed2i_left_camera_optical_frame")
        self.base_frame   = rospy.get_param("~base_frame", "base_link")
        self.marker_frame = rospy.get_param("~marker_frame", self.base_frame)

        self.marker_pub = rospy.Publisher("~people_markers", MarkerArray, queue_size=1)

        # FOV visualization (for RViz only)
        self.fov_angle_deg = rospy.get_param("~fov_angle_deg", 98.0)  # total horizontal FOV
        self.fov_range     = rospy.get_param("~fov_range", 8.0)       # how far to draw the FOV rays (m)

        # --- camera intrinsics (px) ---
        self.fx = self.fy = self.cx0 = self.cy0 = None
        self.caminfo_ready = False
        cam_info_topic = rospy.get_param("~camera_info_topic", "/zed_node/left/camera_info")
        self.caminfo_sub = rospy.Subscriber(cam_info_topic, CameraInfo, self._caminfo_cb, queue_size=1)

        # association & maturity
        self.assoc_xy_gate = rospy.get_param("~assoc_xy_gate", 1.4)  # meters
        self.min_alert_age = rospy.get_param("~min_alert_age", 2)    # frames
        self.min_vel_age   = rospy.get_param("~min_vel_age", 2)      # frames

        # horizon / gains for “crossing” risk
        self.ttc_horizon = float(rospy.get_param("~ttc_horizon", 2.5))   # seconds
        self.dca_limit   = float(rospy.get_param("~dca_limit",   3.0))   # m, distance at closest approach
        self.perp_gain   = float(rospy.get_param("~perp_gain",   0.7))   # m of extra buffer per m/s of perpendicular rel speed
        self.cross_x_max = float(rospy.get_param("~cross_x_max", 3.0))   # m, max forward distance where a lateral crossing matters
        self.cross_y_min = float(rospy.get_param("~cross_y_min", 0.20))  # m, ignore tiny lateral offsets near centerline

        # gates / smoothing
        self.max_dist_jump  = rospy.get_param("~max_dist_jump", MAX_DISTANCE_JUMP)  # m (for both single & multi)
        self.max_misses     = rospy.get_param("~max_misses", 5)                    # drop track after N misses
        self.filter_alpha   = rospy.get_param("~filter_alpha", 0.3)                # EMA for distances
        self.max_threshold  = rospy.get_param("~max_threshold", MAX_POSSIBLE_THRESHOLD_DISTANCE)

        self.min_cross_speed = rospy.get_param("~min_cross_speed", 0.15)  # m/s
        self.min_lateral_fraction = rospy.get_param("~min_lateral_fraction", 0.5)  # min fraction of speed that is lateral to consider for crossing risk

        self._published_marker_ids = set()

        # vehicle speed
        self.current_speed  = 0.0

        # single-target (closest) smoothing/history for legacy radial speed (not used now per-track)
        self.alert_active = False

        # ---------------- YOLO model ----------------
        model_path = rospy.get_param("~yolo_model", "yolov8n-seg.pt")  # <- seg model
        self.imgsz     = int(rospy.get_param("~imgsz", 640))           # inference size
        self.conf_th   = float(rospy.get_param("~conf", 0.25))
        self.iou_th    = float(rospy.get_param("~iou", 0.45))
        self.seg_erode = int(rospy.get_param("~seg_erode_iters", 1))   # erode masks to avoid edges

        rospy.loginfo("Loading YOLO model: %s", model_path)
        self.model = YOLO(model_path)
        rospy.loginfo("YOLO model loaded successfully")

        # Topics – adjust namespace if needed
        rgb_topic   = rospy.get_param("~rgb_topic",   "/zed_node/left/image_rect_color")
        depth_topic = rospy.get_param("~depth_topic", "/zed_node/depth/depth_registered")

        # ---------------- Publishers ----------------
        self.alert_pub = rospy.Publisher("~people_proximity_alert", Bool, queue_size=1)
        self.closest_dist_pub = rospy.Publisher("~closest_person_distance", Float32, queue_size=1)
        self.threshold_pub = rospy.Publisher("~dynamic_threshold", Float32, queue_size=1)

        self.publish_debug = rospy.get_param("~publish_debug", True)
        self.debug_img_pub = rospy.Publisher("~debug_image", Image, queue_size=1)

        self.require_masks = rospy.get_param("~require_masks", True)
        
        self.hazard_level_pub  = rospy.Publisher("~hazard_level", UInt8, queue_size=1)
        self.hazard_margin_pub = rospy.Publisher("~hazard_margin", Float32, queue_size=1)
        self.hazard_ttc_pub    = rospy.Publisher("~hazard_ttc", Float32, queue_size=1)
        self.hazard_dca_pub    = rospy.Publisher("~hazard_dca", Float32, queue_size=1)

        # multi-person outputs
        self.ids_pub   = rospy.Publisher("~people_ids", Int32MultiArray, queue_size=1)
        self.dists_pub = rospy.Publisher("~people_distances", Float32MultiArray, queue_size=1)
        self.speeds_pub= rospy.Publisher("~people_radial_speeds", Float32MultiArray, queue_size=1)
        self.confidence_pub = rospy.Publisher("~people_confidences", Float32MultiArray, queue_size=1)

        # ---------------- Subscribers w/ sync ----------------
        rospy.Subscriber(self.speed_topic, Odometry, self.odom_callback)
        rgb_sub   = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=5, slop=0.05)
        self.ts.registerCallback(self.synced_callback)

        # tracking state
        self.tracks: Dict[int, Track] = {}
        self.next_tid = 1

        rospy.loginfo("Node started. base=%.2f, veh_gain=%.2f, person_gain=%.2f", 
                      self.base_threshold, self.speed_gain, self.person_speed_gain)

    # ---------------- CameraInfo callback ----------------
    def _caminfo_cb(self, msg: CameraInfo):
        # Prefer K (3x3 row-major)
        K = msg.K  # [fx,0,cx, 0,fy,cy, 0,0,1]
        if K and len(K) == 9:
            self.fx = float(K[0]); self.fy = float(K[4])
            self.cx0 = float(K[2]); self.cy0 = float(K[5])
            self.caminfo_ready = True
            rospy.loginfo_once("Camera intrinsics set from CameraInfo: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
                            self.fx, self.fy, self.cx0, self.cy0)
            # You may optionally unsubscribe after first good message:
            # self.caminfo_sub.unregister()

    # ---------------- Depth sampling (unchanged) ----------------
    def _robust_depth_from_mask(self, depth_cv: np.ndarray, mask_bool: np.ndarray,
                            erode_iters: int = 1, iqr_k: float = 1.5,
                            min_valid: int = 50, fallback_percentile: int = 25) -> Optional[float]:
        """
        Compute a robust single distance from a person mask.
        - Erodes edges to avoid background leakage.
        - IQR filtering removes outliers.
        Returns median inlier depth [m] or None.
        """
        if mask_bool is None or mask_bool.size == 0:
            return None

        mb = mask_bool.astype(np.uint8)
        if erode_iters > 0:
            k = np.ones((3, 3), np.uint8)
            mb = cv2.erode(mb, k, iterations=erode_iters)

        if not mb.any():
            return None

        vals = depth_cv[mb.astype(bool)]
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0.0]
        if vals.size < min_valid:
            return None

        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
        inliers = vals[(vals >= lo) & (vals <= hi)]

        if inliers.size >= 5:
            return float(np.median(inliers))
        else:
            # Conservative fallback when few inliers
            return float(np.percentile(vals, fallback_percentile))

    # ---------------- Helpers ----------------
    def _publish_track_markers(
        self,
        per_person_thresholds: Dict[int, float],
        alert_info: AlertInfo,
        stamp: rospy.Time,
        dynamic_threshold: float,
    ):
        """
        Publish RViz markers for each tracked person.

        - Spheres: current track positions (ns='people_tracks').
        - Arrows:  velocity vectors (ns='people_vel').
        - TTC/DCA: predicted closest-approach points + rays (ns='people_ttc').
        - FOV:     camera/vehicle FOV wedge (ns='fov').

        All in marker_frame; currently this should match base_frame once TF is wired.
        """
        if self.marker_pub is None:
            return

        marray = MarkerArray()

        offender_tid = alert_info.offender.tid if alert_info.offender is not None else None

        # ──────────────────────────────
        # 1) Person position spheres
        # ──────────────────────────────
        for tid, tr in self.tracks.items():
            if tr.xy is None or not np.isfinite(tr.xy[0]) or not np.isfinite(tr.xy[1]):
                continue

            thr = per_person_thresholds.get(tid, self.base_threshold)
            dist = tr.filt_dist if tr.filt_dist is not None else tr.dist

            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.marker_frame
            marker.ns = "people_tracks"
            marker.id = tid  # stable per-track
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(tr.xy[0])
            marker.pose.position.y = float(tr.xy[1])
            marker.pose.position.z = 0.0

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.25

            inside = (dist is not None) and np.isfinite(dist) and (dist < thr)

            if offender_tid is not None and tid == offender_tid:
                # Worst offender: yellow and slightly bigger
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.scale.x = 0.30
                marker.scale.y = 0.30
                marker.scale.z = 0.30
            elif inside:
                # Inside threshold: red
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            else:
                # Outside threshold: green
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0

            marker.color.a = 0.9
            marker.lifetime = rospy.Duration(0.5)

            marray.markers.append(marker)

            # Confidence text label (above sphere)
            conf = float(getattr(tr, "conf", 1.0))
            conf_pct = int(round(conf*100))
            txt = Marker()
            txt.header.stamp = stamp
            txt.header.frame_id = self.marker_frame
            txt.ns = "people_conf"
            txt.id = tid
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(tr.xy[0])
            txt.pose.position.y = float(tr.xy[1])
            txt.pose.position.z = 0.5 # above the sphere
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.3  # text height
            txt.text = "ID %d: %d%%" % (tid, conf_pct)
            txt.color.r = 1.0
            txt.color.g = 1.0
            txt.color.b = 1.0
            txt.color.a = 0.9
            txt.lifetime = rospy.Duration(0.5)
            marray.markers.append(txt)

        # ──────────────────────────────
        # 2) Velocity arrows
        # ──────────────────────────────
        VELOCITY_SCALE = 0.75  # meters per (m/s); adjust visually

        for tid, tr in self.tracks.items():
            if tr.xy is None or tr.vxy is None:
                continue
            if not (np.isfinite(tr.xy[0]) and np.isfinite(tr.xy[1])):
                continue

            # Only draw arrows for reasonably mature tracks
            if tr.age < self.min_vel_age:
                continue
            if tr.misses > 0:
                continue

            vx, vy = float(tr.vxy[0]), float(tr.vxy[1])
            speed_norm = math.hypot(vx, vy)
            if speed_norm < 1e-3:
                continue  # skip essentially static

            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.marker_frame
            marker.ns = "people_vel"
            marker.id = tid
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # start at current position
            marker.points = []
            p0 = geometry_msgs.msg.Point()
            p1 = geometry_msgs.msg.Point()
            p0.x = float(tr.xy[0])
            p0.y = float(tr.xy[1])
            p0.z = 0.0

            p1.x = p0.x + VELOCITY_SCALE * vx
            p1.y = p0.y + VELOCITY_SCALE * vy
            p1.z = 0.0

            marker.points.append(p0)
            marker.points.append(p1)

            marker.scale.x = 0.05  # shaft diameter
            marker.scale.y = 0.10  # head diameter
            marker.scale.z = 0.15  # head length

            # Blue arrows
            marker.color.r = 0.0
            marker.color.g = 0.4
            marker.color.b = 1.0
            marker.color.a = 0.9

            marker.lifetime = rospy.Duration(0.5)
            marray.markers.append(marker)

        # ──────────────────────────────
        # 3) TTC/DCA prediction markers
        # ──────────────────────────────
        for tid, tr in self.tracks.items():
            if tr.xy is None or tr.vxy is None:
                continue

            r = np.array(tr.xy, dtype=float)
            v = tr.vxy
            if not np.isfinite(r).all() or not np.isfinite(v).all():
                continue

            t_star, d_star = self._closest_approach(r, v)
            # Only show if in front in time and within horizon
            if not (t_star >= 0.0 and t_star <= self.ttc_horizon):
                continue

            # predicted closest point
            pred = r + t_star * v

            # small sphere at predicted closest approach
            ghost = Marker()
            ghost.header.stamp = stamp
            ghost.header.frame_id = self.marker_frame
            ghost.ns = "people_ttc"
            ghost.id = tid  # same tid, diff ns
            ghost.type = Marker.SPHERE
            ghost.action = Marker.ADD

            ghost.pose.position.x = float(pred[0])
            ghost.pose.position.y = float(pred[1])
            ghost.pose.position.z = 0.0

            ghost.pose.orientation.w = 1.0

            ghost.scale.x = 0.18
            ghost.scale.y = 0.18
            ghost.scale.z = 0.18

            # Cyan-ish
            ghost.color.r = 0.0
            ghost.color.g = 1.0
            ghost.color.b = 1.0
            ghost.color.a = 0.9
            ghost.lifetime = rospy.Duration(0.5)

            marray.markers.append(ghost)

            # line from current pos to predicted closest approach
            line = Marker()
            line.header.stamp = stamp
            line.header.frame_id = self.marker_frame
            line.ns = "people_ttc_ray"
            line.id = tid
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD

            line.points = []
            p0 = geometry_msgs.msg.Point()
            p1 = geometry_msgs.msg.Point()
            p0.x = float(r[0])
            p0.y = float(r[1])
            p0.z = 0.0
            p1.x = float(pred[0])
            p1.y = float(pred[1])
            p1.z = 0.0
            line.points.append(p0)
            line.points.append(p1)

            line.scale.x = 0.03  # line width
            line.color.r = 0.0
            line.color.g = 1.0
            line.color.b = 1.0
            line.color.a = 0.7
            line.lifetime = rospy.Duration(0.5)

            marray.markers.append(line)

            # THRESHOLD CIRCLES AROUND EACH PERSON
            # ──────────────────────────────
            # Per person threshold ring around each person
            # ──────────────────────────────
            thr = float(per_person_thresholds.get(tid, self.base_threshold))
            thr_scaled = thr * 1.5;  # scale up for better visibility; the actual threshold violation is still based on the original thr

            ring = Marker()
            ring.header.stamp = stamp
            ring.header.frame_id = self.marker_frame
            ring.ns = "people_threshold_rings"
            ring.id = tid
            ring.type = Marker.LINE_STRIP
            ring.action = Marker.ADD

            ring.pose.orientation.w = 1.0
            ring.scale.x = 0.03  # line width

            # Color: red if inside thr, green otherwise (same logic as sphere)
            dist = tr.filt_dist if tr.filt_dist is not None else tr.dist
            inside = (dist is not None) and np.isfinite(dist) and (dist < thr)
            if inside:
                ring.color.r, ring.color.g, ring.color.b = 1.0, 0.0, 0.0
            else:
                ring.color.r, ring.color.g, ring.color.b = 0.0, 1.0, 0.0
            ring.color.a = 0.8

            # Build a circle in the XY plane centered at the person's position
            ring.points = []
            cx = float(tr.xy[0])
            cy = float(tr.xy[1])
            N = 40
            for k in range(N + 1):
                ang = 2.0 * math.pi * k / N
                p = geometry_msgs.msg.Point()
                p.x = cx + thr_scaled * math.cos(ang)
                p.y = cy + thr_scaled * math.sin(ang)
                p.z = 0.02
                ring.points.append(p)

            ring.lifetime = rospy.Duration(0.5)
            marray.markers.append(ring)

        # ──────────────────────────────
        # 4) FOV wedge (simple two rays)
        # ──────────────────────────────
        # 4) FOV wedge (simple two rays, aligned with actual axes)
        fov = Marker()
        fov.header.stamp = stamp
        fov.header.frame_id = self.marker_frame
        fov.ns = "fov"
        fov.id = 0
        fov.type = Marker.LINE_LIST
        fov.action = Marker.ADD

        fov.points = []

        # Interpret the plane based on observed behavior:
        #   - forward ≈ negative y
        #   - left    ≈ positive x
        # We'll build the wedge around that forward direction.
        half_angle_rad = math.radians(self.fov_angle_deg * 0.5)

        # Unit vectors in the marker_frame plane
        # Standard REP-103 base_link axes:
        #   x forward, y left, z up
        forward = np.array([1.0, 0.0], dtype=float)   # forward
        leftvec = np.array([0.0, 1.0], dtype=float)   # left


        # Normalize in case
        f_norm = forward / (np.linalg.norm(forward) + 1e-9)
        l_norm = leftvec / (np.linalg.norm(leftvec) + 1e-9)

        # Directions of left/right rays: rotate 'forward' by ±half_angle about +z
        #   dir_left  = cos(a)*f + sin(a)*l
        #   dir_right = cos(a)*f - sin(a)*l
        ca = math.cos(half_angle_rad)
        sa = math.sin(half_angle_rad)

        dir_left  = ca * f_norm + sa * l_norm
        dir_right = ca * f_norm - sa * l_norm

        origin = geometry_msgs.msg.Point()
        origin.x = 0.0
        origin.y = 0.0
        origin.z = 0.0

        left_pt = geometry_msgs.msg.Point()
        left_pt.x = float(dir_left[0] * self.fov_range)
        left_pt.y = float(dir_left[1] * self.fov_range)
        left_pt.z = 0.0

        right_pt = geometry_msgs.msg.Point()
        right_pt.x = float(dir_right[0] * self.fov_range)
        right_pt.y = float(dir_right[1] * self.fov_range)
        right_pt.z = 0.0

        # Two segments: origin->left, origin->right
        fov.points.append(origin)
        fov.points.append(left_pt)
        fov.points.append(origin)
        fov.points.append(right_pt)

        fov.scale.x = 0.03  # line width

        if alert_info.active:
            fov.color.r, fov.color.g, fov.color.b = 1.0, 0.0, 0.0
        else:
            fov.color.r, fov.color.g, fov.color.b = 0.8, 0.8, 0.2
        fov.color.a = 0.9

        fov.lifetime = rospy.Duration(0.0) # persistent
        marray.markers.append(fov)

        current_ids = set(self.tracks.keys())

        # ──────────────────────────────
        # DRAW THRESHOLD CIRCLE based on vehicle speed
        # ──────────────────────────────
        circle = Marker()
        circle.header.stamp = stamp
        circle.header.frame_id = self.marker_frame
        circle.ns = "threshold"
        circle.id = 999
        circle.type = Marker.CYLINDER
        circle.action = Marker.ADD

        circle.pose.position.x = 0.0
        circle.pose.position.y = 0.0
        circle.pose.position.z = 0.0
        circle.pose.orientation.w = 1.0

        circle.scale.x = 3.0 * dynamic_threshold
        circle.scale.y = 3.0 * dynamic_threshold
        circle.scale.z = 0.01

        circle.color.r = 1.0
        circle.color.g = 1.0
        circle.color.b = 0.0
        circle.color.a = 0.2

        circle.lifetime = rospy.Duration(0.2)

        marray.markers.append(circle)


        # Delete markers for tracks that no longer exist
        stale_ids = self._published_marker_ids - current_ids
        for tid in stale_ids:
            for ns in ["people_tracks", "people_vel", "people_ttc", "people_threshold_rings", "people_ttc_ray", "people_conf"]:
                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = self.marker_frame
                m.ns = ns
                m.id = tid
                m.action = Marker.DELETE
                marray.markers.append(m)

        self._published_marker_ids = current_ids

        # ──────────────────────────────
        # Publish all markers
        # ──────────────────────────────
        self.marker_pub.publish(marray)

    def _hazard_level_from_alert(self, alert_info: AlertInfo) -> int:
        """
        Map AlertInfo -> hazard level.
        0 CLEAR, 1 WARN, 2 SLOW, 3 STOP
        """
        # Tunable params
        warn_margin = float(rospy.get_param("~haz_warn_margin", 0.75))   # m
        slow_margin = float(rospy.get_param("~haz_slow_margin", 0.5))  # m
        stop_margin = float(rospy.get_param("~haz_stop_margin", 0.0))  # m

        stop_ttc = float(rospy.get_param("~haz_stop_ttc", 1.0))          # s
        stop_dca = float(rospy.get_param("~haz_stop_dca", 0.8))          # m

        # If no valid tracks/offender
        if alert_info is None:
            return CLEAR

        margin = float(alert_info.margin)

        # If nothing valid, treat as CLEAR (supervisor will still fail-safe on stale)
        if not np.isfinite(margin):
            return CLEAR

        offender = alert_info.offender

        # STOP if deeply inside threshold OR imminent closest approach
        if margin <= stop_margin:
            return STOP

        if offender is not None:
            t_star = float(offender.t_star)
            d_star = float(offender.d_star)
            if np.isfinite(t_star) and np.isfinite(d_star):
                if (0.0 <= t_star <= stop_ttc) and (d_star <= stop_dca):
                    return STOP

        # SLOW if inside threshold
        if margin <= slow_margin or (alert_info.active is True):
            return SLOW

        # WARN if close to threshold
        if margin <= warn_margin:
            return WARN

        return CLEAR

    def _camera_to_base_xy(self, X_cam: float, Y_cam: float, Z_cam: float,
                           stamp: rospy.Time) -> Tuple[float, float]:
        """
        Transform a 3D point from camera frame to base_link frame, return (x, y) in base_link.

        X_cam, Y_cam, Z_cam: coordinates in camera frame (meters)
        stamp: timestamp to use for TF lookup
        """
        pt_cam = PointStamped()
        pt_cam.header.stamp = stamp
        pt_cam.header.frame_id = self.camera_frame # <- now the optical frame
        pt_cam.point.x = X_cam
        pt_cam.point.y = Y_cam
        pt_cam.point.z = Z_cam

        try:
            pt_base = self.tf_buffer.transform(pt_cam, self.base_frame, timeout=rospy.Duration(0.05))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, "TF transform %s -> %s failed: %s",
                                   self.camera_frame, self.base_frame, str(e))
            return 0.0, 0.0
        # In base_link, we now expect:
        #   x ≈ forward, y ≈ left, z ≈ up
        return float(pt_base.point.x), float(pt_base.point.y)


    def _uvZ_to_xy(self, u:int, v:int, Z:float) -> Tuple[float,float]:
        """
        Back-project pixel (u,v) with depth Z (meters) to camera XY on ground plane.
        Assumes pinhole model; for a level camera you can treat (x,y) as forward/right or vice versa.
        Here: x = right, y = down-camera; you can remap later if needed.
        """
        if self.fx is None or self.fy is None or self.cx0 is None or self.cy0 is None:
            # Shouldn't happen if caminfo_ready is True, but be safe
            return 0.0, 0.0
        x = (u - self.cx0) * Z / self.fx
        y = (v - self.cy0) * Z / self.fy
        return float(x), float(y)


    def _closest_approach(self, r:np.ndarray, v:np.ndarray) -> Tuple[float,float]:
        """
        Given relative position r=[x,y] and relative velocity v=[vx,vy],
        return (t_star, d_star) where:
        t_star = time to closest approach (can be negative)
        d_star = distance at closest approach
        """
        v2 = np.dot(v, v)
        if v2 < 1e-6:
            return 0.0, float(np.linalg.norm(r))
        t_star = - float(np.dot(r, v)) / float(v2)
        d_star = float(np.linalg.norm(r + t_star * v))
        return t_star, d_star

    def _overlay_mask(self, img: np.ndarray, mask_bool: np.ndarray,
                  color_bgr: Tuple[int,int,int], alpha: float = 0.4) -> np.ndarray:
        """
        Alpha-blend a solid BGR color onto img wherever mask_bool==True.
        """
        if mask_bool is None or not mask_bool.any():
            return img
        tint = np.zeros_like(img)
        tint[:] = color_bgr
        mask3 = mask_bool.astype(bool)[..., None]  # HxWx1
        blended = (alpha * tint + (1.0 - alpha) * img).astype(img.dtype)
        return np.where(mask3, blended, img)

    def _nearest_track(self, cx: int, cy: int) -> Optional[Track]:
        if not self.tracks:
            return None
        best_id = min(self.tracks.keys(),
                    key=lambda tid: math.hypot(self.tracks[tid].cx - cx,
                                                self.tracks[tid].cy - cy))
        return self.tracks[best_id]

    def odom_callback(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)

    def _mask_centroid(self, mask_bool: np.ndarray):
        """Return (cx, cy) as ints from a boolean mask; None if empty."""
        if mask_bool is None or mask_bool.size == 0:
            return None
        ys, xs = np.where(mask_bool)
        if xs.size == 0:
            return None
        # median is robust to ragged silhouettes
        return int(np.median(xs)), int(np.median(ys))

    def _extract_people(self, result, depth_cv: np.ndarray) -> List[PersonDet]:
        """
        Masks-only extractor.

        Returns list of PersonDet:
          - cx, cy: centroid (pixels)
          - dist:   robust depth (m)
          - mask:   boolean mask (HxW)
          - X, Y:   metric camera-frame coords (m)
        """
        out: List[PersonDet] = []
        h, w = depth_cv.shape

        # Require segmentation masks
        if not (hasattr(result, "masks") and result.masks is not None and hasattr(result.masks, "data")):
            if self.require_masks:
                return out
            else:
                return out  # still no fallback by design

        # Masks as boolean arrays
        mask_stack = result.masks.data  # (N, Mh, Mw) tensor
        mask_stack = mask_stack.cpu().numpy().astype(bool)
        Mh, Mw = mask_stack.shape[-2:]
        need_resize = (Mh != h) or (Mw != w)

        boxes = result.boxes if result.boxes is not None else []
        mask_count = 0 if result.masks is None else len(result.masks.data)

        # Loop detections; use boxes only to read class labels
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            if result.names[cls_id].lower() != "person":
                continue
            if i >= mask_count:
                # no mask for this detection (can happen); skip in mask-only mode
                continue

            m = mask_stack[i]
            if need_resize:
                # nearest-neighbor to preserve mask discreteness
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

            # robust depth from mask (erode edges, IQR)
            dist = self._robust_depth_from_mask(
                depth_cv, m,
                erode_iters=self.seg_erode,
                iqr_k=1.5,
                min_valid=50,
                fallback_percentile=25
            )
            if dist is None:
                continue

            # centroid from mask
            c = self._mask_centroid(m)
            if c is None:
                continue
            cx, cy = c

            # metric XY in camera frame
            X, Y = self._uvZ_to_xy(cx, cy, dist)

            conf = float(box.conf[0]) if hasattr(box, "conf") and box.conf is not None else 1.0

            out.append(
                PersonDet(
                    cx=int(cx),
                    cy=int(cy),
                    dist=float(dist),
                    mask=m,
                    X=float(X),
                    Y=float(Y),
                    conf=conf,
                )
            )

        return out

    def _associate(self, dets: List[Tuple[int,int,float,float,float,float]], tstamp: float):
        """
        det = (cx, cy, dist, X, Y, conf). Gate and cost primarily in metric XY (meters) + range.
        Greedy nearest-neighbor association between current tracks and detections.
            Gates: pixel jump & distance jump.
        """
        track_ids = list(self.tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_dets   = set(range(len(dets)))

        ASSOC_XY_WEIGHT = 10.0
        ASSOC_DEPTH_WEIGHT = 1.0  # for dd

        def cost(tr: Track, det):
            _, _, d, X, Y, _ = det
            dXY = math.hypot(tr.xy[0] - X, tr.xy[1] - Y)
            dd  = abs(tr.dist - d)
            return ASSOC_XY_WEIGHT * dXY + ASSOC_DEPTH_WEIGHT * min(dd, 1.0)

        matches = []
        while unmatched_tracks and unmatched_dets:
            best = None; best_pair = None
            for tid in list(unmatched_tracks):
                tr = self.tracks[tid]
                for j in list(unmatched_dets):
                    cx, cy, d, X, Y, conf = dets[j]
                    # gates in metric space
                    dXY = math.hypot(tr.xy[0] - X, tr.xy[1] - Y)
                    if dXY > self.assoc_xy_gate:
                        continue
                    dd = abs(tr.dist - d)
                    if dd > self.max_dist_jump:
                        continue
                    c = cost(tr, dets[j])

                    if (best is None) or (c < best):
                        best = c; best_pair = (tid, j)
            if best_pair is None: break
            tid, j = best_pair
            matches.append(best_pair)
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(j)

        # update matched
        for tid, j in matches:
            tr = self.tracks[tid]
            cx, cy, d, X, Y, conf = dets[j]
            tr.update(cx, cy, d, tstamp, alpha=self.filter_alpha, xy=(X, Y), conf=conf)

        # new tracks
        for j in list(unmatched_dets):
            cx, cy, d, X, Y, conf = dets[j]
            tid = self.next_tid
            self.next_tid += 1
            self.tracks[tid] = Track(tid, cx, cy, d, tstamp, xy=(X, Y), conf=conf)

        # age out unmatched
        to_delete = []
        for tid in list(unmatched_tracks):
            tr = self.tracks[tid]
            tr.misses += 1
            if tr.misses > self.max_misses:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

    # --------- Small helpers for structure ---------

    def _should_process_frame(self) -> bool:
        """Frame skipping + readiness checks."""
        self._frame_count += 1
        if self.frame_skip > 0 and (self._frame_count % (self.frame_skip + 1)) != 0:
            return False

        if not self.caminfo_ready:
            rospy.logwarn_throttle(5.0, "Waiting for CameraInfo (fx,fy,cx,cy)...")
            return False

        if not hasattr(self, "model") or self.model is None:
            rospy.logwarn_throttle(5.0, "YOLO model not ready yet")
            return False

        return True

    def _compute_dynamic_threshold(self) -> float:
        """Vehicle-speed-based base threshold, clamped by max_threshold."""
        dynamic_threshold = self.base_threshold + self.speed_gain * self.current_speed
        return min(dynamic_threshold, self.max_threshold)

    def _prepare_images(self, rgb_msg, depth_msg):
        """
        Convert incoming ROS images to:
          - vis: BGR8 uint8 for YOLO + debugging
          - depth_cv: 32FC1 depth (m)
        Returns (vis, depth_cv) or (None, None) on error.
        """
        try:
            rgb_raw = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr_throttle(2.0, "cv_bridge passthrough failed: %s", str(e))
            return None, None

        # Normalize to 3-channel uint8 BGR (moved from synced_callback)
        if rgb_raw.ndim == 2:
            # mono8 / mono16
            if rgb_msg.encoding in ("mono16", "16UC1", "16SC1"):
                vis = np.clip(rgb_raw / 256.0, 0, 255).astype(np.uint8)
            else:
                vis = rgb_raw.astype(np.uint8, copy=False)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        elif rgb_raw.ndim == 3:
            h_, w_, ch = rgb_raw.shape
            if ch == 3:
                if rgb_msg.encoding.lower().startswith("rgb"):
                    vis = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
                else:
                    vis = rgb_raw
            elif ch == 4:
                if rgb_msg.encoding.lower().startswith("rgba"):
                    vis = cv2.cvtColor(rgb_raw, cv2.COLOR_RGBA2BGR)
                else:
                    vis = cv2.cvtColor(rgb_raw, cv2.COLOR_BGRA2BGR)
            else:
                rospy.logwarn_throttle(2.0, "Unexpected channel count=%d; using first 3", ch)
                vis = rgb_raw[..., :3]
        else:
            rospy.logwarn_throttle(2.0, "Unexpected image shape: %r", rgb_raw.shape)
            return None, None

        if vis.dtype != np.uint8:
            vis = np.clip(vis, 0, 255).astype(np.uint8)

        depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        return vis, depth_cv

    def _make_safe_alert(self) -> AlertInfo:
        # No offender, margin = +inf, active = False
        return AlertInfo(active=False, margin=float('inf'), offender=None)

    def _handle_no_people(self, vis, dynamic_threshold: float, rgb_msg):
        """
        Called when YOLO finds no valid person detections.
        Publishes: alert=False, empty arrays, aged tracks, and a debug frame.
        """
        rospy.loginfo_throttle(
            2.0,
            "YOLO: no people; veh=%.2f m/s thr=%.2f m",
            self.current_speed, dynamic_threshold
        )

        # Reset alert state
        self.alert_active = False
        self.alert_pub.publish(False)

        # Publish a finite, positive margin instead of inf
        safe_margin = float(dynamic_threshold)
        self.hazard_margin_pub.publish(Float32(data=safe_margin))

        # No offender → publish NaNs for TTC/DCA
        self.hazard_ttc_pub.publish(Float32(data=float("nan")))
        self.hazard_dca_pub.publish(Float32(data=float("nan")))

        # Explicitly publish CLEAR
        self.hazard_level_pub.publish(UInt8(data=0))  # CLEAR

        # Age out existing tracks
        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]
            tr.misses += 1
            if tr.misses > self.max_misses:
                del self.tracks[tid]

        # Publish empties
        self.ids_pub.publish(Int32MultiArray(data=[]))
        self.dists_pub.publish(Float32MultiArray(data=[]))
        self.speeds_pub.publish(Float32MultiArray(data=[]))

        # Debug frame (no detections)
        if self.publish_debug:
            vis_out = vis.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            y = 22
            cv2.putText(vis_out, f"veh_speed: {self.current_speed:.2f} m/s", (10, y),
                        font, 0.6, (255, 255, 255), 2, cv2.LINE_AA); y += 22
            cv2.putText(vis_out, f"threshold: {dynamic_threshold:.2f} m", (10, y),
                        font, 0.6, (255, 255, 255), 2, cv2.LINE_AA); y += 22
            cv2.putText(vis_out, "no detections", (10, y),
                        font, 0.6, (100, 200, 255), 2, cv2.LINE_AA)

            dbg = self.bridge.cv2_to_imgmsg(vis_out, encoding="bgr8")
            dbg.header.stamp = rgb_msg.header.stamp
            dbg.header.frame_id = rgb_msg.header.frame_id or "camera"
            self.debug_img_pub.publish(dbg)

        # --- Publish FOV (and any remaining tracks) even with no people ---
        if self.marker_pub is not None:
            per_person_thresholds = {}  # nothing special per track
            alert_info = self._make_safe_alert()
            self._publish_track_markers(
                per_person_thresholds,
                alert_info,
                rgb_msg.header.stamp,
                dynamic_threshold
            )

    def _update_tracks_from_people(self, people: List[PersonDet], tstamp: float):
        """Build metric dets from PersonDet list and run association in base_link frame."""
        dets = []

        # Convert each detection's camera coords -> base_link coords
        stamp = rospy.Time.from_sec(tstamp)  # same as rgb_msg.header.stamp but float->Time

        for p in people:
            # We know p.dist is roughly the range; we treat camera projection as:
            #   (X_cam, Y_cam, Z_cam) with Z_cam ≈ p.dist (forward)
            X_cam = p.X
            Y_cam = p.Y
            Z_cam = p.dist

            X_base, Y_base = self._camera_to_base_xy(X_cam, Y_cam, Z_cam, stamp)

            dets.append((p.cx, p.cy, p.dist, X_base, Y_base, p.conf))

        self._associate(dets, tstamp)


    def _compute_per_person_thresholds(self, dynamic_threshold: float) -> Dict[int, float]:
        """
        Compute per-track thresholds, inflating base dynamic threshold by:
          - approaching radial speed
          - lateral path-crossing risk in base_link XY plane within TTC horizon.
        """
        per_person_thresholds: Dict[int, float] = {}

        for tid, tr in self.tracks.items():
            thr = dynamic_threshold

            # 1) Approach inflation (radial toward the scooter)
            if tr.radial is not None and np.isfinite(tr.radial) and tr.radial > 0.0:
                thr = min(thr + self.person_speed_gain * tr.radial, self.max_threshold)

            # 2) Crossing risk (direct lateral-crossing test in base_link)
            # base_link convention: x forward, y left. A "path crossing" occurs if y(t) crosses 0 soon
            # while the person is in front (x(t)>0) and close enough ahead (x(t) <= cross_x_max).
            r = np.array(tr.xy, dtype=float) if tr.xy is not None else None
            v = tr.vxy

            if r is not None and v is not None and np.isfinite(r).all() and np.isfinite(v).all():
                x0 = float(r[0])
                y0 = float(r[1])
                vx = float(v[0])
                vy = float(v[1])

                crossing = False
                t_cross = float("nan")
                x_cross = float("nan")

                if abs(y0) >= float(self.cross_y_min) and abs(vy) > 1e-3:
                    t_cross = -y0 / vy
                    if 0.0 <= t_cross <= float(self.ttc_horizon):
                        x_cross = x0 + vx * t_cross
                        # Only care if this crossing happens in front, and not too far ahead.
                        if (x_cross > 0.0) and (x_cross <= float(self.cross_x_max)) and (abs(vy) >= float(self.min_cross_speed)):
                            crossing = True

                rospy.loginfo_throttle(
                    0.5,
                    "[cross_lat] ID %d: dist=%.2f thr=%.2f y0=%.2f vy=%.2f t_cross=%s x_cross=%s crossing=%s",
                    tid,
                    float(tr.filt_dist) if tr.filt_dist is not None else float(tr.dist),
                    thr,
                    y0,
                    vy,
                    ("NA" if not np.isfinite(t_cross) else f"{t_cross:.2f}"),
                    ("NA" if not np.isfinite(x_cross) else f"{x_cross:.2f}"),
                    str(crossing),
                )

                if crossing:
                    thr = min(thr + self.perp_gain * abs(vy), self.max_threshold)

            per_person_thresholds[tid] = thr

        return per_person_thresholds

    
    def _log_tracks(self, per_person_thresholds):
        """Throttled per-track logging of distance, threshold, radial, misses."""
        for tid, tr in sorted(self.tracks.items()):
            rs = "NA" if tr.radial is None or not np.isfinite(tr.radial) else f"{tr.radial:.2f}"
            thr = per_person_thresholds.get(tid, self.base_threshold)

            rospy.loginfo_throttle(
                0.5,
                "[ID %d] dist=%.2f m  thr=%.2f m  radial=%s m/s  misses=%d",
                tid, tr.filt_dist, thr, rs, tr.misses
            )

    def _publish_debug_image(self, vis, rgb_msg, dynamic_threshold: float,
                             people: List[PersonDet], per_person_thresholds: Dict[int, float]):
        """Draw masks, per-person thresholds, and HUD."""
        if not self.publish_debug:
            return

        vis_out = vis.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_dyn_thr = dynamic_threshold

        for p in people:
            mask = p.mask.astype(bool)
            cx, cy = p.cx, p.cy
            d      = p.dist

            tr = self._nearest_track(cx, cy)
            per_thr = per_person_thresholds.get(tr.id, dynamic_threshold) if tr is not None else dynamic_threshold

            # red if inside threshold, else green
            color = (0, 0, 255) if d < per_thr else (0, 255, 0)
            vis_out = self._overlay_mask(vis_out, mask, color, alpha=0.45)

            cv2.circle(vis_out, (cx, cy), 4, (255, 255, 255), -1)
            rs = "NA" if tr is None or tr.radial is None or not np.isfinite(tr.radial) else f"{tr.radial:.2f}"
            label = f"{d:.2f} m  thr={per_thr:.2f}  v_r={rs}"
            org = (max(0, cx - 40), max(0, cy - 12))
            cv2.putText(vis_out, label, org, font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis_out, label, org, font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # HUD unchanged...
        y = 22
        cv2.putText(vis_out, f"veh_speed: {self.current_speed:.2f} m/s", (10, y),
                    font, 0.6, (255, 255, 255), 2, cv2.LINE_AA); y += 22
        cv2.putText(vis_out, f"base_thr: {base_dyn_thr:.2f} m", (10, y),
                    font, 0.6, (255, 255, 255), 2, cv2.LINE_AA); y += 22
        cv2.putText(vis_out, "Legend: RED=inside threshold, GREEN=safe",
                    (10, y), font, 0.6, (180, 255, 180), 2, cv2.LINE_AA)

        dbg = self.bridge.cv2_to_imgmsg(vis_out, encoding="bgr8")
        dbg.header.stamp = rgb_msg.header.stamp
        dbg.header.frame_id = rgb_msg.header.frame_id or "zed_left"
        self.debug_img_pub.publish(dbg)

    def _compute_alert_and_log(
        self,
        dynamic_threshold: float,
        per_person_thresholds: Dict[int, float],
    ) -> AlertInfo:
        """
        Decide global alert from per-person thresholds and log the result.

        margin is now:  min(dist - thr) across all *eligible* tracks
        - negative => someone is inside threshold (violation)
        - positive => closest person is still outside threshold (how much buffer remains)
        - +inf     => no eligible tracks this frame
        """
        best_margin = float('inf')
        best_offender: Optional[OffenderInfo] = None

        for tid, tr in self.tracks.items():
            thr = per_person_thresholds.get(tid, dynamic_threshold)

            # Basic validity
            if tr.filt_dist is None or not np.isfinite(tr.filt_dist):
                continue

            # Maturity / stability filters (keep your intent)
            if tr.age < self.min_alert_age:
                continue
            if tr.misses > 0:
                continue

            # margin is defined for ALL tracks now
            margin = float(tr.filt_dist - thr)

            # compute closest-approach metrics for the offender record
            r = np.array(tr.xy, dtype=float) if tr.xy is not None else np.zeros(2, dtype=float)
            v = tr.vxy if tr.vxy is not None else np.zeros(2, dtype=float)
            t_star, d_star = self._closest_approach(r, v)
            vnorm = float(np.linalg.norm(v)) if v is not None else 0.0

            if margin < best_margin:
                best_margin = margin
                best_offender = OffenderInfo(
                    tid=tid,
                    dist=float(tr.filt_dist),
                    threshold=float(thr),
                    vrad=(None if tr.radial is None else float(tr.radial)),
                    t_star=float(t_star),
                    d_star=float(d_star),
                    vnorm=vnorm,
                )

        # active if best_margin is a real number and negative
        active = (best_offender is not None) and np.isfinite(best_margin) and (best_margin < 0.0)

        # publish alert state (backwards compatible)
        self.alert_active = active
        self.alert_pub.publish(self.alert_active)

        if best_offender is not None and np.isfinite(best_margin):
            o = best_offender
            if active:
                rospy.logwarn(
                    "ALERT: TID=%d dist=%.2f thr=%.2f margin=%.2f | veh=%.2f m/s | "
                    "v_r=%s | t*=%+.2f d*=%.2f | |v_xy|=%.2f",
                    o.tid, o.dist, o.threshold, best_margin, self.current_speed,
                    ("NA" if o.vrad is None or not np.isfinite(o.vrad) else f"{o.vrad:.2f}"),
                    o.t_star, o.d_star, o.vnorm
                )
            else:
                rospy.loginfo_throttle(
                    1.0,
                    "SAFE: closest margin=%.2f m (TID=%d dist=%.2f thr=%.2f) veh=%.2f m/s",
                    best_margin, o.tid, o.dist, o.threshold, self.current_speed
                )
        else:
            rospy.loginfo_throttle(1.0, "SAFE: no eligible tracks (margin=inf) veh=%.2f m/s", self.current_speed)

        return AlertInfo(active=active, margin=best_margin, offender=best_offender)


    # ---------------- Main callback ----------------
    def synced_callback(self, rgb_msg, depth_msg):

        rospy.loginfo_once("RGB in:  encoding=%s", rgb_msg.encoding)
        rospy.loginfo_once("Depth in: encoding=%s", depth_msg.encoding)

        # 1) Basic readiness + frame skipping
        if not self._should_process_frame():
            return

        # 2) Dynamic threshold from vehicle speed
        dynamic_threshold = self._compute_dynamic_threshold()
        self.threshold_pub.publish(Float32(data=float(dynamic_threshold)))

        # 3) Prepare images (BGR8 + depth)
        vis, depth_cv = self._prepare_images(rgb_msg, depth_msg)
        if vis is None or depth_cv is None:
            return

        # 4) YOLO inference
        results = self.model(
            vis,
            imgsz=self.imgsz,
            conf=self.conf_th,
            iou=self.iou_th,
            verbose=False
        )
        result = results[0]

        # 5) Extract people with depth from masks
        people = self._extract_people(result, depth_cv)
        now = rgb_msg.header.stamp.to_sec()

        # 6) No people: reset state + debug
        if not people:
            self._handle_no_people(vis, dynamic_threshold, rgb_msg)
            return

        # 7) Update multi-person tracks from current detections
        self._update_tracks_from_people(people, now)

        # 8) Build and publish arrays (IDs, distances, radial speeds)
        ids, dists, speeds, confidences = [], [], [], []
        closest_dist = None
        best_approach = 0.0

        for tid, tr in self.tracks.items():
            ids.append(tid)
            dists.append(float(tr.filt_dist))
            spd = float("nan") if tr.radial is None else float(tr.radial)
            speeds.append(spd)
            confidences.append(float(getattr(tr, "conf", 1.0)))
            if tr.radial is not None and tr.radial > best_approach:
                best_approach = tr.radial
            if closest_dist is None or tr.filt_dist < closest_dist:
                closest_dist = tr.filt_dist

        self.ids_pub.publish(Int32MultiArray(data=ids))
        self.dists_pub.publish(Float32MultiArray(data=dists))
        self.speeds_pub.publish(Float32MultiArray(data=speeds))
        self.confidence_pub.publish(Float32MultiArray(data=confidences))

        # Still publish closest distance for compatibility
        if closest_dist is not None and np.isfinite(closest_dist):
            self.closest_dist_pub.publish(float(closest_dist))

        # 9) If no valid closest distance, publish debug and bail
        if closest_dist is None or not np.isfinite(closest_dist):
            self.alert_active = False
            self.alert_pub.publish(False)

            if self.publish_debug:
                vis_out = vis.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                y = 22
                cv2.putText(vis_out, f"veh_speed: {self.current_speed:.2f} m/s", (10, y),
                            font, 0.6, (255, 255, 255), 2, cv2.LINE_AA); y += 22
                cv2.putText(vis_out, f"threshold: {dynamic_threshold:.2f} m", (10, y),
                            font, 0.6, (255, 255, 255), 2, cv2.LINE_AA); y += 22
                cv2.putText(vis_out, "no valid closest distance", (10, y),
                            font, 0.6, (100, 200, 255), 2, cv2.LINE_AA)

                dbg = self.bridge.cv2_to_imgmsg(vis_out, encoding="bgr8")
                dbg.header.stamp = rgb_msg.header.stamp
                dbg.header.frame_id = rgb_msg.header.frame_id or "camera"
                self.debug_img_pub.publish(dbg)

            return

        # 10) Per-person thresholds (vehicle + radial + crossing risk)
        per_person_thresholds = self._compute_per_person_thresholds(dynamic_threshold)

        # 11) Per-track logging (log each person's threshold too)
        self._log_tracks(per_person_thresholds)

        # 12) Debug overlay (masks tinted by threshold crossing)
        self._publish_debug_image(vis, rgb_msg, dynamic_threshold,
                                  people, per_person_thresholds)

        # 13) Global alert decision + logging
        alert_info = self._compute_alert_and_log(dynamic_threshold, per_person_thresholds)
        # alert_info.active, alert_info.margin, alert_info.offender are now available

        # ---- Publish hazard outputs for supervisor node ----
        level = self._hazard_level_from_alert(alert_info)

        self.hazard_level_pub.publish(UInt8(data=int(level)))
        margin = alert_info.margin
        if not np.isfinite(margin):
            # Tracks exist but none eligible → treat as safely outside threshold
            margin = float(dynamic_threshold)

        self.hazard_margin_pub.publish(Float32(data=margin))

        if alert_info.offender is not None:
            self.hazard_ttc_pub.publish(Float32(data=float(alert_info.offender.t_star)))
            self.hazard_dca_pub.publish(Float32(data=float(alert_info.offender.d_star)))
        else:
            # publish NaNs or big defaults; NaN is fine for Float32
            self.hazard_ttc_pub.publish(Float32(data=float("nan")))
            self.hazard_dca_pub.publish(Float32(data=float("nan")))

        # 14) RViz markers for tracks in marker_frame
        self._publish_track_markers(per_person_thresholds, alert_info, rgb_msg.header.stamp, dynamic_threshold)


def main():
    rospy.init_node("yolo_people_proximity")
    node = YoloPeopleProximityNode()
    rospy.spin()

if __name__ == "__main__":
    main()
