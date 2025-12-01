#!/usr/bin/env python3

# Tuning knobs you'll most often adjust:
#   ~person_distance_threshold
#   ~speed_gain
#   ~person_speed_gain
#   ~perp_gain
#   ~max_dist_jump — meters allowed between frames for same person; tighten if IDs switch when people cross.
#   ~max_misses — how many frames a person can disappear (occlusion) before we drop the track.
#   ~filter_alpha — higher = more responsive, lower = smoother speeds. 
#   ~ttc_horizon, ~dca_limit — crossing risk parameters. Nudge ttc_horizon up (e.g. 2.0 → 3.0 s) and/or dca_limit up (0.8 → 1.0 m) to catch more crossers.

#improvements: right now everything is relative to the escooter, not to the world. also, perpendicular detection works on the basis that the scooter is always moving exactly forward, which isn't always the case. 
"""
3.1 World-frame / base_link-frame tracking
Right now:
tr.xy is in camera frame.
Risk (closest approach, crossing) is computed in that camera XY plane.
current_speed is just the magnitude of odom linear velocity.
Future upgrade:
Use TF to get a transform from camera → base_link and from base_link → world (e.g. odom or map).
Convert person positions from camera frame to base_link frame:
Let camera frame ≈ [X: right, Y: down, Z: forward] (your comment).
In base_link, you want [x: forward, y: left, z: up].

3.2 True relative velocity (person vs scooter), not just mask motion
Right now tr.vxy is based on person motion in camera XY only. But the scooter is also moving, and potentially turning.
True relative velocity in base_link:
v_rel = v_person_world - v_scooter_world
Simple future pipeline:
Approximate person world pose over time by:
At each frame: get person XY in base_link as above.
Transform base_link → world (odom/map) and store person’s world XY.
Scooter world velocity: you already have Odometry – you can use its twist in odom frame.
Compute v_person_world as finite difference of world XY positions.
Compute v_rel = v_person_world - v_scooter_world.
Then:
Use r_rel and v_rel in _closest_approach.
Use longitudinal component along vehicle heading to determine “approach” vs “moving away”.
Use lateral component for crossing risk.
This would remove your current assumption:
perpendicular detection works on the basis that the scooter is always moving exactly forward
Instead, it would be aligned with the actual velocity vector/orientation from odom.
This is a bigger refactor, but your current structure (Track, PersonDet, AlertInfo) is already in a good shape to support it.

3.3 Dynamic model-based TTC (vs just distance threshold)
Right now you do:
Distance < threshold → alert.
Future variant:
Use a “braking model” TTC:
t_brake = v / a_max  (assuming constant decel)
d_brake = v * t_brake - 0.5 * a_max * t_brake^2
Or more simply: d_safe = v * t_reaction + (v^2)/(2 a_max).
You can then say:
If predicted closest distance d_star < d_safe, raise an alert, even if current distance is still large.
This makes behaviour more interpretable in terms of “can I stop in time” rather than “is person within X meters”.
"""

import cv2

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

DETECTION_DISTANCE = 0.9  # m
VEHICLE_SPEED_GAIN = 0.5   # m per m/s
PERSON_SPEED_GAIN  = 0.5   # m per m/s of person radial speed
FRAME_SKIP = 2             # process every (N+1)th frame
MAX_DISTANCE_JUMP = 1.75   # m (inter-frame distance gate)
MAX_POSSIBLE_THRESHOLD_DISTANCE = 5.0  # m

# ---------------- Single-person detection structure ----------------
@dataclass
class PersonDet:
    """
    Single person detection from YOLO+depth.

    - (cx, cy): mask centroid in pixel coordinates
    - dist:     robust range from depth (m)
    - mask:     boolean mask in image space (HxW)
    - X, Y:     metric coordinates in camera frame (m)
    """
    cx: int
    cy: int
    dist: float
    mask: np.ndarray  # bool HxW
    X: float
    Y: float

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
                 "radial","misses","xy","prev_xy","vxy")

    def __init__(self, tid:int, cx:int, cy:int, dist:float, tstamp:float, xy:Tuple[float,float]):
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

    def update(self, cx:int, cy:int, dist:float, tstamp:float, alpha:float,
               xy:Tuple[float,float], dt_lo:float=0.02, dt_hi:float=0.5):
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

        # --- camera intrinsics (px) ---
        self.fx = self.fy = self.cx0 = self.cy0 = None
        self.caminfo_ready = False
        cam_info_topic = rospy.get_param("~camera_info_topic", "/zed_node/left/camera_info")
        self.caminfo_sub = rospy.Subscriber(cam_info_topic, CameraInfo, self._caminfo_cb, queue_size=1)

        # horizon / gains for “crossing” risk
        self.ttc_horizon = float(rospy.get_param("~ttc_horizon", 2.5))   # seconds
        self.dca_limit   = float(rospy.get_param("~dca_limit",   0.8))   # m, distance at closest approach
        self.perp_gain   = float(rospy.get_param("~perp_gain",   0.6))   # m of extra buffer per m/s of perpendicular rel speed

        # gates / smoothing
        self.max_dist_jump  = rospy.get_param("~max_dist_jump", MAX_DISTANCE_JUMP)  # m (for both single & multi)
        self.max_misses     = rospy.get_param("~max_misses", 5)                    # drop track after N misses
        self.filter_alpha   = rospy.get_param("~filter_alpha", 0.3)                # EMA for distances
        self.max_threshold  = rospy.get_param("~max_threshold", MAX_POSSIBLE_THRESHOLD_DISTANCE)

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

        self.publish_debug = rospy.get_param("~publish_debug", True)
        self.debug_img_pub = rospy.Publisher("~debug_image", Image, queue_size=1)

        self.require_masks = rospy.get_param("~require_masks", True)

        # NEW: multi-person outputs
        self.ids_pub   = rospy.Publisher("~people_ids", Int32MultiArray, queue_size=1)
        self.dists_pub = rospy.Publisher("~people_distances", Float32MultiArray, queue_size=1)
        self.speeds_pub= rospy.Publisher("~people_radial_speeds", Float32MultiArray, queue_size=1)

        # ---------------- Subscribers w/ sync ----------------
        rospy.Subscriber(self.speed_topic, Odometry, self.odom_callback)
        rgb_sub   = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=5, slop=0.05)
        self.ts.registerCallback(self.synced_callback)

        # NEW: tracking state
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

            out.append(
                PersonDet(
                    cx=int(cx),
                    cy=int(cy),
                    dist=float(dist),
                    mask=m,
                    X=float(X),
                    Y=float(Y),
                )
            )

        return out


    def _associate(self, dets: List[Tuple[int,int,float,float,float]], tstamp: float):
        """
        det = (cx, cy, dist, X, Y). Gate and cost primarily in metric XY (meters) + range.
        Greedy nearest-neighbor association between current tracks and detections.
            Gates: pixel jump & distance jump.
        """
        track_ids = list(self.tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_dets   = set(range(len(dets)))

        ASSOC_XY_WEIGHT = 10.0
        ASSOC_DEPTH_WEIGHT = 1.0  # for dd

        def cost(tr: Track, det):
            _, _, d, X, Y = det
            dXY = math.hypot(tr.xy[0] - X, tr.xy[1] - Y)
            dd  = abs(tr.dist - d)
            return ASSOC_XY_WEIGHT * dXY + ASSOC_DEPTH_WEIGHT * min(dd, 1.0)

        matches = []
        while unmatched_tracks and unmatched_dets:
            best = None; best_pair = None
            for tid in list(unmatched_tracks):
                tr = self.tracks[tid]
                for j in list(unmatched_dets):
                    cx, cy, d, X, Y = dets[j]
                    # gates
                    if math.hypot(tr.xy[0] - X, tr.xy[1] - Y) > 1.4:  # meters; tune
                        continue
                    if abs(tr.dist - d) > self.max_dist_jump:
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
            cx, cy, d, X, Y = dets[j]
            tr.update(cx, cy, d, tstamp, alpha=self.filter_alpha, xy=(X, Y))

        # new tracks
        for j in list(unmatched_dets):
            cx, cy, d, X, Y = dets[j]
            tid = self.next_tid
            self.next_tid += 1
            self.tracks[tid] = Track(tid, cx, cy, d, tstamp, xy=(X, Y))

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

    def _update_tracks_from_people(self, people: List[PersonDet], tstamp: float):
        """Build metric dets from PersonDet list and run association."""
        dets = [(p.cx, p.cy, p.dist, p.X, p.Y) for p in people]
        self._associate(dets, tstamp)


    def _compute_per_person_thresholds(self, dynamic_threshold: float) -> Dict[int, float]:
        """
        Compute per-track thresholds, inflating base dynamic threshold by:
          - approaching radial speed
          - crossing risk in camera XY plane within TTC horizon and DCA limit.
        """
        per_person_thresholds: Dict[int, float] = {}
        for tid, tr in self.tracks.items():
            thr = dynamic_threshold

            # original approach inflation
            if tr.radial is not None and tr.radial > 0.0:
                thr = min(thr + self.person_speed_gain * tr.radial, self.max_threshold)

            # crossing risk in camera XY
            r = np.array(tr.xy, dtype=float) if tr.xy is not None else np.zeros(2, dtype=float)
            v = tr.vxy
            t_star, d_star = self._closest_approach(r, v)

            crossing = (
                (t_star >= 0.0) and (t_star <= self.ttc_horizon) and
                (d_star < self.dca_limit) and
                (tr.radial is not None and tr.radial > 0.0)
            )
            if crossing:
                thr = min(thr + self.perp_gain * float(np.linalg.norm(v)), self.max_threshold)

            per_person_thresholds[tid] = thr

        return per_person_thresholds

    def _log_tracks(self):
        """Throttled per-track logging of distance/radial/misses."""
        for tid, tr in sorted(self.tracks.items()):
            rs = "NA" if tr.radial is None or not np.isfinite(tr.radial) else f"{tr.radial:.2f}"
            rospy.loginfo_throttle(
                0.5,
                "[ID %d] dist=%.2f m  radial=%s m/s  misses=%d",
                tid, tr.filt_dist, rs, tr.misses
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

        Returns:
            AlertInfo with:
              - active: whether any person is inside its threshold
              - margin: min(dist - thr) across all tracks (neg = violation)
              - offender: details of the worst offender, if any
        """
        active = False
        best_margin = float('inf')
        best_offender: Optional[OffenderInfo] = None

        for tid, tr in self.tracks.items():
            thr = per_person_thresholds.get(tid, dynamic_threshold)
            dist_ok = (tr.filt_dist is not None) and np.isfinite(tr.filt_dist)
            if not dist_ok:
                continue

            # relative position/velocity in camera XY
            r = tr.xy if tr.xy is not None else np.zeros(2, dtype=float)
            v = tr.vxy
            t_star, d_star = self._closest_approach(r, v)
            vnorm = float(np.linalg.norm(v))

            if tr.filt_dist < thr:
                active = True
                margin = tr.filt_dist - thr  # negative = violation
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

        # publish alert state
        self.alert_active = active
        self.alert_pub.publish(self.alert_active)

        if self.alert_active and best_offender is not None:
            o = best_offender
            rospy.logwarn(
                "ALERT: TID=%d dist=%.2f m < thr=%.2f m | veh=%.2f m/s | v_r=%.2f m/s "
                "| crossing: t*=%+.2f s d*==%.2f m | |v_xy|=%.2f m/s",
                o.tid,
                o.dist,
                o.threshold,
                self.current_speed,
                (float('nan') if o.vrad is None else o.vrad),
                o.t_star,
                o.d_star,
                o.vnorm,
            )
        else:
            # SAFE or no valid offender; best_margin is +inf if no valid tracks
            rospy.loginfo_throttle(
                1.0,
                "SAFE: veh=%.2f m/s | min(dist - thr)=%.2f m",
                self.current_speed,
                best_margin,
            )

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
        ids, dists, speeds = [], [], []
        closest_dist = None
        best_approach = 0.0

        for tid, tr in self.tracks.items():
            ids.append(tid)
            dists.append(float(tr.filt_dist))
            spd = float("nan") if tr.radial is None else float(tr.radial)
            speeds.append(spd)
            if tr.radial is not None and tr.radial > best_approach:
                best_approach = tr.radial
            if closest_dist is None or tr.filt_dist < closest_dist:
                closest_dist = tr.filt_dist

        self.ids_pub.publish(Int32MultiArray(data=ids))
        self.dists_pub.publish(Float32MultiArray(data=dists))
        self.speeds_pub.publish(Float32MultiArray(data=speeds))

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

        # 11) Per-track logging
        self._log_tracks()

        # 12) Debug overlay (masks tinted by threshold crossing)
        self._publish_debug_image(vis, rgb_msg, dynamic_threshold,
                                  people, per_person_thresholds)

        # 13) Global alert decision + logging
        alert_info = self._compute_alert_and_log(dynamic_threshold, per_person_thresholds)
        # alert_info.active, alert_info.margin, alert_info.offender are now available

def main():
    rospy.init_node("yolo_people_proximity")
    node = YoloPeopleProximityNode()
    rospy.spin()

if __name__ == "__main__":
    main()