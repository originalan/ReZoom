#!/usr/bin/env python3
"""
cmd_vel_safety_supervisor.py (ROS1)

Subscribes:
  - /cmd_vel                     (geometry_msgs/Twist)  desired motion
  - /yolo_people_proximity/hazard_level (std_msgs/UInt8) optional; 0=CLEAR 1=WARN 2=SLOW 3=STOP
  - /yolo_people_proximity/people_proximity_alert (std_msgs/Bool) optional fallback if hazard_level not provided
  - /yolo_people_proximity/hazard_margin (std_msgs/Float32) optional; min(dist-thr), negative = inside threshold
  - /yolo_people_proximity/hazard_ttc   (std_msgs/Float32) optional; offender t* (s)
  - /yolo_people_proximity/hazard_dca   (std_msgs/Float32) optional; offender d* (m)

Publishes:
  - /cmd_vel_safety              (geometry_msgs/Twist) safe motion

Behavior:
  - CLEAR/WARN: pass-through (WARN logs)
  - SLOW: scales linear & angular
  - STOP: outputs zero, holds STOP for stop_hold_s
  - If cmd_vel stale or hazard stale: STOP (fail-safe)
  - Optional hysteresis: require N consecutive CLEAR frames to fully release after STOP/SLOW


  SOMETHING TO NOTE: This node affects both LINEAR and STEERING movement. Could possibly remove slowed steering? Need to test
"""

import math
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt8, Bool, Float32

CLEAR, WARN, SLOW, STOP = 0, 1, 2, 3


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class CmdVelSafetySupervisor:
    def __init__(self):
        # ---------------- Topics ----------------
        self.cmd_in_topic = rospy.get_param("~cmd_in", "/cmd_vel")
        self.cmd_out_topic = rospy.get_param("~cmd_out", "/cmd_vel_safety")

        # Preferred input: hazard_level (UInt8)
        self.hazard_level_topic = rospy.get_param(
            "~hazard_level_topic", "/yolo_people_proximity/hazard_level"
        )

        # Optional fallbacks / debug inputs
        self.alert_bool_topic = rospy.get_param(
            "~alert_bool_topic", "/yolo_people_proximity/people_proximity_alert"
        )
        self.margin_topic = rospy.get_param(
            "~margin_topic", "/yolo_people_proximity/hazard_margin"
        )
        self.ttc_topic = rospy.get_param(
            "~ttc_topic", "/yolo_people_proximity/hazard_ttc"
        )
        self.dca_topic = rospy.get_param(
            "~dca_topic", "/yolo_people_proximity/hazard_dca"
        )

        self.fail_open_if_no_hazard = bool(
            rospy.get_param("~fail_open_if_no_hazard", False)
        )

        # ---------------- Timing ----------------
        self.rate_hz = float(rospy.get_param("~rate_hz", 30.0))
        self.cmd_timeout = float(rospy.get_param("~cmd_timeout", 0.25))
        self.hazard_timeout = float(rospy.get_param("~hazard_timeout", 0.25))

        # ---------------- Limits ----------------
        self.max_linear = float(rospy.get_param("~max_linear", 10.0))       # m/s
        self.max_angular = float(rospy.get_param("~max_angular", 10.0))     # rad/s

        # ---------------- SLOW/STOP tuning ----------------
        self.slow_scale = float(rospy.get_param("~slow_scale", 0.5))
        self.warn_scale = float(rospy.get_param("~warn_scale", 0.75))  # optional gentle slow in WARN
        self.use_warn_scale = bool(rospy.get_param("~use_warn_scale", False))

        self.stop_hold_s = float(rospy.get_param("~stop_hold_s", 0.5))  # how long to hold STOP after triggered (even if hazard clears)
        self.clear_frames_to_release = int(rospy.get_param("~clear_frames_to_release", 5))

        # If hazard_level is not published, you can infer from margin/ttc/dca (optional)
        self.enable_infer_from_metrics = bool(rospy.get_param("~enable_infer_from_metrics", False))

        # Inference thresholds (only used if enable_infer_from_metrics=True)
        self.warn_margin = float(rospy.get_param("~warn_margin", 0.75))   # meters
        self.slow_margin = float(rospy.get_param("~slow_margin", 0.5))  # meters
        self.stop_margin = float(rospy.get_param("~stop_margin", 0.0))  # meters
        self.stop_ttc = float(rospy.get_param("~stop_ttc", 1.0))          # seconds
        self.stop_dca = float(rospy.get_param("~stop_dca", 0.8))          # meters

        # ---------------- State ----------------
        self.latest_cmd = Twist()
        self.latest_cmd_stamp = None

        self.hazard_level = CLEAR
        self.hazard_stamp = None

        self.alert_bool = False
        self.alert_stamp = None

        self.margin = float("inf")
        self.ttc = float("nan")
        self.dca = float("nan")
        self.metrics_stamp = None

        self.stop_until = rospy.Time(0)
        self.clear_count = 0
        self.last_out = Twist()

        # ---------------- ROS IO ----------------
        self.pub = rospy.Publisher(self.cmd_out_topic, Twist, queue_size=10)

        rospy.Subscriber(self.cmd_in_topic, Twist, self._cmd_cb, queue_size=20)

        # hazard_level is optional; if no one publishes it, node still works with alert_bool or metrics
        rospy.Subscriber(self.hazard_level_topic, UInt8, self._hazard_level_cb, queue_size=10)
        rospy.Subscriber(self.alert_bool_topic, Bool, self._alert_cb, queue_size=10)

        rospy.Subscriber(self.margin_topic, Float32, self._margin_cb, queue_size=10)
        rospy.Subscriber(self.ttc_topic, Float32, self._ttc_cb, queue_size=10)
        rospy.Subscriber(self.dca_topic, Float32, self._dca_cb, queue_size=10)

        rospy.loginfo(
            "SafetySupervisor: in=%s out=%s hazard_level=%s",
            self.cmd_in_topic, self.cmd_out_topic, self.hazard_level_topic
        )

    # ---------------- Callbacks ----------------
    def _cmd_cb(self, msg: Twist):
        self.latest_cmd = msg
        self.latest_cmd_stamp = rospy.Time.now()

    def _hazard_level_cb(self, msg: UInt8):
        self.hazard_level = int(msg.data)
        self.hazard_stamp = rospy.Time.now()

    def _alert_cb(self, msg: Bool):
        self.alert_bool = bool(msg.data)
        self.alert_stamp = rospy.Time.now()

    def _margin_cb(self, msg: Float32):
        self.margin = float(msg.data)
        self.metrics_stamp = rospy.Time.now()

    def _ttc_cb(self, msg: Float32):
        self.ttc = float(msg.data)
        self.metrics_stamp = rospy.Time.now()

    def _dca_cb(self, msg: Float32):
        self.dca = float(msg.data)
        self.metrics_stamp = rospy.Time.now()

    # ---------------- Helpers ----------------
    def _stale(self, stamp, timeout) -> bool:
        if stamp is None:
            return True
        return (rospy.Time.now() - stamp).to_sec() > timeout

    def _infer_level_from_metrics(self) -> int:
        """
        Optional fallback classifier based on margin/ttc/dca.
        margin < 0 means inside threshold.
        """
        # If we have no metrics, return CLEAR
        if not math.isfinite(self.margin):
            return CLEAR

        # STOP conditions
        if self.margin <= self.stop_margin:
            return STOP
        # if math.isfinite(self.ttc) and math.isfinite(self.dca):
        #     if (0.0 <= self.ttc <= self.stop_ttc) and (self.dca <= self.stop_dca):
        #         return STOP

        # SLOW/WARN/CLEAR
        if self.margin <= self.slow_margin:
            return SLOW
        if self.margin <= self.warn_margin:
            return WARN
        return CLEAR

    # ---------------- Main step ----------------
    def step(self, dt: float):
        now = rospy.Time.now()

        cmd_stale = self._stale(self.latest_cmd_stamp, self.cmd_timeout)
        hazard_stale = self._stale(self.hazard_stamp, self.hazard_timeout)
        alert_stale = self._stale(self.alert_stamp, self.hazard_timeout)
        metrics_stale = self._stale(self.metrics_stamp, self.hazard_timeout)

        # Start from desired command; if cmd stale -> stop
        target = Twist()
        if not cmd_stale:
            target = self.latest_cmd

        # Decide hazard level:
        level = None

        # 1) Prefer hazard_level if it's fresh
        if not hazard_stale:
            level = self.hazard_level

        # 2) Else fallback to metrics inference if enabled and fresh
        if level is None and self.enable_infer_from_metrics and (not metrics_stale):
            level = self._infer_level_from_metrics()

        # 3) Else fallback to alert_bool if fresh
        if level is None and (not alert_stale):
            level = STOP if self.alert_bool else CLEAR

        # 4) Else choose fail-open or fail-safe
        if level is None:
            if self.fail_open_if_no_hazard:
                level = CLEAR
                rospy.logwarn_throttle(
                    1.0,
                    "[safety] No fresh hazard input; fail-open enabled, passing through cmd_vel"
                )
            else:
                level = STOP
                rospy.logwarn_throttle(
                    1.0,
                    "[safety] No fresh hazard input; fail-safe STOP"
                )

        # STOP hold
        if level == STOP:
            self.stop_until = max(self.stop_until, now + rospy.Duration(self.stop_hold_s))
            self.clear_count = 0

        level_effective = STOP if now < self.stop_until else level

        # CLEAR hysteresis counter
        if level_effective == CLEAR:
            self.clear_count += 1
        else:
            self.clear_count = 0

        # Apply policy
        if level_effective == STOP:
            target.linear.x = 0.0
            target.angular.z = 0.0
            rospy.logwarn_throttle(0.5, "[safety] STOP")

        elif level_effective == SLOW:
            s = clamp(self.slow_scale, 0.0, 1.0)
            target.linear.x *= s
            target.angular.z *= s
            rospy.logwarn_throttle(0.5, "[safety] SLOW scale=%.2f", s)

        elif level_effective == WARN:
            rospy.logwarn_throttle(0.75, "[safety] WARN")
            if self.use_warn_scale:
                s = clamp(self.warn_scale, 0.0, 1.0)
                target.linear.x *= s
                target.angular.z *= s

        # Gentle re-entry right after STOP/SLOW: require N clear frames
        if (level_effective == CLEAR) and (self.clear_count < self.clear_frames_to_release):
            target.linear.x *= 0.6
            target.angular.z *= 0.6

        # Caps
        target.linear.x = clamp(target.linear.x, -self.max_linear, self.max_linear)
        target.angular.z = clamp(target.angular.z, -self.max_angular, self.max_angular)

        # Ramp (maybe remove? Does escooter already have this?)
        # out = self._apply_ramp(target, dt)
        out = target

        # Publish heartbeat
        self.pub.publish(out)
        self.last_out = out


def main():
    rospy.init_node("cmd_vel_safety_supervisor")

    node = CmdVelSafetySupervisor()
    rate = rospy.Rate(node.rate_hz)

    last = rospy.Time.now()
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        dt = (now - last).to_sec()
        if dt <= 0.0:
            dt = 1.0 / node.rate_hz
        node.step(dt)
        last = now
        rate.sleep()


if __name__ == "__main__":
    main()
