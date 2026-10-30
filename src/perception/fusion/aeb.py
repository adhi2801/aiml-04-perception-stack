"""Autonomous Emergency Braking logic: tracked objects + distance -> time-to-collision -> action.

Pure Python (no ROS, no CARLA), so every rule is unit-tested and the same code runs in the
offline pipeline, the CARLA loop and the ROS2 decision node.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Action(IntEnum):
    CRUISE = 0
    WARN = 1
    BRAKE = 2


@dataclass
class TrackedObject:
    track_id: int
    cls: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    distance_m: float | None
    conf: float = 1.0


@dataclass
class Threat:
    track_id: int
    cls: str
    distance_m: float
    ttc_s: float | None
    closing_mps: float
    in_corridor: bool
    action: Action


@dataclass
class AEBConfig:
    vulnerable_classes: tuple[str, ...] = ("pedestrian", "rider", "bicycle", "motorcycle")
    ttc_warn_s: float = 3.0
    ttc_brake_s: float = 1.8
    min_distance_m: float = 6.0
    corridor_half_width: float = 0.18
    hold_frames: int = 15
    history: int = 8

    @classmethod
    def from_dict(cls, d: dict) -> "AEBConfig":
        d = {**d, "vulnerable_classes": tuple(d.get("vulnerable_classes", cls.vulnerable_classes))}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def estimate_ttc(times: np.ndarray, dists: np.ndarray) -> tuple[float | None, float]:
    """Least-squares slope of distance over time. Returns (ttc_s or None if not closing, closing speed m/s).

    A line fit over several frames is far less noisy than differencing two monocular depth samples.
    """
    if len(times) < 3:
        return None, 0.0
    slope = np.polyfit(times - times[0], dists, 1)[0]
    closing = -slope
    if closing <= 0.2:  # receding, or closing slower than measurement noise
        return None, float(closing)
    return float(dists[-1] / closing), float(closing)


@dataclass
class AEB:
    cfg: AEBConfig = field(default_factory=AEBConfig)

    def __post_init__(self):
        self._hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=self.cfg.history))
        self._hold = 0
        self.last_threat: Threat | None = None

    def in_corridor(self, box, img_w: int, corridor: tuple[float, float] | None) -> bool:
        """corridor = (left_x, right_x) of the ego lane at the object's bottom row, or None for a fixed band."""
        cx = (box[0] + box[2]) / 2
        if corridor is not None:
            left, right = corridor
            margin = 0.15 * (right - left)  # objects stepping into the lane count
            return left - margin <= cx <= right + margin
        half = self.cfg.corridor_half_width * img_w
        return abs(cx - img_w / 2) <= half

    def step(self, t: float, objects: list[TrackedObject], img_w: int,
             corridor_fn=None) -> tuple[Action, Threat | None]:
        """corridor_fn(y_pixel) -> (left_x, right_x) | None, typically from the lane detector."""
        live = set()
        worst: Threat | None = None
        for o in objects:
            live.add(o.track_id)
            if o.distance_m is None or not np.isfinite(o.distance_m):
                continue
            h = self._hist[o.track_id]
            h.append((t, o.distance_m))
            arr = np.array(h)
            ttc, closing = estimate_ttc(arr[:, 0], arr[:, 1])
            corridor = corridor_fn(o.box[3]) if corridor_fn else None
            inside = self.in_corridor(o.box, img_w, corridor)

            action = Action.CRUISE
            if inside:
                vulnerable = o.cls in self.cfg.vulnerable_classes
                if o.distance_m < self.cfg.min_distance_m or (ttc is not None and ttc < self.cfg.ttc_brake_s):
                    action = Action.BRAKE
                elif (ttc is not None and ttc < self.cfg.ttc_warn_s) or (vulnerable and o.distance_m < 2 * self.cfg.min_distance_m):
                    action = Action.WARN
            threat = Threat(o.track_id, o.cls, o.distance_m, ttc, closing, inside, action)
            if worst is None or (threat.action, -threat.distance_m) > (worst.action, -worst.distance_m):
                worst = threat

        for tid in list(self._hist):  # forget tracks that disappeared
            if tid not in live:
                del self._hist[tid]

        action = worst.action if worst else Action.CRUISE
        if action == Action.BRAKE:
            self._hold = self.cfg.hold_frames
        elif self._hold > 0:  # hysteresis: avoid brake/release chatter when detections flicker
            self._hold -= 1
            action = Action.BRAKE
        self.last_threat = worst
        return action, worst
