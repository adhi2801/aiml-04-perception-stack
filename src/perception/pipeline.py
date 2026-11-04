"""The full perception stack for one camera frame:

    YOLO (PyTorch / ONNX / TensorRT) + ByteTrack -> tracked boxes
    MiDaS (+ metric calibration) or ground-plane geometry -> distance per object
    classical lane detector -> ego corridor
    AEB -> CRUISE / WARN / BRAKE

Used identically by the offline video runner, the direct CARLA loop and the ROS2 perception node.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from perception.depth.metric import ground_plane_distance, object_distance, to_metric
from perception.fusion.aeb import AEB, Action, AEBConfig, Threat, TrackedObject
from perception.lanes.classical import ClassicalLaneDetector, LaneResult

COLORS = {Action.CRUISE: (80, 200, 80), Action.WARN: (0, 200, 255), Action.BRAKE: (0, 0, 255)}
# COCO-pretrained baselines use different names for the classes AEB cares about; normalize to BDD100K names.
CLASS_ALIASES = {"person": "pedestrian", "motorbike": "motorcycle"}
DRIVING_CLASSES = {"pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
                   "traffic light", "traffic sign", "stop sign"}


@dataclass
class FrameResult:
    objects: list[TrackedObject]
    lanes: LaneResult
    action: Action
    threat: Threat | None
    timings_ms: dict = field(default_factory=dict)


class PerceptionPipeline:
    def __init__(self, cfg_path: str | Path = "configs/pipeline.yaml", use_depth_model: bool = True,
                 weights: str | None = None):
        from ultralytics import YOLO

        self.cfg = yaml.safe_load(Path(cfg_path).read_text())
        cam, det = self.cfg["camera"], self.cfg["detector"]
        self.w, self.h, self.fov = cam["width"], cam["height"], cam["fov_deg"]
        self.cam_h = cam["mount_height_m"]
        self.model = YOLO(weights or det["weights"], task="detect")
        self.names = {i: CLASS_ALIASES.get(n, n) for i, n in self.model.names.items()}
        self.lanes = ClassicalLaneDetector(self.w, self.h, self.cfg["lanes"]["src"], self.cfg["lanes"]["smoothing"])
        self.aeb = AEB(AEBConfig.from_dict(self.cfg["aeb"]))
        dcfg = self.cfg["depth"]
        self.depth_model = None
        if use_depth_model:
            from perception.depth.midas import MiDaS
            self.depth_model = MiDaS(dcfg["model"], device=dcfg.get("device"))
        self.depth_scale, self.depth_shift = dcfg.get("scale"), dcfg.get("shift")

    def __call__(self, bgr: np.ndarray, t: float | None = None, gt_depth: np.ndarray | None = None) -> FrameResult:
        t = time.monotonic() if t is None else t
        det = self.cfg["detector"]
        timings = {}

        t0 = time.perf_counter()
        r = self.model.track(bgr, imgsz=det["imgsz"], conf=det["conf"], iou=det["iou"], half=det["half"],
                             tracker=self.cfg["tracker"], persist=True, device=det.get("device"), verbose=False)[0]
        timings["detect_track"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        depth_m = None
        if gt_depth is not None:
            depth_m = gt_depth  # oracle mode: isolates detection/AEB errors from depth errors
        elif self.depth_model is not None and self.depth_scale is not None:
            depth_m = to_metric(self.depth_model(bgr), self.depth_scale, self.depth_shift)
        timings["depth"] = (time.perf_counter() - t0) * 1000

        objects = []
        boxes = r.boxes
        if boxes is not None and boxes.id is not None:
            for xyxy, cls, conf, tid in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int),
                                            boxes.conf.cpu().numpy(), boxes.id.cpu().numpy().astype(int)):
                if self.names[int(cls)] not in DRIVING_CLASSES:  # COCO baseline also sees cups, benches, ...
                    continue
                box = tuple(float(v) for v in xyxy)
                dist = object_distance(depth_m, box) if depth_m is not None else \
                    ground_plane_distance(box, self.h, self.w, self.fov, self.cam_h)
                objects.append(TrackedObject(int(tid), self.names[int(cls)], box, dist, float(conf)))

        t0 = time.perf_counter()
        lanes = self.lanes(bgr)
        timings["lanes"] = (time.perf_counter() - t0) * 1000

        action, threat = self.aeb.step(t, objects, self.w, self.lanes.corridor_at if lanes.valid else None)
        timings["total"] = sum(timings.values())
        return FrameResult(objects, lanes, action, threat, timings)

    def draw(self, bgr: np.ndarray, res: FrameResult) -> np.ndarray:
        out = self.lanes.draw(bgr, res.lanes)
        for o in res.objects:
            x1, y1, x2, y2 = map(int, o.box)
            hot = res.threat is not None and o.track_id == res.threat.track_id and res.threat.action > Action.CRUISE
            color = COLORS[res.threat.action] if hot else (255, 180, 60)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if hot else 2)
            label = f"#{o.track_id} {o.cls}" + (f" {o.distance_m:.1f}m" if o.distance_m else "")
            cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        banner = res.action.name
        if res.threat and res.threat.ttc_s is not None:
            banner += f"  TTC {res.threat.ttc_s:.1f}s"
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(out, banner, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[res.action], 2, cv2.LINE_AA)
        fps = 1000 / max(res.timings_ms.get("total", 1), 1e-3)
        cv2.putText(out, f"{fps:5.1f} FPS  det {res.timings_ms.get('detect_track', 0):.0f}ms  "
                         f"depth {res.timings_ms.get('depth', 0):.0f}ms  lanes {res.timings_ms.get('lanes', 0):.0f}ms",
                    (out.shape[1] - 520, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        return out
