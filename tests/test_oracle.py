"""Oracle detection: simulator boxes replace detector + tracker, and the rest of the pipeline is unchanged."""

from __future__ import annotations

import numpy as np

from perception.fusion.aeb import AEB, Action, AEBConfig
from perception.lanes.classical import LaneResult
from perception.pipeline import PerceptionPipeline, oracle_boxes

W, H = 1280, 720


class NoLanes:
    def __call__(self, bgr):
        return LaneResult(None, None, np.zeros((H, W), np.uint8), None, False)

    def corridor_at(self, y):
        return None


class NoDetector:
    def track(self, *a, **k):
        raise AssertionError("oracle detection must not run the detector")


def pipeline() -> PerceptionPipeline:
    """The pipeline without its __init__ (which loads YOLO weights): only what the oracle path touches."""
    p = PerceptionPipeline.__new__(PerceptionPipeline)
    p.cfg = {"detector": {}}
    p.w, p.h, p.fov, p.cam_h = W, H, 90, 1.5
    p.model, p.lanes, p.depth_source, p.depth_model = NoDetector(), NoLanes(), "ground_plane", None
    p.aeb = AEB(AEBConfig())
    return p


def pedestrian_box(dist_m: float) -> list[float]:
    """A pedestrian centred in the image, its feet where the ground plane puts `dist_m` (90 deg FOV, 1.5 m camera)."""
    f = W / 2  # focal length in pixels for a 90 deg horizontal field of view
    y_feet = H / 2 + f * 1.5 / dist_m
    return [W / 2 - 20, y_feet - 80, W / 2 + 20, y_feet]


def test_oracle_boxes_use_actor_ids_and_full_confidence():
    out = oracle_boxes([{"id": 42, "cls": "pedestrian", "box": [1, 2, 3, 4], "distance_m": 9.0}])
    assert out == [(42, "pedestrian", (1.0, 2.0, 3.0, 4.0), 1.0)]


def test_oracle_detection_skips_the_detector_and_brakes_for_a_closing_pedestrian():
    p, bgr = pipeline(), np.zeros((H, W, 3), np.uint8)
    actions = []
    for i, d in enumerate(np.linspace(20, 5, 12)):  # closing at ~13.6 m/s
        res = p(bgr, t=i * 0.1, gt_objects=[{"id": 7, "cls": "pedestrian", "box": pedestrian_box(d)}])
        actions.append(res.action)
        assert [o.track_id for o in res.objects] == [7]
    assert actions[-1] == Action.BRAKE
    assert abs(res.objects[0].distance_m - 5.0) < 0.5  # ground-plane distance from the oracle box


def test_oracle_depth_overrides_ground_plane_distance_for_oracle_boxes():
    p, bgr = pipeline(), np.zeros((H, W, 3), np.uint8)
    depth = np.full((H, W), 33.0, np.float32)
    res = p(bgr, t=0.0, gt_depth=depth, gt_objects=[{"id": 1, "cls": "car", "box": pedestrian_box(10)}])
    assert abs(res.objects[0].distance_m - 33.0) < 1e-3
