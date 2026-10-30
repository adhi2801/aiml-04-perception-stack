"""CPU-only tests for the safety-relevant logic (no CARLA, ROS2, GPU or dataset needed)."""
import cv2
import numpy as np
import pytest

from perception.depth.metric import (depth_metrics, fit_inverse_affine, focal_px, ground_plane_distance,
                                     object_distance, to_metric)
from perception.fusion.aeb import AEB, Action, AEBConfig, TrackedObject, estimate_ttc
from perception.lanes.classical import ClassicalLaneDetector

W = 1280
CENTER_BOX = (600, 300, 680, 500)   # in the ego corridor
SIDE_BOX = (20, 300, 100, 500)      # far left, outside the corridor


def run(aeb, dists, box=CENTER_BOX, cls="car", dt=0.1):
    out = None
    for i, d in enumerate(dists):
        out = aeb.step(i * dt, [TrackedObject(1, cls, box, d)], W)
    return out


# ---------- TTC ----------
def test_ttc_constant_closing_speed():
    t = np.arange(8) * 0.1
    d = 20 - 10 * t  # closing at 10 m/s
    ttc, closing = estimate_ttc(t, d)
    assert closing == pytest.approx(10, rel=1e-6)
    assert ttc == pytest.approx(d[-1] / 10, rel=1e-6)


def test_ttc_none_when_receding_or_noise():
    t = np.arange(8) * 0.1
    assert estimate_ttc(t, 10 + 5 * t)[0] is None
    assert estimate_ttc(t, 10 + np.random.default_rng(0).normal(0, 0.02, 8))[0] is None


# ---------- AEB decisions ----------
def test_brakes_for_fast_approach_in_corridor():
    action, threat = run(AEB(), [30 - 15 * 0.1 * i for i in range(8)])  # 15 m/s closing, ~19.5 m -> TTC ~1.3 s
    assert action == Action.BRAKE and threat.ttc_s < 1.8


def test_ignores_same_threat_outside_corridor():
    action, threat = run(AEB(), [30 - 15 * 0.1 * i for i in range(8)], box=SIDE_BOX)
    assert action == Action.CRUISE and threat.in_corridor is False


def test_warns_for_close_pedestrian_even_if_static():
    action, _ = run(AEB(), [9.0] * 6, cls="pedestrian")
    assert action == Action.WARN


def test_brakes_inside_min_distance():
    action, _ = run(AEB(), [4.0] * 3)
    assert action == Action.BRAKE


def test_hysteresis_holds_then_releases():
    aeb = AEB(AEBConfig(hold_frames=3))
    assert run(aeb, [4.0])[0] == Action.BRAKE
    held = [aeb.step(1 + i * 0.1, [], W)[0] for i in range(4)]
    assert held == [Action.BRAKE] * 3 + [Action.CRUISE]


def test_lane_corridor_overrides_fixed_band():
    aeb = AEB()
    assert aeb.in_corridor(SIDE_BOX, W, corridor=(0, 200))
    assert not aeb.in_corridor(CENTER_BOX, W, corridor=(0, 200))


# ---------- depth ----------
def test_inverse_affine_recovers_metric_depth():
    rng = np.random.default_rng(0)
    gt = rng.uniform(2, 60, (120, 160))
    inv_rel = (1 / gt - 0.01) / 0.05 + rng.normal(0, 0.01, gt.shape)  # true s=0.05, t=0.01, small noise
    s, t = fit_inverse_affine(inv_rel, gt)
    assert s == pytest.approx(0.05, rel=0.05) and t == pytest.approx(0.01, abs=0.005)
    m = depth_metrics(to_metric(inv_rel, s, t), gt)
    assert m["delta1"] > 0.9


def test_object_distance_ignores_background():
    depth = np.full((100, 100), 50.0)
    depth[30:80, 40:60] = 10.0
    assert object_distance(depth, (35, 25, 65, 85)) == pytest.approx(10.0)


def test_ground_plane_distance_geometry():
    f = focal_px(1280, 90)  # 640 px
    # bottom edge 64 px below the horizon at 1.5 m camera height -> 640 * 1.5 / 64 = 15 m
    assert ground_plane_distance((0, 0, 10, 360 + 64), 720, 1280, 90, 1.5) == pytest.approx(15.0)
    assert ground_plane_distance((0, 0, 10, 300), 720, 1280, 90, 1.5) is None  # above horizon


# ---------- lanes ----------
def synthetic_road():
    img = np.full((720, 1280, 3), 60, np.uint8)
    cv2.line(img, (300, 700), (580, 460), (255, 255, 255), 10)
    cv2.line(img, (1060, 700), (700, 460), (255, 255, 255), 10)
    return img


def test_classical_lanes_find_both_lines_and_corridor():
    det = ClassicalLaneDetector(1280, 720, [[0.16, 0.95], [0.44, 0.63], [0.56, 0.63], [0.86, 0.95]])
    res = det(synthetic_road())
    assert res.valid and res.mask.any()
    left, right = det.corridor_at(650)
    assert left < 640 < right
