"""Turning MiDaS relative depth into meters, and measuring how wrong it is.

MiDaS predicts *relative inverse* depth: correct up to an unknown scale and shift. We fit
    1 / z  =  s * inv_rel + t
by least squares against ground-truth depth (CARLA's depth camera gives it for free), then
report standard monocular-depth metrics. A second, model-free estimate comes from flat-ground
geometry, z = f * h / (v_bottom - v_horizon), which is a useful cross-check on distant objects.
"""
from __future__ import annotations

import math

import numpy as np


def fit_inverse_affine(inv_rel: np.ndarray, gt_depth: np.ndarray, max_depth: float = 80.0) -> tuple[float, float]:
    m = np.isfinite(gt_depth) & (gt_depth > 0.5) & (gt_depth < max_depth) & np.isfinite(inv_rel)
    x, y = inv_rel[m].ravel(), 1.0 / gt_depth[m].ravel()
    A = np.stack([x, np.ones_like(x)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(s), float(t)


def to_metric(inv_rel: np.ndarray, scale: float, shift: float, max_depth: float = 120.0) -> np.ndarray:
    inv = scale * inv_rel + shift
    return np.where(inv > 1.0 / max_depth, 1.0 / np.maximum(inv, 1e-6), max_depth)


def depth_metrics(pred: np.ndarray, gt: np.ndarray, max_depth: float = 80.0) -> dict:
    m = np.isfinite(gt) & (gt > 0.5) & (gt < max_depth) & np.isfinite(pred) & (pred > 0)
    p, g = pred[m], gt[m]
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse_m": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "n": int(m.sum()),
    }


def object_distance(depth_m: np.ndarray, box, shrink: float = 0.3) -> float | None:
    """Robust distance to a detected object: median depth over the box core.

    The central region avoids background pixels at the box edges; the median ignores the
    road/sky that still leaks in.
    """
    h, w = depth_m.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    xa, xb = int(max(0, x1 + shrink * bw)), int(min(w, x2 - shrink * bw))
    ya, yb = int(max(0, y1 + shrink * bh)), int(min(h, y2 - shrink * 0.5 * bh))
    if xb <= xa or yb <= ya:
        return None
    patch = depth_m[ya:yb, xa:xb]
    patch = patch[np.isfinite(patch)]
    return float(np.median(patch)) if patch.size else None


def focal_px(width: int, fov_deg: float) -> float:
    return width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))


def ground_plane_distance(box, img_h: int, img_w: int, fov_deg: float, cam_height_m: float,
                          horizon_y: float | None = None) -> float | None:
    """Distance from the box's bottom edge, assuming a flat road and a level camera."""
    f = focal_px(img_w, fov_deg)
    v0 = img_h / 2 if horizon_y is None else horizon_y
    dv = box[3] - v0
    return float(f * cam_height_m / dv) if dv > 1 else None
