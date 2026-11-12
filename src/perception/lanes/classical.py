"""Classical lane detection: color+gradient thresholds -> bird's-eye warp -> sliding windows -> polynomial.

Deliberately interpretable, so it is the baseline the learned segmentation model is compared against
(see lanes/evaluate.py). Temporal EMA smoothing plus a sanity check on lane width keep it from
jumping between frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LaneResult:
    left_fit: np.ndarray | None    # x = a*y^2 + b*y + c in bird's-eye pixels
    right_fit: np.ndarray | None
    mask: np.ndarray               # uint8 lane-line mask in the ORIGINAL image (for evaluation)
    center_offset_px: float | None  # + means the car sits right of the lane center
    valid: bool


class ClassicalLaneDetector:
    def __init__(self, width: int, height: int, src_frac: list[list[float]], smoothing: float = 0.7,
                 n_windows: int = 9, margin: int = 80, min_pix: int = 40, mask_fn=None):
        """mask_fn(bgr) -> uint8 lane-pixel mask replaces the hand-tuned colour/edge threshold (step 1), e.g. the
        learned LaneSegmenter; the bird's-eye fit, sanity check, smoothing and ego corridor stay the same."""
        self.w, self.h = width, height
        self.mask_fn = mask_fn
        src = np.float32([[x * width, y * height] for x, y in src_frac])
        dst = np.float32([[0.25 * width, height], [0.25 * width, 0], [0.75 * width, 0], [0.75 * width, height]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        self.smoothing, self.n_windows, self.margin, self.min_pix = smoothing, n_windows, margin, min_pix
        self._left: np.ndarray | None = None
        self._right: np.ndarray | None = None

    # ---- 1. binary lane-pixel mask ----
    @staticmethod
    def threshold(bgr: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
        l, s = hls[..., 1], hls[..., 2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sobel = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        sobel = (255 * sobel / max(sobel.max(), 1e-6)).astype(np.uint8)
        white = (l > 200)
        yellow = (s > 120) & (l > 100) & (hls[..., 0] > 15) & (hls[..., 0] < 35)
        edges = (sobel > 40) & (l > 120)
        return ((white | yellow | edges) * 255).astype(np.uint8)

    # ---- 2. sliding-window search in bird's-eye view ----
    def _search(self, warped: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        hist = warped[warped.shape[0] // 2:].sum(axis=0)
        mid = hist.shape[0] // 2
        bases = [int(np.argmax(hist[:mid])), int(np.argmax(hist[mid:]) + mid)]
        ys, xs = warped.nonzero()
        win_h = warped.shape[0] // self.n_windows
        fits = []
        for base in bases:
            x_cur, idx = base, []
            for wdw in range(self.n_windows):
                y_lo, y_hi = warped.shape[0] - (wdw + 1) * win_h, warped.shape[0] - wdw * win_h
                good = ((ys >= y_lo) & (ys < y_hi) & (xs >= x_cur - self.margin) & (xs < x_cur + self.margin)).nonzero()[0]
                idx.append(good)
                if len(good) > self.min_pix:
                    x_cur = int(xs[good].mean())
            idx = np.concatenate(idx)
            fits.append(np.polyfit(ys[idx], xs[idx], 2) if len(idx) > 3 * self.min_pix else None)
        return fits[0], fits[1]

    def _smooth(self, prev, new):
        if new is None:
            return prev
        return new if prev is None else self.smoothing * prev + (1 - self.smoothing) * new

    # ---- 3. full frame ----
    def __call__(self, bgr: np.ndarray) -> LaneResult:
        binary = self.mask_fn(bgr) if self.mask_fn is not None else self.threshold(bgr)
        warped = cv2.warpPerspective(binary, self.M, (self.w, self.h), flags=cv2.INTER_NEAREST)
        left, right = self._search(warped)

        # Sanity check: reject a pair whose bottom separation is implausible for one lane.
        if left is not None and right is not None:
            y = self.h - 1
            width = np.polyval(right, y) - np.polyval(left, y)
            if not (0.3 * self.w < width < 0.75 * self.w):
                left = right = None
        self._left, self._right = self._smooth(self._left, left), self._smooth(self._right, right)

        mask_bev = np.zeros((self.h, self.w), np.uint8)
        ploty = np.arange(self.h)
        for fit in (self._left, self._right):
            if fit is not None:
                pts = np.stack([np.polyval(fit, ploty), ploty], axis=1).astype(np.int32)
                cv2.polylines(mask_bev, [pts], False, 255, thickness=12)
        mask = cv2.warpPerspective(mask_bev, self.Minv, (self.w, self.h), flags=cv2.INTER_NEAREST)

        offset = None
        if self._left is not None and self._right is not None:
            lane_center_bev = (np.polyval(self._left, self.h - 1) + np.polyval(self._right, self.h - 1)) / 2
            offset = float(self.w / 2 - lane_center_bev)
        return LaneResult(self._left, self._right, mask, offset, self._left is not None and self._right is not None)

    def corridor_at(self, y_img: float) -> tuple[float, float] | None:
        """Ego-lane (left_x, right_x) in the ORIGINAL image at row y_img. Used by AEB."""
        if self._left is None or self._right is None:
            return None
        ploty = np.arange(0, self.h, 4, dtype=np.float32)
        out = []
        for fit in (self._left, self._right):
            pts = np.stack([np.polyval(fit, ploty), ploty], axis=1).astype(np.float32)[None]
            img_pts = cv2.perspectiveTransform(pts, self.Minv)[0]
            # visible part of the lane only; sample x at the requested row
            order = np.argsort(img_pts[:, 1])
            out.append(float(np.interp(y_img, img_pts[order, 1], img_pts[order, 0])))
        return (out[0], out[1]) if out[0] < out[1] else None

    def draw(self, bgr: np.ndarray, res: LaneResult) -> np.ndarray:
        out = bgr.copy()
        if res.valid:
            ploty = np.arange(self.h)
            left = np.stack([np.polyval(self._left, ploty), ploty], 1)
            right = np.stack([np.polyval(self._right, ploty), ploty], 1)[::-1]
            poly = np.zeros_like(bgr)
            cv2.fillPoly(poly, [np.concatenate([left, right]).astype(np.int32)], (0, 180, 0))
            out = cv2.addWeighted(out, 1, cv2.warpPerspective(poly, self.Minv, (self.w, self.h)), 0.3, 0)
        out[res.mask > 0] = (0, 255, 255)
        return out
