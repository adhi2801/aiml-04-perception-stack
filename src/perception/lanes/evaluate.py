"""Classical (OpenCV) vs learned (LR-ASPP) lane detection on held-out CARLA sequences.

  python -m perception.lanes.evaluate --seq data/carla/Town10_test --weights models/lane_lraspp.pt

Metrics (lower half of the image, where lanes matter for control):
  * tolerant pixel precision / recall / F1: a predicted pixel counts if a GT lane pixel is within
    `tol` px (and vice versa), so a correctly placed line drawn 2 px wider is not penalized
  * latency per frame (ms)
  * robustness: per-sequence breakdown (weather / town) when several --seq are given
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from perception.lanes.classical import ClassicalLaneDetector
from perception.lanes.seg_model import LaneSegmenter


def tolerant_prf(pred: np.ndarray, gt: np.ndarray, tol: int = 5) -> tuple[int, int, int, int]:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    p, g = pred > 0, gt > 0
    g_dil, p_dil = cv2.dilate(g.astype(np.uint8), k) > 0, cv2.dilate(p.astype(np.uint8), k) > 0
    return int((p & g_dil).sum()), int(p.sum()), int((g & p_dil).sum()), int(g.sum())


def evaluate(seq: Path, classical: ClassicalLaneDetector, learned: LaneSegmenter, tol: int) -> dict:
    acc = {"classical": np.zeros(4), "learned": np.zeros(4)}
    lat = {"classical": [], "learned": []}
    for rgb_path in sorted((seq / "rgb").glob("*.jpg")):
        img = cv2.imread(str(rgb_path))
        gt = cv2.imread(str(seq / "lanes" / (rgb_path.stem + ".png")), 0)
        roi = slice(img.shape[0] // 2, None)
        for name, fn in (("classical", lambda im: classical(im).mask), ("learned", learned)):
            t0 = time.perf_counter()
            pred = fn(img)
            lat[name].append((time.perf_counter() - t0) * 1000)
            acc[name] += tolerant_prf(pred[roi], gt[roi], tol)
    out = {}
    for name, (tp_p, n_p, tp_g, n_g) in acc.items():
        prec, rec = tp_p / max(n_p, 1), tp_g / max(n_g, 1)
        out[name] = {"precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
                     "latency_ms_p50": round(float(np.median(lat[name][5:])), 2)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="+", type=Path, required=True)
    ap.add_argument("--weights", default="models/lane_lraspp.pt")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--tol", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    learned = LaneSegmenter(args.weights)
    results = {}
    for seq in args.seq:
        meta = json.loads((seq / "meta.json").read_text())
        classical = ClassicalLaneDetector(meta["camera"]["width"], meta["camera"]["height"], cfg["lanes"]["src"],
                                          cfg["lanes"]["smoothing"])
        results[seq.name] = {"town": meta["town"], "weather": meta["weather"], **evaluate(seq, classical, learned, args.tol)}
        print(seq.name, json.dumps(results[seq.name]))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/lanes_comparison.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
