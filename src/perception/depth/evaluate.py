"""Calibrate MiDaS to meters on training sequences, then measure distance error on held-out ones.

  python -m perception.depth.evaluate --calib data/carla/Town03_clear --test data/carla/Town10_test --write-config

Reports (reports/depth_eval.json):
  * dense metrics vs CARLA ground truth: AbsRel, RMSE, delta<1.25
  * per-object distance error, MiDaS vs flat-ground geometry, binned by true distance
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from perception.depth.metric import (depth_metrics, fit_inverse_affine, ground_plane_distance, object_distance,
                                     to_metric)
from perception.depth.midas import MiDaS

BINS = [(0, 10), (10, 20), (20, 40), (40, 80)]


def frames(seq: Path, stride: int):
    gts = [json.loads(line) for line in (seq / "gt.jsonl").read_text().splitlines()]
    for g in gts[::stride]:
        name = f"{g['frame']:06d}"
        yield (cv2.imread(str(seq / "rgb" / f"{name}.jpg")),
               np.load(seq / "depth" / f"{name}.npy").astype(np.float32), g["objects"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", nargs="+", type=Path, required=True)
    ap.add_argument("--test", nargs="+", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--write-config", action="store_true", help="store the fitted scale/shift in the config")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cam = cfg["camera"]
    midas = MiDaS(cfg["depth"]["model"])

    # 1) Calibration: pool sub-sampled pixels from the calibration sequences.
    xs, ys = [], []
    for seq in args.calib:
        for img, depth, _ in frames(seq, args.stride):
            inv = midas(img)[::8, ::8]
            xs.append(inv.ravel())
            ys.append(depth[::8, ::8].ravel())
    scale, shift = fit_inverse_affine(np.concatenate(xs), np.concatenate(ys))
    print(f"fitted 1/z = {scale:.6g} * midas + {shift:.6g}")

    # 2) Held-out evaluation.
    dense, per_obj = [], {"midas": {b: [] for b in BINS}, "ground_plane": {b: [] for b in BINS}}
    for seq in args.test:
        for img, depth, objs in frames(seq, args.stride):
            pred = to_metric(midas(img), scale, shift)
            dense.append(depth_metrics(pred, depth))
            for o in objs:
                true = o["distance_m"]
                b = next((b for b in BINS if b[0] <= true < b[1]), None)
                if b is None:
                    continue
                est = {"midas": object_distance(pred, o["box"]),
                       "ground_plane": ground_plane_distance(o["box"], cam["height"], cam["width"], cam["fov_deg"],
                                                             cam["mount_height_m"])}
                for k, v in est.items():
                    if v is not None:
                        per_obj[k][b].append(abs(v - true) / true)

    n = sum(d["n"] for d in dense)
    summary = {
        "calibration": {"scale": scale, "shift": shift, "sequences": [s.name for s in args.calib]},
        "dense": {k: round(sum(d[k] * d["n"] for d in dense) / n, 4) for k in ("abs_rel", "rmse_m", "delta1")},
        "object_distance_rel_error_median": {
            k: {f"{a}-{b}m": (round(float(np.median(v)), 3) if v else None) for (a, b), v in bins.items()}
            for k, bins in per_obj.items()},
        "object_counts": {f"{a}-{b}m": len(v) for (a, b), v in per_obj["midas"].items()},
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/depth_eval.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if args.write_config:
        text = Path(args.config).read_text()
        text = text.replace("scale: null", f"scale: {scale:.8g}").replace("shift: null", f"shift: {shift:.8g}")
        Path(args.config).write_text(text)


if __name__ == "__main__":
    main()
