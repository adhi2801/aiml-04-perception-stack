"""PyTorch vs ONNX Runtime vs TensorRT (FP16, INT8): latency, FPS and accuracy cost, on THIS GPU.

  python -m perception.optimize.benchmark --weights models/yolo11s_bdd.pt --data data/bdd_yolo/data.yaml

Exports each format with Ultralytics, then for each one:
  * latency: 50 warm-up + 300 timed single-image inferences (p50 / p95 / mean, FPS = 1000 / mean)
  * accuracy: mAP50-95 and mAP50 on the val split
Writes reports/benchmark.json + reports/benchmark.md and logs a table to W&B if WANDB_API_KEY is set.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def time_model(model, imgs: list[np.ndarray], imgsz: int, half: bool, warmup: int = 50, runs: int = 300) -> dict:
    for i in range(warmup):
        model.predict(imgs[i % len(imgs)], imgsz=imgsz, half=half, verbose=False)
    lat = []
    for i in range(runs):
        t0 = time.perf_counter()
        model.predict(imgs[i % len(imgs)], imgsz=imgsz, half=half, verbose=False)
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    return {"p50_ms": round(float(np.percentile(lat, 50)), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2),
            "mean_ms": round(float(lat.mean()), 2), "fps": round(float(1000 / lat.mean()), 1)}


def main() -> None:
    import cv2
    import torch
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--images", type=int, default=64, help="val images used for timing")
    ap.add_argument("--skip-int8", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data).parent
    img_paths = sorted((data_root / "images" / "val").glob("*.jpg"))[: args.images]
    imgs = [cv2.imread(str(p)) for p in img_paths]

    base = YOLO(args.weights)
    variants = [("PyTorch FP32", args.weights, False)]
    variants.append(("ONNX Runtime FP32", base.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True), False))
    variants.append(("TensorRT FP16", base.export(format="engine", imgsz=args.imgsz, half=True, workspace=2), True))
    if not args.skip_int8:
        # INT8 calibrates on real val images; the engine filename collides with FP16, so rename it.
        int8 = Path(base.export(format="engine", imgsz=args.imgsz, int8=True, data=args.data, workspace=2))
        int8 = int8.rename(int8.with_name(int8.stem + "_int8.engine"))
        variants.append(("TensorRT INT8", str(int8), False))

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    rows = []
    for name, path, half in variants:
        m = YOLO(path, task="detect")
        speed = time_model(m, imgs, args.imgsz, half)
        val = m.val(data=args.data, imgsz=args.imgsz, half=half, batch=1, verbose=False, plots=False)
        row = {"variant": name, "file": Path(path).name, "size_mb": round(Path(path).stat().st_size / 2**20, 1), **speed,
               "map50_95": round(float(val.box.map), 4), "map50": round(float(val.box.map50), 4)}
        rows.append(row)
        print(json.dumps(row))
        del m
        torch.cuda.empty_cache()

    ref = rows[0]
    for r in rows:
        r["speedup_vs_pytorch"] = round(ref["mean_ms"] / r["mean_ms"], 2)
        r["map_drop_pts"] = round(100 * (ref["map50_95"] - r["map50_95"]), 2)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/benchmark.json").write_text(json.dumps({"gpu": gpu, "imgsz": args.imgsz, "rows": rows}, indent=2))
    cols = ["variant", "size_mb", "p50_ms", "p95_ms", "fps", "speedup_vs_pytorch", "map50_95", "map_drop_pts"]
    md = [f"GPU: **{gpu}**, input {args.imgsz}x{args.imgsz}, batch 1\n",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    Path("reports/benchmark.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    if os.environ.get("WANDB_API_KEY"):
        import wandb
        run = wandb.init(project="aiml04-perception", job_type="benchmark", config={"gpu": gpu, "imgsz": args.imgsz})
        run.log({"benchmark": wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])})
        run.finish()


if __name__ == "__main__":
    main()
