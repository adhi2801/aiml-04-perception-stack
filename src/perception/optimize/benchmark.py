"""PyTorch vs ONNX Runtime vs TensorRT (FP16, INT8): latency and accuracy cost of each runtime, on THIS machine.

  python scripts/build_trt.py --onnx models/yolo11s_carla.onnx                    # FP16 engine (once)
  python scripts/build_trt.py --onnx models/yolo11s_carla.onnx --int8             # INT8 engine (once)
  python -m perception.optimize.benchmark --weights models/yolo11s_carla.pt --data data/carla_yolo/data.yaml

The ONNX model and TensorRT engines are looked up next to the weights (<stem>.onnx, <stem>.engine,
<stem>_int8.engine). They are built beforehand by scripts/build_trt.py, because Ultralytics' in-process engine export
peaked near 15 GB of RAM on a 16 GB laptop and was killed. Pass --export to let Ultralytics export missing ones.

For every variant:
  * latency over `--runs` single-image predictions after `--warmup` warm-up calls: end-to-end (what the pipeline
    pays: preprocess + model + NMS) and model-only (Ultralytics' own `inference` timing), as p50 / p95 / mean;
  * accuracy: mAP50-95 and mAP50 on the val split (held-out tail of each training town, never the test town), so
    "what does quantisation cost" is answered without touching the test set.
GPU rows and CPU rows are reported separately: the low-VRAM CARLA config runs the detector on the CPU, so the CPU
number is the one the closed-loop tests actually pay there.
Writes reports/benchmark.json + reports/benchmark.md (environment included) and logs to W&B if WANDB_API_KEY is set.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


def _pct(a: list[float], q: float) -> float:
    return round(float(np.percentile(a, q)), 2)


def time_model(model, imgs: list[np.ndarray], imgsz: int, half: bool, device: str, warmup: int, runs: int) -> dict:
    """End-to-end and model-only latency of single-image predictions."""
    for i in range(warmup):
        model.predict(imgs[i % len(imgs)], imgsz=imgsz, half=half, device=device, verbose=False)
    e2e, infer = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        res = model.predict(imgs[i % len(imgs)], imgsz=imgsz, half=half, device=device, verbose=False)
        e2e.append((time.perf_counter() - t0) * 1000)
        infer.append(float(res[0].speed["inference"]))
    mean = float(np.mean(e2e))
    return {
        "p50_ms": _pct(e2e, 50),
        "p95_ms": _pct(e2e, 95),
        "mean_ms": round(mean, 2),
        "fps": round(1000 / mean, 1),
        "model_p50_ms": _pct(infer, 50),
        "model_p95_ms": _pct(infer, 95),
        "p50_spread_ms": block_spread(e2e),
    }


def block_spread(lat: list[float], blocks: int = 5) -> list[float]:
    """Lowest and highest p50 over consecutive blocks of the timed runs.

    A laptop CPU changes clock speed with heat and power, so one p50 can look precise while a rerun differs by 2x
    (seen here: PyTorch on the CPU measured 331 ms, then 170 ms). A wide spread flags a number that is not stable.
    """
    parts = [p for p in np.array_split(np.asarray(lat), blocks) if len(p)]
    p50s = [float(np.percentile(p, 50)) for p in parts]
    return [round(min(p50s), 2), round(max(p50s), 2)]


def environment(device_name: str) -> dict:
    """Versions and hardware behind the numbers, so a reader can tell whether they apply to their setup."""
    env = {
        "device": device_name,
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
    }
    for p in ("torch", "ultralytics", "tensorrt-cu12", "onnxruntime-gpu", "onnxruntime"):
        with contextlib.suppress(md.PackageNotFoundError):
            env[p] = md.version(p)
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        env["nvidia_driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    return env


def ort_cuda_active(onnx: Path) -> bool:
    """True only if ONNX Runtime really runs this model on CUDA.

    `get_available_providers()` lists CUDA even when its DLLs fail to load; the session then silently runs on the
    CPU (seen here: onnxruntime-gpu 1.31 wants CUDA 13, torch ships CUDA 12, and the "CUDA" row timed 188 ms, the
    same as the CPU row). So open a real session and look at the providers it ended up with.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    ort.set_default_logger_severity(4)
    session = ort.InferenceSession(str(onnx), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return session.get_providers()[0] == "CUDAExecutionProvider"


def resolve_variants(
    weights: Path, imgsz: int, data: str, allow_export: bool, cpu: bool, cuda_check=ort_cuda_active
) -> list[tuple]:
    """(name, path, half, device) for every runtime whose file exists (or, with --export, can be exported)."""
    onnx = weights.with_suffix(".onnx")
    fp16 = weights.with_suffix(".engine")
    int8 = weights.with_name(weights.stem + "_int8.engine")
    if allow_export:
        from ultralytics import YOLO

        base = YOLO(str(weights))
        if not onnx.exists():
            onnx = Path(base.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True))
        if not fp16.exists():
            fp16 = Path(base.export(format="engine", imgsz=imgsz, half=True, workspace=2))
        if not int8.exists():
            Path(base.export(format="engine", imgsz=imgsz, int8=True, data=data, workspace=2)).rename(int8)
    out = [("PyTorch FP32", weights, False, "0")]
    for name, path, half in (
        ("ONNX Runtime FP32 (CUDA)", onnx, False),
        ("TensorRT FP16", fp16, True),
        ("TensorRT INT8", int8, False),
    ):
        if path.exists() and name.startswith("ONNX") and not cuda_check(path):
            print(f"skip {name}: ONNX Runtime fell back to the CPU (CUDA provider failed to load)")
        elif path.exists():
            out.append((name, path, half, "0"))
        else:
            print(f"skip {name}: {path} not found (build it with scripts/build_trt.py, or pass --export)")
    if cpu:
        out.append(("PyTorch FP32 (CPU)", weights, False, "cpu"))
        if onnx.exists():
            out.append(("ONNX Runtime FP32 (CPU)", onnx, False, "cpu"))
    return out


def to_markdown(env: dict, imgsz: int, rows: list[dict]) -> str:
    cols = [
        "variant",
        "size_mb",
        "p50_ms",
        "p95_ms",
        "model_p50_ms",
        "fps",
        "speedup_vs_pytorch_gpu",
        "map50_95",
        "map_drop_pts",
    ]
    lines = [
        f"Device: **{env['device']}** (driver {env.get('nvidia_driver', '?')}), CPU: {env.get('cpu')}, input "
        f"{imgsz}x{imgsz}, batch 1. Latency = end-to-end predict (preprocess + model + NMS); model_p50_ms = model "
        "only. Accuracy on the val split (never the test town).\n",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def add_relative(rows: list[dict]) -> None:
    """Speed-up and mAP drop of each row against the first row (PyTorch FP32 on the GPU)."""
    ref = rows[0]
    for r in rows:
        r["speedup_vs_pytorch_gpu"] = round(ref["mean_ms"] / r["mean_ms"], 2)
        r["map_drop_pts"] = round(100 * (ref["map50_95"] - r["map50_95"]), 2)


def main() -> None:
    import cv2
    import torch
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--images", type=int, default=64, help="val images cycled through for timing")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU rows")
    ap.add_argument("--export", action="store_true", help="let Ultralytics export missing ONNX/engines (RAM heavy)")
    ap.add_argument("--out", default="reports/benchmark")
    args = ap.parse_args()

    img_paths = sorted((Path(args.data).parent / "images" / "val").glob("*.jpg"))[: args.images]
    if not img_paths:
        raise SystemExit(f"no val images next to {args.data}")
    imgs = [cv2.imread(str(p)) for p in img_paths]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    rows = []
    for name, path, half, device in resolve_variants(args.weights, args.imgsz, args.data, args.export, not args.no_cpu):
        on_cpu = device == "cpu"
        m = YOLO(str(path), task="detect")
        runs = max(50, args.runs // 3) if on_cpu else args.runs
        speed = time_model(m, imgs, args.imgsz, half, device, 10 if on_cpu else args.warmup, runs)
        val = m.val(data=args.data, imgsz=args.imgsz, half=half, batch=1, device=device, verbose=False, plots=False)
        row = {
            "variant": name,
            "device": "cpu" if on_cpu else "gpu",
            "file": path.name,
            "size_mb": round(path.stat().st_size / 2**20, 1),
            "timed_runs": runs,
            **speed,
            "map50_95": round(float(val.box.map), 4),
            "map50": round(float(val.box.map50), 4),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    add_relative(rows)
    env = environment(gpu)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps({"env": env, "imgsz": args.imgsz, "batch": 1, "accuracy_split": "val", "rows": rows}, indent=2)
    )
    table = to_markdown(env, args.imgsz, rows)
    out.with_suffix(".md").write_text(table)
    print(table)

    if os.environ.get("WANDB_API_KEY"):
        import wandb

        run = wandb.init(project="aiml04-perception", job_type="benchmark", config=env)
        run.log({"benchmark": wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])})
        run.finish()


if __name__ == "__main__":
    main()
