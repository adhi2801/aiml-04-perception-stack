"""Where does quantisation lose accuracy? Per-class mAP of each runtime on the val split.

  python -m perception.optimize.precision_report --data data/carla_yolo/data.yaml \
      --model fp32=models/yolo11s_carla.pt --model fp16=models/yolo11s_carla.engine \
      --model int8=models/yolo11s_carla_int8.engine

The benchmark reports one mAP per runtime; an average can hide a class that collapses. This scores every model on
the same val images (never the test town) and writes the per-class mAP50-95 and the drop of each class against the
first model, so a fix (for example keeping the detection head in FP16) can be aimed at the class that pays.
Writes reports/precision_report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def per_class_drop(per_model: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """mAP points lost per class by every model against the first one (positive = worse)."""
    ref_name = next(iter(per_model))
    ref = per_model[ref_name]
    return {
        name: {c: round(100 * (ref[c] - scores[c]), 2) for c in ref if c in scores}
        for name, scores in per_model.items()
        if name != ref_name
    }


def main() -> None:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", action="append", required=True, help="label=weights, repeatable; first = reference")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, default=Path("reports/precision_report.json"))
    args = ap.parse_args()

    per_model: dict[str, dict[str, float]] = {}
    overall: dict[str, float] = {}
    instances: dict[str, int] = {}
    for spec in args.model:
        label, weights = spec.split("=", 1)
        half = weights.endswith(".engine") and "int8" not in weights
        m = YOLO(weights, task="detect")
        val = m.val(data=args.data, imgsz=args.imgsz, half=half, batch=1, device=0, verbose=False, plots=False)
        names = val.names
        per_model[label] = {
            names[int(c)]: round(float(v), 4)
            for c, v in zip(val.box.ap_class_index, val.box.maps[val.box.ap_class_index], strict=True)
        }
        overall[label] = round(float(val.box.map), 4)
        if not instances:
            counts = val.nt_per_class
            instances = {names[int(c)]: int(counts[int(c)]) for c in val.box.ap_class_index}
        print(label, overall[label], per_model[label], flush=True)

    report = {
        "split": "val (held-out tail of each training town)",
        "instances": instances,
        "map50_95": overall,
        "per_class_map50_95": per_model,
        "per_class_drop_pts": per_class_drop(per_model),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["per_class_drop_pts"], indent=2))


if __name__ == "__main__":
    main()
