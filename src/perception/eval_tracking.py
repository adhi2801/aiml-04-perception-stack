"""Multi-object tracking metrics (MOTA, IDF1, ID switches) against CARLA ground-truth actor IDs.

  python -m perception.eval_tracking --seq data/carla/Town02_test --weights models/yolo11s_bdd.engine --tracker bytetrack.yaml
  python -m perception.eval_tracking --seq data/carla/Town02_test --weights models/yolo11s_bdd.engine --tracker botsort.yaml

Comparing trackers on identical detections isolates association quality from detection quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

if not hasattr(np, "asfarray"):  # motmetrics 1.4.0 (latest release) still calls it; NumPy 2 removed it
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
import motmetrics as mm  # noqa: E402

# CARLA GT only knows these; map detector classes onto them for matching. COCO says "person"/"motorbike",
# BDD100K says "pedestrian"/"motorcycle": both must count, or every COCO pedestrian is silently dropped.
EVAL_CLASSES = {"pedestrian": "pedestrian", "person": "pedestrian", "rider": "pedestrian", "car": "car",
                "truck": "car", "bus": "car", "motorcycle": "car", "motorbike": "car", "bicycle": "car"}


def xyxy_to_xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


def run(seq: Path, weights: str, tracker: str, imgsz: int, conf: float, min_height: int,
        device: str | None = None) -> mm.MOTAccumulator:
    from ultralytics import YOLO

    model = YOLO(weights, task="detect")
    acc = mm.MOTAccumulator(auto_id=True)
    gts = [json.loads(line) for line in (seq / "gt.jsonl").read_text().splitlines()]
    for g in gts:
        img = cv2.imread(str(seq / "rgb" / f"{g['frame']:06d}.jpg"))
        r = model.track(img, imgsz=imgsz, conf=conf, tracker=tracker, persist=True, device=device, verbose=False)[0]
        hyp_ids, hyp_boxes = [], []
        if r.boxes is not None and r.boxes.id is not None:
            for b, c, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                                 r.boxes.id.cpu().numpy().astype(int)):
                if model.names[c] in EVAL_CLASSES and b[3] - b[1] >= min_height:
                    hyp_ids.append(int(tid))
                    hyp_boxes.append(xyxy_to_xywh(b))
        gt = [o for o in g["objects"] if o["box"][3] - o["box"][1] >= min_height]
        dist = mm.distances.iou_matrix([xyxy_to_xywh(o["box"]) for o in gt], hyp_boxes, max_iou=0.5)
        acc.update([o["id"] for o in gt], hyp_ids, dist)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="+", type=Path, required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--min-height", type=int, default=20, help="ignore tiny far-away objects in both GT and predictions")
    ap.add_argument("--device", help="e.g. cpu or 0; default: ultralytics picks")
    args = ap.parse_args()

    accs = [run(s, args.weights, args.tracker, args.imgsz, args.conf, args.min_height, args.device) for s in args.seq]
    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=[s.name for s in args.seq], generate_overall=True,
                              metrics=["mota", "motp", "idf1", "num_switches", "mostly_tracked", "mostly_lost",
                                       "num_false_positives", "num_misses", "precision", "recall"])
    print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names))
    Path("reports").mkdir(exist_ok=True)
    out = Path("reports") / f"tracking_{Path(args.tracker).stem}.json"
    out.write_text(json.dumps({k: {m: (None if np.isnan(v) else float(v)) for m, v in row.items()}
                               for k, row in summary.to_dict(orient="index").items()}, indent=2))


if __name__ == "__main__":
    main()
