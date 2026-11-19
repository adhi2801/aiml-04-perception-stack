"""Multi-object tracking metrics (MOTA, IDF1, ID switches) against CARLA ground-truth actor IDs.

  python -m perception.eval_tracking --seq data/carla/Town02_test --weights models/yolo11s_carla.engine \
      --tracker bytetrack.yaml
  python -m perception.eval_tracking --seq data/carla/Town02_test --weights models/yolo11s_carla.engine \
      --tracker botsort.yaml

Comparing trackers on identical detections isolates association quality from detection quality.

The tracker the pipeline uses is chosen on VALIDATION frames (the last --val-tail of each training sequence, the
frames carla_to_yolo held out), by overall IDF1, declared before the run because IDF1 measures association, the only
thing the tracker changes. The test town is then scored once with that choice:

  python -m perception.eval_tracking --val-tail 0.15 --operating-points reports/operating_points.json --label carla \
      --weights models/yolo11s_carla.pt --tracker botsort.yaml --seq data/carla/Town01_sunset ...
  python -m perception.eval_tracking --choose reports/tracking_val_bytetrack.json reports/tracking_val_botsort.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def _motmetrics():
    """Imported on use, so the pure helpers (and their tests) do not need motmetrics installed."""
    if not hasattr(np, "asfarray"):  # motmetrics 1.4.0 (latest release) still calls it; NumPy 2 removed it
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)  # noqa: NPY201 (the shim itself)
    import motmetrics

    return motmetrics


# CARLA GT only knows these; map detector classes onto them for matching. COCO says "person"/"motorbike",
# BDD100K says "pedestrian"/"motorcycle": both must count, or every COCO pedestrian is silently dropped.
EVAL_CLASSES = {
    "pedestrian": "pedestrian",
    "person": "pedestrian",
    "rider": "pedestrian",
    "car": "car",
    "truck": "car",
    "bus": "car",
    "motorcycle": "car",
    "motorbike": "car",
    "bicycle": "car",
}


def xyxy_to_xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


TEST_TOWN = "Town02"


def val_frames(lines: list[str], val_tail: float | None) -> list[str]:
    """The last `val_tail` of a sequence (contiguous, so tracks stay intact); everything when val_tail is None."""
    return lines[int(len(lines) * (1 - val_tail)) :] if val_tail else lines


# Added after the first validation run showed BoT-SORT's step at 47.8 ms p95 (camera-motion compensation): the
# camera runs at 20 Hz, so a 50 ms frame cannot spend most of it on association. Decided on validation timings only.
LATENCY_BUDGET_P95_MS = 10.0


def choose_tracker(reports: dict[str, dict], budget_p95_ms: float | None = LATENCY_BUDGET_P95_MS) -> dict:
    """Highest overall IDF1 on validation among trackers whose per-frame step fits the latency budget (p95);
    ties go to fewer ID switches. reports: tracker name -> report."""
    for name, r in reports.items():
        if not r.get("_meta", {}).get("val_tail"):
            raise ValueError(f"{name}: not a validation report; the tracker must be chosen on validation frames")

    def p95(n: str) -> float | None:
        return (reports[n]["_meta"].get("association_ms") or {}).get("p95")

    eligible = [n for n in reports if budget_p95_ms is None or (p95(n) is not None and p95(n) <= budget_p95_ms)]
    if not eligible:
        raise ValueError(f"no tracker fits the {budget_p95_ms} ms p95 budget")
    ranked = sorted(eligible, key=lambda n: (-reports[n]["OVERALL"]["idf1"], reports[n]["OVERALL"]["num_switches"]))
    best_unconstrained = max(reports, key=lambda n: reports[n]["OVERALL"]["idf1"])
    return {
        "chosen": ranked[0],
        "criterion": "highest overall IDF1 on validation frames (ties: fewer ID switches)"
        + (f" among trackers with a per-frame step p95 <= {budget_p95_ms} ms" if budget_p95_ms is not None else ""),
        "budget_added_after_first_run": budget_p95_ms is not None,
        "best_ignoring_budget": best_unconstrained,
        "idf1": {n: round(reports[n]["OVERALL"]["idf1"], 4) for n in reports},
        "mota": {n: round(reports[n]["OVERALL"]["mota"], 4) for n in reports},
        "id_switches": {n: int(reports[n]["OVERALL"]["num_switches"]) for n in reports},
        "step_p95_ms": {n: p95(n) for n in reports},
        "sequences": reports[ranked[0]]["_meta"]["sequences"],
        "conf": reports[ranked[0]]["_meta"]["conf"],
    }


def run(
    seq: Path,
    weights: str,
    tracker: str,
    imgsz: int,
    conf: float,
    min_height: int,
    device: str | None = None,
    val_tail: float | None = None,
):
    from ultralytics import YOLO

    mm = _motmetrics()
    model = YOLO(weights, task="detect")  # a fresh model per sequence, so no track survives across sequences
    acc = mm.MOTAccumulator(auto_id=True)
    timings: list[float] = []
    gts = [json.loads(line) for line in val_frames((seq / "gt.jsonl").read_text().splitlines(), val_tail)]
    for g in gts:
        img = cv2.imread(str(seq / "rgb" / f"{g['frame']:06d}.jpg"))
        t0 = time.perf_counter()
        r = model.track(img, imgsz=imgsz, conf=conf, tracker=tracker, persist=True, device=device, verbose=False)[0]
        total_ms = (time.perf_counter() - t0) * 1000
        # Ultralytics times preprocess + inference + NMS itself; the rest of the call is the tracker's association
        # (and, for BoT-SORT, its camera-motion compensation), which is the cost this comparison is about.
        timings.append(max(total_ms - sum(r.speed.values()), 0.0))
        hyp_ids, hyp_boxes = [], []
        if r.boxes is not None and r.boxes.id is not None:
            for b, c, tid in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.id.cpu().numpy().astype(int),
                strict=True,
            ):
                if model.names[c] in EVAL_CLASSES and b[3] - b[1] >= min_height:
                    hyp_ids.append(int(tid))
                    hyp_boxes.append(xyxy_to_xywh(b))
        gt = [o for o in g["objects"] if o["box"][3] - o["box"][1] >= min_height]
        dist = mm.distances.iou_matrix([xyxy_to_xywh(o["box"]) for o in gt], hyp_boxes, max_iou=0.5)
        acc.update([o["id"] for o in gt], hyp_ids, dist)
    return acc, timings[1:]  # the first frame includes warm-up


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--choose", nargs="+", type=Path, help="validation reports to choose the tracker from")
    ap.add_argument("--seq", nargs="+", type=Path)
    ap.add_argument("--weights")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument(
        "--min-height", type=int, default=20, help="ignore tiny far-away objects in both GT and predictions"
    )
    ap.add_argument("--device", help="e.g. cpu or 0; default: ultralytics picks")
    ap.add_argument("--val-tail", type=float, help="score only the held-out last fraction of each training sequence")
    ap.add_argument("--operating-points", type=Path, help="reports/operating_points.json (overrides --conf)")
    ap.add_argument("--label", default="carla", help="which model's operating point to use")
    args = ap.parse_args()

    if args.choose:
        loaded = [json.loads(p.read_text()) for p in args.choose]
        reports = {Path(r["_meta"]["tracker"]).name: r for r in loaded}
        choice = choose_tracker(reports)
        print(json.dumps(choice, indent=2))
        Path("reports/tracker_choice.json").write_text(json.dumps(choice, indent=2))
        return
    if not args.seq or not args.weights:
        ap.error("--seq and --weights are required unless --choose")
    if args.val_tail and any(TEST_TOWN in s.name for s in args.seq):
        raise SystemExit(f"{TEST_TOWN} is the test town: the tracker is chosen on training-town validation frames")
    if args.operating_points:
        args.conf = json.loads(args.operating_points.read_text())["models"][args.label]["conf"]

    mm = _motmetrics()
    runs = [
        run(s, args.weights, args.tracker, args.imgsz, args.conf, args.min_height, args.device, args.val_tail)
        for s in args.seq
    ]
    accs = [a for a, _ in runs]
    assoc_ms = np.array([t for _, ts in runs for t in ts])
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        names=[s.name for s in args.seq],
        generate_overall=True,
        metrics=[
            "mota",
            "motp",
            "idf1",
            "num_switches",
            "mostly_tracked",
            "mostly_lost",
            "num_false_positives",
            "num_misses",
            "precision",
            "recall",
        ],
    )
    print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names))
    Path("reports").mkdir(exist_ok=True)
    out = Path("reports") / f"tracking_{'val_' if args.val_tail else ''}{Path(args.tracker).stem}.json"
    result = {
        k: {m: (None if np.isnan(v) else float(v)) for m, v in row.items()}
        for k, row in summary.to_dict(orient="index").items()
    }
    result["_meta"] = {
        "weights": args.weights,
        "tracker": args.tracker,
        "conf": args.conf,
        "conf_source": "validation" if args.operating_points else "fixed",
        "min_height": args.min_height,
        "val_tail": args.val_tail,
        "sequences": [s.name for s in args.seq],
        "association_ms": {
            "p50": round(float(np.percentile(assoc_ms, 50)), 2),
            "p95": round(float(np.percentile(assoc_ms, 95)), 2),
            "frames": len(assoc_ms),
            "note": "per-frame time of the tracker step alone (track() minus Ultralytics pre/infer/NMS times)",
        },
    }
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
