"""Detector accuracy on the unseen test town, comparable across label spaces (COCO vs CARLA-tuned).

  python -m perception.eval_detection --seq data/carla/Town02_test \
      --model coco=yolo11s.pt --model carla=models/yolo11s_carla.pt

CARLA's ground truth only distinguishes pedestrians from vehicles, so both the ground truth and every detector's
classes are collapsed onto {pedestrian, vehicle} before matching (COCO "person" and "car/truck/bus/motorcycle" count;
traffic lights/signs are ignored). Reported per model:
  * mAP50-95 and mAP50 (COCO-style: 101-point interpolation, greedy matching by confidence) with a 95 % bootstrap CI
    over frames (frames are resampled, not boxes, because boxes within a frame are correlated);
  * recall at the operating confidence (the one the pipeline uses), split by ground-truth distance band, which is
    what the AEB depends on (a pedestrian missed at 15 m matters more than one missed at 70 m).
This script is meant to be run once per model on the test town; nothing here feeds back into training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EVAL = {"pedestrian": "pedestrian", "person": "pedestrian", "rider": "pedestrian",
        "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "vehicle", "motorbike": "vehicle"}
CLASSES = ["pedestrian", "vehicle"]
IOUS = np.linspace(0.5, 0.95, 10)
BANDS = [(0, 15), (15, 30), (30, 50), (50, 1e9)]


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match_frame(pb, pc, ps, gb, gc) -> np.ndarray:
    """Greedy COCO matching per class and IoU threshold. Returns tp[n_pred, 10]."""
    tp = np.zeros((len(pb), len(IOUS)), bool)
    for c in range(len(CLASSES)):
        pi, gi = np.where(pc == c)[0], np.where(gc == c)[0]
        if len(pi) == 0 or len(gi) == 0:
            continue
        pi = pi[np.argsort(-ps[pi])]
        iou = box_iou(pb[pi], gb[gi])
        for t, thr in enumerate(IOUS):
            used = np.zeros(len(gi), bool)
            for k in range(len(pi)):
                cand = np.where(~used & (iou[k] >= thr))[0]
                if len(cand):
                    j = cand[np.argmax(iou[k, cand])]
                    used[j] = True
                    tp[pi[k], t] = True
    return tp


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> np.ndarray:
    """AP per IoU threshold (101-point interpolated)."""
    if n_gt == 0:
        return np.full(len(IOUS), np.nan)
    if len(conf) == 0:
        return np.zeros(len(IOUS))
    order = np.argsort(-conf)
    tpc = np.cumsum(tp[order], 0)
    fpc = np.cumsum(~tp[order], 0)
    rec = tpc / n_gt
    prec = tpc / (tpc + fpc)
    out = []
    for t in range(len(IOUS)):
        p = np.maximum.accumulate(prec[::-1, t])[::-1]
        r = rec[:, t]
        idx = np.searchsorted(r, np.linspace(0, 1, 101), side="left")
        out.append(np.mean([p[i] if i < len(p) else 0.0 for i in idx]))
    return np.array(out)


def map_from_frames(frames: list[dict]) -> tuple[float, float, dict]:
    aps = {}
    for c in range(len(CLASSES)):
        tp = [f["tp"][f["pc"] == c] for f in frames]
        conf = [f["ps"][f["pc"] == c] for f in frames]
        n_gt = int(sum((f["gc"] == c).sum() for f in frames))
        aps[CLASSES[c]] = average_precision(np.concatenate(tp) if tp else np.zeros((0, 10), bool),
                                            np.concatenate(conf) if conf else np.zeros(0), n_gt)
    stack = np.array([a for a in aps.values() if not np.isnan(a).all()])
    return float(stack.mean()), float(stack[:, 0].mean()), {k: float(v.mean()) for k, v in aps.items()}


def predict_sequence(seq: Path, weights: str, imgsz: int, min_height: int, op_conf: float) -> list[dict]:
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights, task="detect")
    frames = []
    for line in (seq / "gt.jsonl").read_text().splitlines():
        g = json.loads(line)
        objs = [o for o in g["objects"] if o["cls"] in EVAL and o["box"][3] - o["box"][1] >= min_height]
        gb = np.array([o["box"] for o in objs], float).reshape(-1, 4)
        gc = np.array([CLASSES.index(EVAL[o["cls"]]) for o in objs], int)
        gd = np.array([o.get("distance_m", np.nan) for o in objs], float)
        img = cv2.imread(str(seq / "rgb" / f"{g['frame']:06d}.jpg"))
        r = model.predict(img, imgsz=imgsz, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy()
        names = [model.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
        keep = np.array([n in EVAL for n in names], bool)
        pb = b[keep] if len(b) else np.zeros((0, 4))
        pc = np.array([CLASSES.index(EVAL[n]) for n, k in zip(names, keep) if k], int)
        ps = r.boxes.conf.cpu().numpy()[keep] if len(b) else np.zeros(0)
        tp = match_frame(pb, pc, ps, gb, gc)
        # recall at the operating point, per GT object (IoU >= 0.5, same class, confidence >= op_conf)
        hit = np.zeros(len(gb), bool)
        sel = ps >= op_conf
        if sel.any() and len(gb):
            iou = box_iou(gb, pb[sel])
            same = gc[:, None] == pc[sel][None, :]
            hit = ((iou >= 0.5) & same).any(1)
        frames.append({"tp": tp, "ps": ps, "pc": pc, "gc": gc, "gd": gd, "hit": hit})
    return frames


def summarize(frames: list[dict], n_boot: int, seed: int, op_conf: float) -> dict:
    m, m50, per_class = map_from_frames(frames)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(frames), len(frames))
        boots.append(map_from_frames([frames[i] for i in idx])[:2])
    boots = np.array(boots)
    ci = lambda col: [round(float(np.percentile(boots[:, col], q)), 4) for q in (2.5, 97.5)]  # noqa: E731
    recall = {}
    for c in range(len(CLASSES)):
        for lo, hi in BANDS:
            hits = np.concatenate([f["hit"][(f["gc"] == c) & (f["gd"] >= lo) & (f["gd"] < hi)] for f in frames])
            label = f"{lo}-{int(hi)}m" if hi < 1e9 else f"{lo}m+"
            recall.setdefault(CLASSES[c], {})[label] = {"recall": round(float(hits.mean()), 3) if len(hits) else None,
                                                       "n": int(len(hits))}
    return {"map50_95": round(m, 4), "map50_95_ci95": ci(0), "map50": round(m50, 4), "map50_ci95": ci(1),
            "per_class_map50_95": {k: round(v, 4) for k, v in per_class.items()},
            "operating_conf": op_conf, "recall_by_distance": recall}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="label=weights, repeatable")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-height", type=int, default=6, help="same speck filter as carla_to_yolo")
    ap.add_argument("--conf", type=float, default=0.3, help="operating confidence of the pipeline")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("reports/detection_eval.json"))
    args = ap.parse_args()

    meta = json.loads((args.seq / "meta.json").read_text())
    result = {"sequence": args.seq.name, "town": meta["town"], "weather": meta["weather"],
              "frames": meta["frames"], "classes": CLASSES, "min_height_px": args.min_height, "models": {}}
    for spec in args.model:
        label, weights = spec.split("=", 1)
        frames = predict_sequence(args.seq, weights, args.imgsz, args.min_height, args.conf)
        result["models"][label] = {"weights": weights, **summarize(frames, args.boot, args.seed, args.conf)}
        print(label, json.dumps(result["models"][label]))
    result["gt_objects"] = {c: int(sum((f["gc"] == i).sum() for f in frames)) for i, c in enumerate(CLASSES)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
