"""BDD100K detection labels (det_20 JSON) -> YOLO format, plus the class-balance analysis.

  python -m perception.data.bdd_to_yolo --bdd-root /path/bdd100k --out data/bdd_yolo --train-limit 20000

Writes: images symlinked/copied, labels/*.txt, data.yaml, reports/bdd_stats.json, reports/figures/class_balance.png
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]
# Older BDD JSON uses different names for some classes.
ALIASES = {"person": "pedestrian", "motor": "motorcycle", "bike": "bicycle"}
W, H = 1280, 720


def convert(frames: list[dict], img_dir: Path, out: Path, split: str, link: bool) -> dict:
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    counts, small, per_img, attrs = Counter(), Counter(), [], Counter()
    for f in frames:
        src = img_dir / f["name"]
        if not src.exists():
            continue
        dst = out / "images" / split / f["name"]
        if not dst.exists():
            if link:
                try:
                    os.symlink(src, dst)
                except OSError:  # Windows without symlink rights
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
        lines = []
        for lab in f.get("labels") or []:
            cat = ALIASES.get(lab.get("category"), lab.get("category"))
            if cat not in CLASSES or "box2d" not in lab:
                continue
            b = lab["box2d"]
            x1, y1, x2, y2 = max(0, b["x1"]), max(0, b["y1"]), min(W, b["x2"]), min(H, b["y2"])
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            counts[cat] += 1
            if (x2 - x1) * (y2 - y1) < 32 * 32:  # COCO "small" object
                small[cat] += 1
            lines.append(f"{CLASSES.index(cat)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        (out / "labels" / split / (Path(f["name"]).stem + ".txt")).write_text("\n".join(lines))
        per_img.append(len(lines))
        a = f.get("attributes", {})
        attrs[f"timeofday={a.get('timeofday', '?')}"] += 1
        attrs[f"weather={a.get('weather', '?')}"] += 1
    return {"images": len(per_img), "boxes": sum(counts.values()), "per_class": dict(counts),
            "small_frac": {c: round(small[c] / counts[c], 3) for c in counts},
            "mean_boxes_per_image": round(sum(per_img) / max(len(per_img), 1), 2), "attributes": dict(attrs)}


def plot_balance(stats: dict, path: Path) -> None:
    pc = stats["train"]["per_class"]
    order = sorted(pc, key=pc.get, reverse=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].bar(order, [pc[c] for c in order], color="#3b7dd8")
    ax[0].set_yscale("log")
    ax[0].set_title("BDD100K train: boxes per class (log scale)")
    ax[0].tick_params(axis="x", rotation=45)
    sf = stats["train"]["small_frac"]
    ax[1].bar(order, [sf.get(c, 0) for c in order], color="#e07b39")
    ax[1].set_title("Fraction of boxes smaller than 32x32 px")
    ax[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd-root", type=Path, required=True,
                    help="folder with images/100k/{train,val} and labels/det_20/det_{train,val}.json")
    ap.add_argument("--out", type=Path, default=Path("data/bdd_yolo"))
    ap.add_argument("--train-limit", type=int, default=20000, help="subset size that fits a free GPU budget")
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stats = {}
    for split, limit in (("train", args.train_limit), ("val", args.val_limit)):
        label_file = args.bdd_root / "labels" / "det_20" / f"det_{split}.json"
        frames = json.loads(label_file.read_text())
        random.Random(args.seed).shuffle(frames)
        stats[split] = convert(frames[:limit], args.bdd_root / "images" / "100k" / split, args.out, split, not args.copy)
        print(split, json.dumps({k: stats[split][k] for k in ("images", "boxes", "mean_boxes_per_image")}))

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/bdd_stats.json").write_text(json.dumps(stats, indent=2))
    plot_balance(stats, Path("reports/figures/class_balance.png"))


if __name__ == "__main__":
    main()
