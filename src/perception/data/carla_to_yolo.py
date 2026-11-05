"""CARLA ground truth (record_dataset.py's gt.jsonl) -> YOLO detection dataset, for fine-tuning the COCO detector.

  python -m perception.data.carla_to_yolo --seq data/carla/Town03_clear data/carla/Town03_rain --out data/carla_yolo

Split is temporal: the last --val-frac of every sequence is validation, so near-duplicate neighbouring frames
never straddle train and val. The test town (Town02) is never converted; it stays for eval_tracking only.
Class ids follow bdd_to_yolo.CLASSES, so a CARLA- or BDD-tuned model plugs into the same pipeline names.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from perception.data.bdd_to_yolo import CLASSES


def convert(seqs: list[Path], out: Path, val_frac: float, min_px: int) -> dict:
    counts = {"train": Counter(), "val": Counter()}
    for seq in seqs:
        meta = json.loads((seq / "meta.json").read_text())
        w, h = meta["camera"]["width"], meta["camera"]["height"]
        rows = [json.loads(line) for line in (seq / "gt.jsonl").read_text().splitlines()]
        cut = int(len(rows) * (1 - val_frac))
        for i, row in enumerate(rows):
            split = "train" if i < cut else "val"
            name = f"{seq.name}_{row['frame']:06d}"
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(seq / "rgb" / f"{row['frame']:06d}.jpg", out / "images" / split / f"{name}.jpg")
            lines = []
            for o in row["objects"]:
                x1, y1, x2, y2 = o["box"]
                if o["cls"] not in CLASSES or x2 - x1 < min_px or y2 - y1 < min_px:
                    continue
                counts[split][o["cls"]] += 1
                lines.append(f"{CLASSES.index(o['cls'])} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    return {k: dict(v) for k, v in counts.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/carla_yolo"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--min-px", type=int, default=6, help="drop boxes thinner than this (unlearnable specks)")
    args = ap.parse_args()
    print(json.dumps(convert(args.seq, args.out, args.val_frac, args.min_px), indent=2))


if __name__ == "__main__":
    main()
