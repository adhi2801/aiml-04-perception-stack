"""Fine-tune COCO YOLO11 on CARLA ground truth (data/carla_yolo, built by perception.data.carla_to_yolo).

  python -m perception.train_detector --data data/carla_yolo/data.yaml --epochs 40 --batch 8

Model selection (Ultralytics' best.pt = best fitness epoch) uses only the val split, which is the last 15 % of each
TRAINING town's sequence. The test town (Town02) is never converted into this dataset; it is scored once afterwards
by perception.eval_detection. The run's args, results.csv and weights are copied into models/ and reports/ so the
README numbers trace back to a file.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/carla_yolo/data.yaml")
    ap.add_argument("--base", default="yolo11s.pt", help="COCO-pretrained starting point")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="8 fits a 4 GB GPU at 640")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="yolo11s_carla")
    ap.add_argument("--resume", default=None, help="last.pt of an interrupted run; continues with its saved args")
    args = ap.parse_args()

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        model = YOLO(args.base)
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                    seed=args.seed, deterministic=False, project="runs/detect", name=args.name, exist_ok=True,
                    cos_lr=True, close_mosaic=5, fliplr=0.5, mosaic=1.0, mixup=0.05, hsv_h=0.02, patience=15,
                    plots=True, verbose=False)
    run = Path(model.trainer.save_dir)
    Path("models").mkdir(exist_ok=True)
    shutil.copy2(run / "weights" / "best.pt", Path("models") / f"{args.name}.pt")
    out = Path("reports") / "detector_train"
    out.mkdir(parents=True, exist_ok=True)
    for f in ["results.csv", "args.yaml", "results.png", "confusion_matrix_normalized.png"]:
        if (run / f).exists():
            shutil.copy2(run / f, out / f)
    val = YOLO(Path("models") / f"{args.name}.pt").val(data=args.data, imgsz=args.imgsz, batch=args.batch,
                                                       plots=False, verbose=False)
    names = val.names
    summary = {"split": "val (held-out tail of each training town; used for best-epoch selection)",
               "map50_95": round(float(val.box.map), 4), "map50": round(float(val.box.map50), 4),
               "per_class_map50_95": {names[int(c)]: round(float(m), 4)
                                      for c, m in zip(val.box.ap_class_index, val.box.maps[val.box.ap_class_index])}}
    (out / "val_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
