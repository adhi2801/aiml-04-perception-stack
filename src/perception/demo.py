"""Run the whole stack (detect + track + distance + lanes + AEB) on a recorded drive, no CARLA or ROS needed.

  python scripts/fetch_assets.py                     # weights + the Town02 test drive (GitHub Release)
  python -m perception.demo --seq data/carla/Town02_test --out reports/demo_town02.mp4 --gif reports/demo_town02.gif

The camera geometry (resolution, FOV, mount height) is taken from the sequence's meta.json, so ground-plane distances
are correct for the recording. Writes the annotated video, an optional GIF excerpt, and per-frame timings
(reports/demo_timings.json). AEB decisions are shown but, of course, cannot act on a recording.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from perception.pipeline import PerceptionPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=Path, default=Path("data/carla/Town02_test"))
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--weights", help="detector weights; default: configs/pipeline.yaml")
    ap.add_argument("--out", type=Path, default=Path("reports/demo_town02.mp4"))
    ap.add_argument("--gif", type=Path, help="also write a GIF excerpt")
    ap.add_argument("--gif-start", type=int, default=0)
    ap.add_argument("--gif-frames", type=int, default=120)
    ap.add_argument("--gif-width", type=int, default=640)
    ap.add_argument("--device", help="cpu or 0 (default: detector config)")
    args = ap.parse_args()

    meta = json.loads((args.seq / "meta.json").read_text())
    cam = meta["camera"]
    over = {"camera": {"width": cam["width"], "height": cam["height"], "fov_deg": cam["fov"],
                       "mount_height_m": cam["z"]}}
    if args.device:
        over["detector"] = {"device": args.device, "half": args.device != "cpu"}
    pipe = PerceptionPipeline(args.config, use_depth_model=False, weights=args.weights, overrides=over)
    fps = meta.get("fps", 10.0)
    frames = sorted((args.seq / "rgb").glob("*.jpg"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cam["width"], cam["height"]))
    gif = []
    timings, actions = [], {}
    for i, p in enumerate(frames):
        bgr = cv2.imread(str(p))
        res = pipe(bgr, t=i / fps)
        vis = pipe.draw(bgr, res)
        writer.write(vis)
        timings.append(res.timings_ms)
        actions[res.action.name] = actions.get(res.action.name, 0) + 1
        if args.gif and args.gif_start <= i < args.gif_start + args.gif_frames:
            h = int(vis.shape[0] * args.gif_width / vis.shape[1])
            gif.append(cv2.cvtColor(cv2.resize(vis, (args.gif_width, h), interpolation=cv2.INTER_AREA),
                                    cv2.COLOR_BGR2RGB))
    writer.release()
    if gif:
        from PIL import Image
        imgs = [Image.fromarray(f).quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in gif]
        imgs[0].save(args.gif, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    steady = timings[10:] or timings
    summary = {"sequence": args.seq.name, "frames": len(frames), "lane_source": pipe.lane_source,
               "weights": args.weights or pipe.cfg["detector"]["weights"],
               "ms_p50": {k: round(float(np.median([t[k] for t in steady])), 1) for k in steady[0]},
               "actions": actions}
    Path("reports/demo_timings.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
