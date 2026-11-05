"""Record synthetic sequences with perfect labels: RGB, metric depth, lane-line mask, tracked 2D boxes.

  python carla_tools/record_dataset.py --town Town03 --frames 600 --out data/carla/Town03_clear
  python carla_tools/record_dataset.py --town Town10HD_Opt --frames 600 --weather WetCloudyNoon --out data/carla/Town10_test

One folder per sequence:
  rgb/000123.jpg  depth/000123.png (uint16 centimetres)  lanes/000123.png (255 = lane marking)  gt.jsonl
Used for: lane model training/eval, MiDaS scale calibration, depth error by distance, MOT metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import carla
import cv2
import numpy as np

from carla_common import CamSpec, attach_camera, depth_to_meters, get_frame, gt_boxes, rgb_to_bgr, spawn_ego, sync_world

ROAD_LINE_TAG = 24  # CARLA >= 0.9.14 semantic tag for lane markings (it was 6 in older releases)


def spawn_traffic(client, world, tm, actors, n_vehicles: int, n_walkers: int, seed: int):
    rng = np.random.default_rng(seed)
    bl = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    rng.shuffle(spawns)
    vbps = [b for b in bl.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    batch = [carla.command.SpawnActor(rng.choice(vbps), sp).then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port()))
             for sp in spawns[1:n_vehicles + 1]]
    for r in client.apply_batch_sync(batch, True):
        if not r.error:
            actors.append(world.get_actor(r.actor_id))
    walker_bps = bl.filter("walker.pedestrian.*")
    ctrl_bp = bl.find("controller.ai.walker")
    for _ in range(n_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        w = world.try_spawn_actor(rng.choice(walker_bps), carla.Transform(loc))
        if w is None:
            continue
        actors.append(w)
        c = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
        actors.append(c)
        world.tick()
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(float(rng.uniform(1.0, 2.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
  # 127.0.0.1, not localhost: CARLA listens on IPv4 only and localhost can resolve to ::1
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town03")
    ap.add_argument("--weather", default="ClearNoon", help="any carla.WeatherParameters preset name")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--every", type=int, default=2, help="save every Nth tick (20 FPS sim -> 10 FPS data)")
    ap.add_argument("--vehicles", type=int, default=40)
    ap.add_argument("--walkers", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720, help="use 800x450 with configs/pipeline_lowvram.yaml")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    for sub in ("rgb", "depth", "lanes"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    client = carla.Client(args.host, args.port)
    client.set_timeout(60)
    spec = CamSpec(args.width, args.height)
    with sync_world(client, fps=20, town=args.town) as (world, tm, actors):
        world.set_weather(getattr(carla.WeatherParameters, args.weather))
        ego = spawn_ego(world, actors, spawn_index=0)
        rgb_cam, rgb_q = attach_camera(world, actors, ego, spec, "rgb")
        _, depth_q = attach_camera(world, actors, ego, spec, "depth")
        _, seg_q = attach_camera(world, actors, ego, spec, "semantic")

        spawn_traffic(client, world, tm, actors, args.vehicles, args.walkers, args.seed)
        ego.set_autopilot(True, tm.get_port())
        for _ in range(40):  # let traffic settle
            f = world.tick()
            for q in (rgb_q, depth_q, seg_q):
                get_frame(q, f)

        saved = 0
        with open(args.out / "gt.jsonl", "w") as gt_file:
            tick = 0
            while saved < args.frames:
                frame = world.tick()
                rgb, dep, sem = get_frame(rgb_q, frame), get_frame(depth_q, frame), get_frame(seg_q, frame)
                tick += 1
                if tick % args.every:
                    continue
                bgr = rgb_to_bgr(rgb)
                depth = depth_to_meters(dep)
                tags = np.frombuffer(sem.raw_data, np.uint8).reshape(sem.height, sem.width, 4)[..., 2]
                name = f"{saved:06d}"
                cv2.imwrite(str(args.out / "rgb" / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                # uint16 centimetres in a lossless PNG: ~5-10x smaller than raw float arrays, 1 cm resolution
                cv2.imwrite(str(args.out / "depth" / f"{name}.png"), np.clip(depth * 100, 0, 65535).astype(np.uint16))
                cv2.imwrite(str(args.out / "lanes" / f"{name}.png"), ((tags == ROAD_LINE_TAG) * 255).astype(np.uint8))
                gt_file.write(json.dumps({"frame": saved, "t": round(tick / 20, 3), "objects":
                                          gt_boxes(world, rgb_cam, ego, spec, depth)}) + "\n")
                saved += 1
                if saved % 100 == 0:
                    print(f"{saved}/{args.frames}")
    meta = {"town": args.town, "weather": args.weather, "frames": saved, "camera": vars(spec), "fps": 20 / args.every}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
