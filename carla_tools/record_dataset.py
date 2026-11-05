"""Record synthetic sequences with perfect labels: RGB, metric depth, lane-line mask, tracked 2D boxes.

  python carla_tools/record_dataset.py --town Town03 --frames 600 --out data/carla/Town03_clear
  python carla_tools/record_dataset.py --town Town02_Opt --frames 600 --weather WetCloudySunset --out data/carla/Town02_test

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


def ahead_dist(origin: carla.Transform, loc: carla.Location) -> float:
    """Distance for ordering spawns around the ego: anything behind it counts as 100 m further away."""
    fwd, to = origin.get_forward_vector(), loc - origin.location
    return loc.distance(origin.location) + (100.0 if to.x * fwd.x + to.y * fwd.y <= 0 else 0.0)


def busiest_spawn(world) -> int:
    """Index of the spawn point with the most other spawn points within 80 m ahead: a busy street, not a map edge."""
    sps = world.get_map().get_spawn_points()
    return int(np.argmax([sum(ahead_dist(sp, o.location) < 80 for o in sps) for sp in sps]))


def spawn_traffic(client, world, tm, actors, n_vehicles: int, n_walkers: int, seed: int, near: carla.Transform):
    """Traffic and pedestrians around `near` (the ego), ahead of it first, so the camera actually sees them."""
    rng = np.random.default_rng(seed)
    bl = world.get_blueprint_library()
    spawns = sorted(world.get_map().get_spawn_points(), key=lambda sp: ahead_dist(near, sp.location))
    vbps = [b for b in bl.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    batch = [carla.command.SpawnActor(rng.choice(vbps), sp).then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port()))
             for sp in spawns[:n_vehicles] if sp.location.distance(near.location) > 5]
    for r in client.apply_batch_sync(batch, True):
        if not r.error:
            actors.append(world.get_actor(r.actor_id))
    walker_bps = bl.filter("walker.pedestrian.*")
    ctrl_bp = bl.find("controller.ai.walker")
    # sidewalk points sampled map-wide, nearest first: most land within ~150 m of the ego's start
    candidates = [world.get_random_location_from_navigation() for _ in range(n_walkers * 15)]
    candidates = sorted((c for c in candidates if c is not None), key=lambda c: ahead_dist(near, c))
    for loc in candidates[:n_walkers]:
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
    ap.add_argument("--spawn", type=int, default=-1, help="ego spawn point index; -1 = busiest street ahead")
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
        spawn = busiest_spawn(world) if args.spawn < 0 else args.spawn
        ego = spawn_ego(world, actors, spawn_index=spawn)
        rgb_cam, rgb_q = attach_camera(world, actors, ego, spec, "rgb")
        _, depth_q = attach_camera(world, actors, ego, spec, "depth")
        _, seg_q = attach_camera(world, actors, ego, spec, "semantic")

        # the spawn point, not ego.get_transform(): in sync mode a new actor reports (0, 0, 0) until the next tick
        start = world.get_map().get_spawn_points()[spawn]
        spawn_traffic(client, world, tm, actors, args.vehicles, args.walkers, args.seed, start)
        ego.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(ego, 100)  # keep moving: frames parked at a red light add little
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
    meta = {"town": args.town, "weather": args.weather, "frames": saved, "ego_spawn": spawn, "camera": vars(spec), "fps": 20 / args.every}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
