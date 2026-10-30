"""Closed-loop AEB evaluation in CARLA (no ROS, for repeatable batch runs and the demo video).

The ego drives on Traffic Manager autopilot, which is told to IGNORE pedestrians and vehicles,
so the only thing that can stop the car is our perception stack.

  python carla_tools/run_scenarios.py --scenario pedestrian_crossing --trials 10 --video reports/demo_ped.mp4
  python carla_tools/run_scenarios.py --scenario lead_vehicle_brake --trials 10
  python carla_tools/run_scenarios.py --scenario pedestrian_crossing --oracle-depth   # isolates depth error

Per trial we log: collided?, minimum gap to the hazard, time from hazard visible to first BRAKE, and FPS.
Aggregates go to reports/scenarios_<name>.json (and W&B if WANDB_API_KEY is set).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import carla
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carla_common import (CamSpec, attach_camera, attach_collision, depth_to_meters, get_frame, gt_boxes,  # noqa: E402
                          rgb_to_bgr, spawn_ego, sync_world)

from perception.fusion.aeb import Action  # noqa: E402
from perception.pipeline import PerceptionPipeline  # noqa: E402


def ahead(transform: carla.Transform, meters: float, lateral: float = 0.0) -> carla.Location:
    f, r = transform.get_forward_vector(), transform.get_right_vector()
    loc = transform.location
    return carla.Location(loc.x + f.x * meters + r.x * lateral, loc.y + f.y * meters + r.y * lateral, loc.z + 1.0)


def setup_hazard(world, actors, ego, scenario: str, rng: random.Random):
    """Returns (hazard actor, per-tick function that drives it)."""
    wp = world.get_map().get_waypoint(ego.get_location())
    trig = rng.uniform(32, 45)
    hazard_wp = wp.next(trig + 25)[0]
    bl = world.get_blueprint_library()
    if scenario == "pedestrian_crossing":
        side = rng.choice([-1, 1])
        walker = world.spawn_actor(rng.choice(bl.filter("walker.pedestrian.*")),
                                   carla.Transform(ahead(hazard_wp.transform, 0, side * 4.5)))
        actors.append(walker)
        direction = hazard_wp.transform.get_right_vector() * (-side)
        speed = rng.uniform(1.2, 2.0)

        def tick(_):
            if ego.get_location().distance(walker.get_location()) < trig:
                walker.apply_control(carla.WalkerControl(direction=direction, speed=speed))
        return walker, tick

    if scenario == "lead_vehicle_brake":
        lead = world.spawn_actor(bl.find("vehicle.audi.a2"), carla.Transform(
            hazard_wp.transform.location + carla.Location(z=0.5), hazard_wp.transform.rotation))
        actors.append(lead)
        state = {"t": 0}

        def tick(_):
            state["t"] += 1
            if state["t"] < 60:  # roll forward, then slam the brakes
                lead.apply_control(carla.VehicleControl(throttle=0.4))
            else:
                lead.apply_control(carla.VehicleControl(brake=1.0))
        return lead, tick
    raise ValueError(scenario)


def run_trial(client, pipe: PerceptionPipeline, args, trial: int, writer) -> dict:
    rng = random.Random(trial)
    spec = CamSpec(pipe.w, pipe.h, pipe.fov, z=pipe.cam_h)
    with sync_world(client, fps=args.fps) as (world, tm, actors):
        ego = spawn_ego(world, actors, spawn_index=args.spawn + trial % 5)
        rgb_cam, rgb_q = attach_camera(world, actors, ego, spec, "rgb")
        _, depth_q = attach_camera(world, actors, ego, spec, "depth")
        collisions = attach_collision(world, actors, ego)
        ego.set_autopilot(True, tm.get_port())
        tm.ignore_walkers_percentage(ego, 100)
        tm.ignore_vehicles_percentage(ego, 100)
        tm.vehicle_percentage_speed_difference(ego, -args.speed_boost)
        world.tick()
        hazard, hazard_tick = setup_hazard(world, actors, ego, args.scenario, rng)
        pipe.aeb.__post_init__()  # reset track history between trials
        pipe.model.predictor = None  # reset ByteTrack state

        braking, first_visible, first_brake, min_gap, fps = False, None, None, 1e9, []
        for step in range(int(args.seconds * args.fps)):
            frame = world.tick()
            t = step / args.fps
            hazard_tick(step)
            bgr = rgb_to_bgr(get_frame(rgb_q, frame))
            depth = depth_to_meters(get_frame(depth_q, frame))
            res = pipe(bgr, t=t, gt_depth=depth if args.oracle_depth else None)
            fps.append(1000 / max(res.timings_ms["total"], 1e-3))

            if first_visible is None and any(g["id"] == hazard.id for g in gt_boxes(world, rgb_cam, ego, spec, depth)):
                first_visible = t
            if res.action == Action.BRAKE:
                first_brake = t if first_brake is None else first_brake
                if not braking:
                    ego.set_autopilot(False, tm.get_port())
                    braking = True
                ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            elif braking:
                ego.set_autopilot(True, tm.get_port())
                braking = False

            gap = ego.get_location().distance(hazard.get_location()) - ego.bounding_box.extent.x
            min_gap = min(min_gap, gap)
            if writer is not None:
                writer.write(cv2.resize(pipe.draw(bgr, res), (args.video_w, args.video_w * 9 // 16)))
            if ego.get_velocity().length() < 0.1 and braking and step > 40:
                break  # stopped safely

        return {"trial": trial, "collided": any("walker" in c[1] or "vehicle" in c[1] for c in collisions),
                "min_gap_m": round(min_gap, 2),
                "reaction_s": round(first_brake - first_visible, 2) if first_brake is not None and first_visible is not None else None,
                "braked": first_brake is not None, "mean_fps": round(float(np.mean(fps)), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("CARLA_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--scenario", choices=["pedestrian_crossing", "lead_vehicle_brake"], default="pedestrian_crossing")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=25)
    ap.add_argument("--fps", type=float, default=20)
    ap.add_argument("--spawn", type=int, default=0)
    ap.add_argument("--speed-boost", type=float, default=0, help="%% faster than the speed limit")
    ap.add_argument("--oracle-depth", action="store_true")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--weights", help="override detector weights, e.g. yolo11s.pt (COCO baseline)")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--video-w", type=int, default=1280)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30)
    pipe = PerceptionPipeline(args.config, use_depth_model=not args.oracle_depth, weights=args.weights)
    writer = None
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                 (args.video_w, args.video_w * 9 // 16))

    trials = []
    for i in range(args.trials):
        r = run_trial(client, pipe, args, i, writer)
        trials.append(r)
        print(json.dumps(r))
    if writer:
        writer.release()

    reactions = [t["reaction_s"] for t in trials if t["reaction_s"] is not None]
    summary = {"scenario": args.scenario, "trials": len(trials), "oracle_depth": args.oracle_depth,
               "collision_rate": round(sum(t["collided"] for t in trials) / len(trials), 3),
               "brake_rate": round(sum(t["braked"] for t in trials) / len(trials), 3),
               "median_min_gap_m": round(float(np.median([t["min_gap_m"] for t in trials])), 2),
               "median_reaction_s": round(float(np.median(reactions)), 2) if reactions else None,
               "mean_fps": round(float(np.mean([t["mean_fps"] for t in trials])), 1),
               "per_trial": trials, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = Path("reports") / f"scenarios_{args.scenario}{'_oracle' if args.oracle_depth else ''}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_trial"}, indent=2))

    if os.environ.get("WANDB_API_KEY"):
        import wandb
        run = wandb.init(project="aiml04-perception", job_type="carla-scenarios", config=vars(args) | {"video": str(args.video)})
        run.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        if args.video:
            run.log({"demo": wandb.Video(str(args.video))})
        run.finish()


if __name__ == "__main__":
    main()
