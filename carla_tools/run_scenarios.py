"""Closed-loop AEB evaluation in CARLA with Euro NCAP-style scenarios (no ROS, for repeatable batch runs and the demo video).

The ego drives on Traffic Manager autopilot at a fixed test speed and is told to IGNORE pedestrians and vehicles,
so the only thing that can stop the car is our perception stack. Scenario geometry lives in perception/scenarios.py.

  python carla_tools/run_scenarios.py --scenario cpna25 --speeds 20,30,40 --trials 5 --video reports/demo_cpna25.mp4
  python carla_tools/run_scenarios.py --scenario cpfa50 --speeds 20,30,40 --trials 5
  python carla_tools/run_scenarios.py --scenario ccrb --speeds 30,40 --trials 5
  python carla_tools/run_scenarios.py --scenario cpna25 --oracle-depth   # perfect depth: isolates the depth error

Per trial we log: collided?, impact speed, minimum gap, time from hazard visible to first BRAKE, and FPS.
Per test speed we report the avoided rate and mean speed reduction (the basis of Euro NCAP's AEB score).
Writes reports/ncap_<scenario>.json and .md (and logs to W&B if WANDB_API_KEY is set).
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

from perception import scenarios as sc  # noqa: E402
from perception.fusion.aeb import Action  # noqa: E402
from perception.pipeline import PerceptionPipeline  # noqa: E402


def ahead(transform: carla.Transform, meters: float, lateral: float = 0.0) -> carla.Location:
    f, r = transform.get_forward_vector(), transform.get_right_vector()
    loc = transform.location
    return carla.Location(loc.x + f.x * meters + r.x * lateral, loc.y + f.y * meters + r.y * lateral, loc.z + 1.0)


def front_gap(ego, point: carla.Location) -> float:
    """Distance from the ego's front bumper to `point`, measured along the ego's heading."""
    f, e = ego.get_transform().get_forward_vector(), ego.get_location()
    return (point.x - e.x) * f.x + (point.y - e.y) * f.y - ego.bounding_box.extent.x


def setup_hazard(world, actors, ego, scenario: str, rng: random.Random, tm=None, test_kmh: float | None = None):
    """Returns (hazard actor, per-tick function that drives it). `tm`/`test_kmh` are only needed for CCRb."""
    s = sc.get(scenario)
    wp = world.get_map().get_waypoint(ego.get_location())
    bl = world.get_blueprint_library()
    if s.kind == "pedestrian":
        # Far enough ahead for the ego to settle at the test speed before the pedestrian is released.
        hazard_wp = wp.next(110.0)[0]
        walker = world.spawn_actor(rng.choice(bl.filter("walker.pedestrian.*")),
                                   carla.Transform(ahead(hazard_wp.transform, 0, s.side * s.start_offset_m)))
        actors.append(walker)
        direction = hazard_wp.transform.get_right_vector() * (-s.side)
        width = 2 * ego.bounding_box.extent.y
        state = {"walking": False}

        def tick(_):
            if not state["walking"]:
                kmh = ego.get_velocity().length() * 3.6
                if kmh > 1 and front_gap(ego, hazard_wp.transform.location) <= sc.release_distance(s, kmh, width):
                    state["walking"] = True
            if state["walking"]:
                walker.apply_control(carla.WalkerControl(direction=direction, speed=s.ped_kmh / 3.6))
        return walker, tick

    # CCRb: a lead car drives ahead at the test speed, then brakes as hard as it can.
    hazard_wp = wp.next(s.headway_m + 2 * ego.bounding_box.extent.x)[0]
    lead = world.spawn_actor(bl.find("vehicle.audi.a2"), carla.Transform(
        hazard_wp.transform.location + carla.Location(z=0.5), hazard_wp.transform.rotation))
    actors.append(lead)
    state = {"steady": 0, "braking": False}
    if tm is not None:
        lead.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(lead, 100)
        if test_kmh:
            tm.set_desired_speed(lead, test_kmh)

    def tick(_):
        if not state["braking"]:
            target = test_kmh or 30.0
            state["steady"] = state["steady"] + 1 if ego.get_velocity().length() * 3.6 > 0.9 * target else 0
            if state["steady"] >= 40 or tm is None:  # ~2 s at the test speed (20 fps)
                state["braking"] = True
                if tm is not None:
                    lead.set_autopilot(False, tm.get_port())
        if state["braking"]:
            lead.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    return lead, tick


def run_trial(client, pipe: PerceptionPipeline, args, trial: int, test_kmh: float, writer) -> dict:
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
        tm.ignore_lights_percentage(ego, 100)
        tm.set_desired_speed(ego, test_kmh)
        world.tick()
        hazard, hazard_tick = setup_hazard(world, actors, ego, args.scenario, rng, tm, test_kmh)
        pipe.aeb.__post_init__()  # reset track history between trials
        pipe.model.predictor = None  # reset ByteTrack state

        braking, first_visible, first_brake, min_gap, fps = False, None, None, 1e9, []
        impact_kmh, kmh = None, 0.0
        for step in range(int(args.seconds * args.fps)):
            kmh = ego.get_velocity().length() * 3.6  # speed going into this tick (the impact speed if we hit now)
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
            if impact_kmh is None and any(c[1].startswith(("walker", "vehicle")) for c in collisions):
                impact_kmh = kmh
            if writer is not None:
                writer.write(cv2.resize(pipe.draw(bgr, res), (args.video_w, args.video_w * 9 // 16)))
            if impact_kmh is not None:
                break  # hit: the impact speed is what Euro NCAP scores
            if ego.get_velocity().length() < 0.1 and braking and step > 40:
                break  # stopped safely

        return {"trial": trial, "test_kmh": test_kmh, "collided": impact_kmh is not None,
                "impact_kmh": round(impact_kmh, 1) if impact_kmh is not None else 0.0,
                "min_gap_m": round(min_gap, 2),
                "reaction_s": round(first_brake - first_visible, 2) if first_brake is not None and first_visible is not None else None,
                "braked": first_brake is not None, "mean_fps": round(float(np.mean(fps)), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("CARLA_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--scenario", choices=[*sc.SCENARIOS, *sc.ALIASES], default="cpna25")
    ap.add_argument("--speeds", default="20,30,40", help="comma-separated test speeds in km/h")
    ap.add_argument("--trials", type=int, default=5, help="trials per test speed")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--fps", type=float, default=20)
    ap.add_argument("--spawn", type=int, default=0)
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

    scenario = sc.get(args.scenario)
    trials = []
    for test_kmh in [float(v) for v in args.speeds.split(",")]:
        for i in range(args.trials):
            r = run_trial(client, pipe, args, i, test_kmh, writer)
            trials.append(r)
            print(json.dumps(r))
    if writer:
        writer.release()

    reactions = [t["reaction_s"] for t in trials if t["reaction_s"] is not None]
    by_speed = sc.summarize(trials)
    summary = {"scenario": scenario.name, "trials": len(trials), "oracle_depth": args.oracle_depth,
               "by_speed": by_speed,
               "collision_rate": round(sum(t["collided"] for t in trials) / len(trials), 3),
               "brake_rate": round(sum(t["braked"] for t in trials) / len(trials), 3),
               "median_min_gap_m": round(float(np.median([t["min_gap_m"] for t in trials])), 2),
               "median_reaction_s": round(float(np.median(reactions)), 2) if reactions else None,
               "mean_fps": round(float(np.mean([t["mean_fps"] for t in trials])), 1),
               "per_trial": trials, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    stem = f"ncap_{sc.ALIASES.get(args.scenario, args.scenario)}{'_oracle' if args.oracle_depth else ''}"
    out = Path("reports") / f"{stem}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    title = scenario.name + (" (oracle depth)" if args.oracle_depth else "")
    out.with_suffix(".md").write_text(sc.to_markdown(title, by_speed))
    print(sc.to_markdown(title, by_speed))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_trial", "by_speed")}, indent=2))

    if os.environ.get("WANDB_API_KEY"):
        import wandb
        run = wandb.init(project="aiml04-perception", job_type="carla-scenarios", config=vars(args) | {"video": str(args.video)})
        run.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        run.log({"by_speed": wandb.Table(columns=list(by_speed[0]), data=[list(r.values()) for r in by_speed])})
        if args.video:
            run.log({"demo": wandb.Video(str(args.video))})
        run.finish()


if __name__ == "__main__":
    main()
