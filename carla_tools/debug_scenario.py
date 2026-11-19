"""Scenario geometry check without perception: is the hazard on the ego's path, and does the ego reach it?

  python carla_tools/debug_scenario.py --scenario cpna25 --kmh 20

Logs, every second of sim time: ego speed, the ego's front gap to the crossing point, the walker's lateral offset
from the ego's heading line, and whether the walker has been released. A scenario is only worth running through
the full pipeline once this shows the walker crossing in front of the car.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import carla

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from carla_common import attach_collision, spawn_ego, straight_spawn_points, sync_world
from run_scenarios import front_gap, setup_hazard

from perception import scenarios as sc


def lateral(ego, loc: carla.Location) -> float:
    r, e = ego.get_transform().get_right_vector(), ego.get_location()
    return (loc.x - e.x) * r.x + (loc.y - e.y) * r.y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="cpna25")
    ap.add_argument("--kmh", type=float, default=20)
    ap.add_argument("--spawn", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=40)
    args = ap.parse_args()
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60)
    with sync_world(client, fps=20) as (world, tm, actors):
        print("map", world.get_map().name)
        straight = straight_spawn_points(world.get_map())
        print(f"{len(straight)} straight spawn points")
        ego = spawn_ego(world, actors, spawn_index=straight[args.spawn % len(straight)])
        collisions = attach_collision(world, actors, ego)
        ego.set_autopilot(True, tm.get_port())
        tm.ignore_walkers_percentage(ego, 100)
        tm.ignore_vehicles_percentage(ego, 100)
        tm.ignore_lights_percentage(ego, 100)
        tm.auto_lane_change(ego, False)
        tm.set_desired_speed(ego, args.kmh)
        world.tick()
        hazard, tick = setup_hazard(world, actors, ego, sc.get(args.scenario), random.Random(0), tm, args.kmh)
        world.tick()  # actor locations are only valid after a tick
        start = hazard.get_location()
        print(f"hazard at {start}, straight-line {ego.get_location().distance(start):.1f} m")
        for step in range(int(args.seconds * 20)):
            world.tick()
            tick(step)
            if step % 20 == 0:
                h = hazard.get_location()
                print(
                    f"t={step / 20:5.1f}s v={ego.get_velocity().length() * 3.6:5.1f} km/h "
                    f"gap={front_gap(ego, h):6.1f} m lateral={lateral(ego, h):5.1f} m "
                    f"dist={ego.get_location().distance(h):6.1f} m"
                )
            if collisions:
                print("collision with", collisions[0][1], "at t", step / 20)
                break


if __name__ == "__main__":
    main()
