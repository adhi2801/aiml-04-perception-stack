"""Falsification of the AEB in closed loop: search CARLA scenarios for failures and map the failure boundary.

Each trial is a nearside pedestrian crossing (CPNA geometry) whose ego speed, walking speed, impact overlap, time of
day, rain and an occluding parked van come from the search (perception.falsify.SPACE). The falsifier and a random
search get the same trial budget, so the README can say how much faster the search finds failures.

  python carla_tools/falsify_carla.py --budget 60                    # falsifier + random baseline, 60 trials each
  python carla_tools/falsify_carla.py --budget 60 --skip-random      # falsifier only
  python scripts/plot_falsification.py                               # figures from reports/falsification.json

Writes reports/falsification.json after every trial (a crash or CARLA timeout loses at most one trial; --resume
continues from it).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import carla

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scenarios import run_trial  # noqa: E402

from perception import falsify as fz  # noqa: E402
from perception import scenarios as sc  # noqa: E402
from perception.pipeline import PerceptionPipeline  # noqa: E402

OUT = Path("reports") / "falsification.json"


def make_simulator(client, pipe, args, log: list[dict]):
    def simulate(p: dict) -> float:
        scenario = sc.Scenario("CPNA-search", "pedestrian", side=+1, overlap=p["overlap"], ped_kmh=p["ped_kmh"],
                               start_offset_m=4.0)
        for attempt in range(2):  # CARLA occasionally times out on a reload; one retry, then count it as lost
            try:
                r = run_trial(client, pipe, args, len(log), p["ego_kmh"], None, scenario=scenario,
                              conditions={"sun_altitude_deg": p["sun_altitude_deg"], "rain": p["rain"]},
                              occluded=bool(p["occluded"]))
                break
            except RuntimeError as exc:
                print(f"trial failed ({exc}); retrying" if attempt == 0 else f"trial lost: {exc}")
        else:
            raise SystemExit("CARLA is not responding")
        m = fz.margin(r["collided"], r["min_gap_m"], r["impact_kmh"])
        log.append({**p, **r, "margin": m})
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in log[-1].items()}))
        return m
    return simulate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("CARLA_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--n-init", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-random", action="store_true")
    ap.add_argument("--seconds", type=float, default=25)
    ap.add_argument("--fps", type=float, default=20)
    ap.add_argument("--spawn", type=int, default=0)
    ap.add_argument("--oracle-depth", action="store_true")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--weights", help="detector weights (default: the config's)")
    args = ap.parse_args()
    args.scenario = "cpna25"  # run_trial needs a default; every trial passes its own scenario

    client = carla.Client(args.host, args.port)
    client.set_timeout(60)
    pipe = PerceptionPipeline(args.config, use_depth_model=not args.oracle_depth, weights=args.weights)
    OUT.parent.mkdir(exist_ok=True)

    f_log, r_log = [], []
    falsifier = fz.Falsifier(make_simulator(client, pipe, args, f_log), n_init=args.n_init, seed=args.seed)
    t0 = time.time()
    f_hist = falsifier.run(args.budget)
    r_hist = [] if args.skip_random else fz.random_search(make_simulator(client, pipe, args, r_log), args.budget,
                                                          seed=args.seed)
    meta = {"weights": args.weights or pipe.cfg["detector"]["weights"], "oracle_depth": args.oracle_depth,
            "config": args.config, "budget": args.budget, "seed": args.seed, "minutes": round((time.time() - t0) / 60, 1),
            "trial_details": {"falsifier": f_log, "random": r_log}}
    out = fz.save(OUT, f_hist, r_hist or f_hist[:0], meta)
    if f_hist:
        out["boundary"] = {
            "speed_vs_sun_occluded": fz.boundary_grid(f_hist, fz.SPACE, "ego_kmh", "sun_altitude_deg",
                                                      {"occluded": 1, "ped_kmh": 5.0, "overlap": 0.25, "rain": 0.0}),
            "speed_vs_sun_clear_view": fz.boundary_grid(f_hist, fz.SPACE, "ego_kmh", "sun_altitude_deg",
                                                        {"occluded": 0, "ped_kmh": 5.0, "overlap": 0.25, "rain": 0.0}),
            "speed_vs_ped_speed": fz.boundary_grid(f_hist, fz.SPACE, "ego_kmh", "ped_kmh",
                                                   {"occluded": 0, "sun_altitude_deg": 45.0, "overlap": 0.25,
                                                    "rain": 0.0}),
        }
        OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({"falsifier": out["falsifier"]["summary"], "random": out["random"]["summary"]}, indent=2)[:3000])


if __name__ == "__main__":
    main()
