"""Per-frame trace of one closed-loop trial, to find which stage failed when the car did not brake.

Wraps the perception pipeline so that every frame logs what the AEB saw (each object's class, estimated distance,
corridor test, TTC, action) next to the simulator's truth (ego speed, true bumper-to-bumper gap to the hazard).

    python carla_tools/trace_scenario.py --scenario ccrb --speed 40 --trial 0 [--oracle-detection] [--oracle-depth]

Writes reports/trace_<scenario>_<speed>kmh_t<trial><oracle suffix>.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import carla

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_scenarios as rs

from perception.pipeline import PerceptionPipeline

ROOT = Path(__file__).resolve().parents[1]


class TracingPipeline:
    """Forwards everything to the real pipeline; records one row per frame."""

    def __init__(self, pipe, client):
        self._pipe, self._client, self.rows = pipe, client, []

    def __getattr__(self, name):
        return getattr(self._pipe, name)

    def __call__(self, bgr, t, **kw):
        res = self._pipe(bgr, t=t, **kw)
        actors = self._client.get_world().get_actors()  # fetched per frame: run_trial may load a new map
        ego = next(a for a in actors.filter("vehicle.*") if a.attributes.get("role_name") == "hero")
        others = [a for a in actors.filter("vehicle.*") if a.id != ego.id] + list(actors.filter("walker.*"))
        hazard = min(others, key=lambda a: a.get_location().distance(ego.get_location())) if others else None
        # front_gap measures to the hazard's centre; subtract its half-length for a bumper-to-bumper gap
        true_gap = (
            rs.front_gap(ego, hazard.get_location()) - hazard.bounding_box.extent.x if hazard is not None else None
        )
        th = res.threat
        self.rows.append(
            {
                "t": round(t, 2),
                "hazard_id": hazard.id if hazard is not None else "",
                "ego_kmh": round(ego.get_velocity().length() * 3.6, 1),
                "hazard_kmh": round(hazard.get_velocity().length() * 3.6, 1) if hazard is not None else "",
                "true_gap_m": round(true_gap, 2) if true_gap is not None else "",
                "n_objects": len(res.objects),
                "objects": "; ".join(
                    f"{o.cls}:{o.distance_m if o.distance_m is None else round(o.distance_m, 1)}" for o in res.objects
                ),
                "threat_cls": th.cls if th else "",
                "threat_dist_m": round(th.distance_m, 2) if th else "",
                "threat_ttc_s": round(th.ttc_s, 2) if th and th.ttc_s is not None else "",
                "threat_in_corridor": th.in_corridor if th else "",
                "action": res.action.name,
            }
        )
        return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--scenario", default="ccrb")
    ap.add_argument("--speed", type=float, default=40)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--fps", type=float, default=20)
    ap.add_argument("--spawn", type=int, default=0)
    ap.add_argument("--oracle-depth", action="store_true")
    ap.add_argument("--oracle-detection", action="store_true")
    ap.add_argument("--config", default="configs/pipeline_lowvram.yaml")
    ap.add_argument("--weights", default="models/yolo11s_carla.pt")
    ap.add_argument("--video-w", type=int, default=1280)
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(120)
    pipe = PerceptionPipeline(args.config, use_depth_model=not args.oracle_depth, weights=args.weights)
    tracer = TracingPipeline(pipe, client)
    result = rs.run_trial(client, tracer, args, args.trial, args.speed, None)

    out = ROOT / "reports" / f"trace_{args.scenario}_{int(args.speed)}kmh_t{args.trial}{rs.oracle_suffix(args)}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tracer.rows[0]))
        w.writeheader()
        w.writerows(tracer.rows)
    print(result)
    print(f"{len(tracer.rows)} frames -> {out}")


if __name__ == "__main__":
    main()
