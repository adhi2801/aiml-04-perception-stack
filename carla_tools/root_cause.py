"""Replay every collision the falsifier found under three oracle conditions and attribute each one to a stage.

  python carla_tools/root_cause.py                       # reads reports/falsification.json
  python carla_tools/root_cause.py --source random       # the random-search failures instead

Each failure is re-run with its own scenario parameters and trial index (so the same spawn point and seed), first
as deployed to confirm it still fails (CARLA is not bit-exact between runs; a failure that does not reproduce is
reported as such, not attributed), then with oracle detection, oracle depth and both. The rule that turns the three
outcomes into a cause is perception.root_cause.attribute. Writes reports/root_cause.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import carla

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scenarios import run_trial

from perception import root_cause as rc
from perception import scenarios as sc
from perception.pipeline import PerceptionPipeline

CONDITIONS = {"as_deployed": (False, False), "det": (True, False), "depth": (False, True), "both": (True, True)}


def replay(client, pipe, args, trial: dict, oracle_detection: bool, oracle_depth: bool) -> dict:
    args.oracle_detection, args.oracle_depth = oracle_detection, oracle_depth
    scenario = sc.Scenario(
        "CPNA-search", "pedestrian", side=+1, overlap=trial["overlap"], ped_kmh=trial["ped_kmh"], start_offset_m=4.0
    )
    return run_trial(
        client,
        pipe,
        args,
        trial["trial"],
        trial["ego_kmh"],
        None,
        scenario=scenario,
        conditions={"sun_altitude_deg": trial["sun_altitude_deg"], "rain": trial["rain"]},
        occluded=bool(trial["occluded"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("CARLA_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--source", choices=("falsifier", "random"), default="falsifier")
    ap.add_argument("--falsification", type=Path, default=Path("reports/falsification.json"))
    ap.add_argument("--seconds", type=float, default=25)
    ap.add_argument("--fps", type=float, default=20)
    ap.add_argument("--spawn", type=int, default=0)
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--weights")
    args = ap.parse_args()
    args.scenario = "cpna25"

    found = json.loads(args.falsification.read_text())
    failures = [t for t in found["trial_details"][args.source] if t["collided"]]
    print(f"{len(failures)} {args.source} failures to replay")
    client = carla.Client(args.host, args.port)
    client.set_timeout(60)
    pipe = PerceptionPipeline(args.config, use_depth_model=False, weights=args.weights)

    rows, not_reproduced = [], []
    for t in failures:
        outcome = {name: replay(client, pipe, args, t, *flags) for name, flags in CONDITIONS.items()}
        ok = {name: not r["collided"] for name, r in outcome.items()}
        record = {
            "trial": t["trial"],
            "params": {k: t[k] for k in ("ego_kmh", "ped_kmh", "overlap", "sun_altitude_deg", "rain", "occluded")},
            "impact_kmh": {name: r["impact_kmh"] for name, r in outcome.items()},
            "det_ok": ok["det"],
            "depth_ok": ok["depth"],
            "both_ok": ok["both"],
        }
        if ok["as_deployed"]:
            not_reproduced.append(record)
        else:
            record["cause"] = rc.attribute(ok["det"], ok["depth"], ok["both"])
            rows.append(record)
        print(json.dumps(record))

    report = {
        "source": args.source,
        "weights": args.weights or pipe.cfg["detector"]["weights"],
        "config": args.config,
        **rc.summarize(rows),
        "not_reproduced": len(not_reproduced),
        "per_failure": rows,
        "not_reproduced_details": not_reproduced,
    }
    out = Path("reports") / "root_cause.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("failures", "by_cause", "not_reproduced")}, indent=2))


if __name__ == "__main__":
    main()
