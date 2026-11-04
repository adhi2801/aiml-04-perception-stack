"""Euro NCAP-style AEB test scenarios: geometry and scoring, without any CARLA dependency (so it is unit-tested in CI).

Naming follows the Euro NCAP AEB VRU / Car-to-Car protocols:
  CPNA-25 / CPNA-75  Car-to-Pedestrian Nearside Adult: an adult walks in from the kerb side (the right, since CARLA
                     drives on the right) and, if the car does not brake, is hit at 25% / 75% of the car's width
                     measured from that side.
  CPFA-50            Car-to-Pedestrian Farside Adult: walks in from the far (left) side, impact at the car's centre.
  CCRb               Car-to-Car Rear braking: a lead car at the same speed brakes hard in front of the ego.

The pedestrian is released at the moment that makes an un-braked car hit them exactly at the overlap point, so a
"no collision" result really is the AEB's doing. Speeds and overlaps follow the protocol. Start offsets, the
car-following setup and the friction model are simplified, so results are "Euro NCAP-style", not a rating.
Euro NCAP scores speed reduction for test speeds up to 40 km/h, which is what `summarize` reports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str                 # "pedestrian" | "lead_vehicle"
    side: int = 0             # +1 walks in from the right (nearside), -1 from the left (farside)
    overlap: float = 0.5      # impact point as a fraction of the car width, measured from the pedestrian's side
    ped_kmh: float = 5.0      # walking speed
    start_offset_m: float = 0.0   # where the pedestrian waits, lateral distance from the lane centre
    headway_m: float = 12.0   # CCRb: initial gap to the lead car


SCENARIOS = {
    "cpna25": Scenario("CPNA-25", "pedestrian", side=+1, overlap=0.25, ped_kmh=5.0, start_offset_m=4.0),
    "cpna75": Scenario("CPNA-75", "pedestrian", side=+1, overlap=0.75, ped_kmh=5.0, start_offset_m=4.0),
    "cpfa50": Scenario("CPFA-50", "pedestrian", side=-1, overlap=0.50, ped_kmh=8.0, start_offset_m=6.0),
    "ccrb": Scenario("CCRb", "lead_vehicle", headway_m=12.0),
}
ALIASES = {"pedestrian_crossing": "cpna25", "lead_vehicle_brake": "ccrb"}  # names used before the Euro NCAP rework


def get(name: str) -> Scenario:
    return SCENARIOS[ALIASES.get(name, name)]


def impact_lateral(s: Scenario, car_width: float) -> float:
    """Signed lateral position (m, + = right of the lane centre) where an un-braked car would hit the pedestrian."""
    return s.side * (car_width / 2 - s.overlap * car_width)


def release_distance(s: Scenario, ego_kmh: float, car_width: float) -> float:
    """Gap (m) from the ego's front bumper to the crossing line at which the pedestrian must start walking.

    Both arrive at the impact point at the same time: walk distance / walk speed = release gap / ego speed.
    """
    walk = abs(s.side * s.start_offset_m - impact_lateral(s, car_width))
    return ego_kmh / 3.6 * walk / (s.ped_kmh / 3.6)


def summarize(trials: list[dict]) -> list[dict]:
    """Per test speed: avoided rate, mean impact speed and mean speed reduction (Euro NCAP's AEB score basis).

    Each trial needs `test_kmh`, `collided` and `impact_kmh` (0 if avoided).
    """
    rows = []
    for v in sorted({t["test_kmh"] for t in trials}):
        ts = [t for t in trials if t["test_kmh"] == v]
        impact = np.array([t["impact_kmh"] if t["collided"] else 0.0 for t in ts])
        rows.append({"test_kmh": v, "trials": len(ts),
                     "avoided_rate": round(float(np.mean([not t["collided"] for t in ts])), 3),
                     "mean_impact_kmh": round(float(impact.mean()), 1),
                     "mean_speed_reduction_kmh": round(float(v - impact.mean()), 1)})
    return rows


def to_markdown(name: str, rows: list[dict]) -> str:
    lines = [f"### {name}", "", "| Test speed | Trials | Avoided | Mean impact speed | Mean speed reduction |",
             "|---|---|---|---|---|"]
    lines += [f"| {r['test_kmh']:g} km/h | {r['trials']} | {r['avoided_rate']:.0%} | {r['mean_impact_kmh']:.1f} km/h | "
              f"{r['mean_speed_reduction_kmh']:.1f} km/h |" for r in rows]
    return "\n".join(lines) + "\n"
