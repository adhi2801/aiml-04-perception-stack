"""Root cause of a closed-loop AEB failure, by replacing one perception stage at a time with the simulator's truth.

A failure (collision) is replayed with the same scenario parameters and seed under three oracle conditions:

  oracle detection   simulator boxes and actor ids instead of detector + tracker; distance still estimated
  oracle depth       detector + tracker as deployed; distance read from the simulator's depth map
  oracle both        perfect perception: what still fails is the AEB logic or the vehicle's braking

The cause is the smallest set of stages whose replacement removes the failure. This is a counterfactual test on the
same trial, not a correlation over many trials, so one replay per condition says something about that failure.
"""

from __future__ import annotations

from collections import Counter

CAUSES = (
    "detection",  # oracle detection alone avoids it
    "depth",  # oracle depth alone avoids it
    "either",  # each oracle alone avoids it: two weaknesses, fixing one is enough
    "detection + depth",  # only both together avoid it
    "decision / braking",  # perfect perception still collides: the AEB rule or the physics
)


def attribute(det_ok: bool, depth_ok: bool, both_ok: bool) -> str:
    """Cause of one failure from whether each oracle replay avoided the collision."""
    if det_ok and depth_ok:
        return "either"
    if det_ok:
        return "detection"
    if depth_ok:
        return "depth"
    if both_ok:
        return "detection + depth"
    return "decision / braking"


def summarize(rows: list[dict]) -> dict:
    """rows: one per replayed failure, each with det_ok, depth_ok, both_ok. Counts per cause, in CAUSES order."""
    counts = Counter(attribute(r["det_ok"], r["depth_ok"], r["both_ok"]) for r in rows)
    return {"failures": len(rows), "by_cause": {c: counts.get(c, 0) for c in CAUSES}}
