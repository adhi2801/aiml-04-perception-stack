"""Every combination of oracle outcomes maps to exactly one cause, and the summary counts them in a fixed order."""

from __future__ import annotations

import itertools

import pytest

from perception.root_cause import CAUSES, attribute, summarize


@pytest.mark.parametrize(
    ("det_ok", "depth_ok", "both_ok", "cause"),
    [
        (True, False, True, "detection"),
        (False, True, True, "depth"),
        (True, True, True, "either"),
        (False, False, True, "detection + depth"),
        (False, False, False, "decision / braking"),
    ],
)
def test_attribution(det_ok, depth_ok, both_ok, cause):
    assert attribute(det_ok, depth_ok, both_ok) == cause


def test_every_outcome_has_a_known_cause():
    for combo in itertools.product((False, True), repeat=3):
        assert attribute(*combo) in CAUSES


def test_summary_counts_in_fixed_order():
    rows = [
        {"det_ok": True, "depth_ok": False, "both_ok": True},
        {"det_ok": True, "depth_ok": False, "both_ok": True},
        {"det_ok": False, "depth_ok": False, "both_ok": False},
    ]
    s = summarize(rows)
    assert s["failures"] == 3
    assert list(s["by_cause"]) == list(CAUSES)
    assert s["by_cause"]["detection"] == 2 and s["by_cause"]["decision / braking"] == 1
