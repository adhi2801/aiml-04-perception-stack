"""The tracker is chosen on validation frames by IDF1, and the test town can never be used to choose it."""

from __future__ import annotations

import pytest

from perception.eval_tracking import choose_tracker, val_frames


def report(idf1: float, switches: int, val_tail: float | None = 0.15, p95: float = 3.0) -> dict:
    return {
        "OVERALL": {"idf1": idf1, "mota": 0.3, "num_switches": switches},
        "_meta": {"val_tail": val_tail, "sequences": ["Town01_sunset"], "conf": 0.25, "association_ms": {"p95": p95}},
    }


def test_val_frames_keeps_the_contiguous_tail():
    lines = [str(i) for i in range(100)]
    assert val_frames(lines, 0.15) == [str(i) for i in range(85, 100)]
    assert val_frames(lines, None) == lines


def test_highest_idf1_wins():
    choice = choose_tracker({"bytetrack.yaml": report(0.40, 10), "botsort.yaml": report(0.55, 30)})
    assert choice["chosen"] == "botsort.yaml"
    assert choice["idf1"] == {"bytetrack.yaml": 0.4, "botsort.yaml": 0.55}


def test_a_tie_goes_to_fewer_id_switches():
    choice = choose_tracker({"bytetrack.yaml": report(0.5, 12), "botsort.yaml": report(0.5, 30)})
    assert choice["chosen"] == "bytetrack.yaml"


def test_a_test_town_report_cannot_choose_the_tracker():
    with pytest.raises(ValueError, match="validation"):
        choose_tracker({"bytetrack.yaml": report(0.4, 10), "botsort.yaml": report(0.6, 5, val_tail=None)})


def test_a_tracker_over_the_latency_budget_is_not_chosen_even_with_the_best_idf1():
    choice = choose_tracker({"botsort.yaml": report(0.91, 6, p95=47.8), "botsort_nogmc.yaml": report(0.89, 5, p95=5.2)})
    assert choice["chosen"] == "botsort_nogmc.yaml"
    assert choice["best_ignoring_budget"] == "botsort.yaml"
    assert choice["budget_added_after_first_run"] is True


def test_no_tracker_within_budget_is_an_error_not_a_silent_pick():
    with pytest.raises(ValueError, match="budget"):
        choose_tracker({"botsort.yaml": report(0.91, 6, p95=47.8)})
