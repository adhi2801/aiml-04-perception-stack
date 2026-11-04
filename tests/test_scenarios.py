"""Euro NCAP-style scenario geometry and scoring (CPU only, no CARLA)."""
import pytest

from perception import scenarios as sc

WIDTH = 1.85  # Tesla Model 3, the CARLA ego


def test_impact_points():
    assert sc.impact_lateral(sc.get("cpfa50"), WIDTH) == pytest.approx(0.0)
    assert sc.impact_lateral(sc.get("cpna25"), WIDTH) == pytest.approx(WIDTH / 4)   # nearer the kerb side
    assert sc.impact_lateral(sc.get("cpna75"), WIDTH) == pytest.approx(-WIDTH / 4)


def test_release_timing_makes_an_unbraked_car_hit():
    s = sc.get("cpna25")
    for kmh in (20, 30, 40):
        gap = sc.release_distance(s, kmh, WIDTH)
        walk = s.start_offset_m - sc.impact_lateral(s, WIDTH)
        assert gap / (kmh / 3.6) == pytest.approx(walk / (s.ped_kmh / 3.6))   # same arrival time
    assert sc.release_distance(s, 40, WIDTH) == pytest.approx(2 * sc.release_distance(s, 20, WIDTH))


def test_aliases_and_summary():
    assert sc.get("pedestrian_crossing").name == "CPNA-25" and sc.get("lead_vehicle_brake").name == "CCRb"
    trials = [{"test_kmh": 20, "collided": False, "impact_kmh": 0},
              {"test_kmh": 20, "collided": False, "impact_kmh": 0},
              {"test_kmh": 40, "collided": True, "impact_kmh": 15.0},
              {"test_kmh": 40, "collided": False, "impact_kmh": 0}]
    rows = sc.summarize(trials)
    assert rows[0] == {"test_kmh": 20, "trials": 2, "avoided_rate": 1.0, "mean_impact_kmh": 0.0,
                       "mean_speed_reduction_kmh": 20.0}
    assert rows[1]["avoided_rate"] == 0.5 and rows[1]["mean_speed_reduction_kmh"] == 32.5
    assert "| 40 km/h | 2 | 50% |" in sc.to_markdown("CPNA-25", rows)
