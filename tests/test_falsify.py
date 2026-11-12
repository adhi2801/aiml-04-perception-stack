import json

import numpy as np

from perception.falsify import GP, SPACE, Falsifier, boundary_grid, latin_hypercube, margin, random_search, save, \
    summary, to_params


def toy_aeb(p: dict) -> float:
    """A synthetic closed loop whose failures sit in the fast + dark + occluded corner (about 28 % of the space;
    on it the falsifier finds 19-23 failures in 40 trials over 5 seeds vs 6-13 for random search)."""
    speed = (p["ego_kmh"] - 15) / 35
    dark = (60 - p["sun_altitude_deg"]) / 70
    risk = 1.6 * speed + 1.2 * dark + 0.9 * p["occluded"] + 0.3 * p["ped_kmh"] / 10
    return 6.0 - 2.4 * risk  # metres of margin; negative = collision


def test_margin_sign_convention():
    assert margin(False, 3.2, 0) == 3.2
    assert margin(True, 0.0, 27.0) == -27.0


def test_latin_hypercube_stratifies_every_dimension():
    u = latin_hypercube(20, 6, np.random.default_rng(0))
    for j in range(6):
        assert sorted(np.floor(u[:, j] * 20).astype(int)) == list(range(20))


def test_to_params_respects_ranges_and_integers():
    p = to_params(np.array([0, 1, 0.5, 0.5, 0.25, 0.7]))
    assert p["ego_kmh"] == 15 and p["ped_kmh"] == 10 and p["occluded"] == 1.0
    assert all(sp.low <= p[sp.name] <= sp.high for sp in SPACE)


def test_gp_interpolates_and_is_uncertain_far_from_data():
    rng = np.random.default_rng(0)
    X = rng.random((40, 2)) * 0.5            # data only in the lower-left quarter
    y = np.sin(4 * X[:, 0]) + X[:, 1]
    gp = GP().fit(X, y)
    mu, sd = gp.predict(X)
    assert np.abs(mu - y).max() < 0.1
    _, sd_far = gp.predict(np.array([[0.95, 0.95]]))
    assert sd_far[0] > 3 * sd.mean()


def test_falsifier_finds_more_failures_than_random_search_with_the_same_budget():
    wins, found_f, found_r = 0, [], []
    for seed in range(5):
        f = summary(Falsifier(toy_aeb, n_init=10, seed=seed).run(40))
        r = summary(random_search(toy_aeb, 40, seed=seed))
        found_f.append(f["failures"])
        found_r.append(r["failures"])
        wins += f["failures"] > r["failures"]
    assert wins >= 4, (found_f, found_r)
    assert np.mean(found_f) > 2 * np.mean(found_r)


def test_lcb_steps_dig_for_the_worst_case():
    hist = Falsifier(toy_aeb, n_init=10, seed=0).run(30)
    assert {h["how"] for h in hist} == {"init", "straddle", "lcb"}
    worst = summary(hist)["worst"]
    assert worst["margin"] < 0 and worst["params"]["occluded"] == 1


def test_boundary_slice_matches_the_true_failure_region():
    hist = Falsifier(toy_aeb, n_init=12, seed=1).run(60)
    g = boundary_grid(hist, SPACE, "ego_kmh", "sun_altitude_deg", fixed={"occluded": 1, "ped_kmh": 6.5,
                                                                        "overlap": 0.5, "rain": 40})
    p = np.array(g["p_fail"])
    assert p[-1, 0] > 0.8   # fastest + darkest corner: failure
    assert p[0, -1] < 0.2   # slowest + brightest corner: safe


def test_save_writes_both_searches(tmp_path):
    f = Falsifier(toy_aeb, n_init=5, seed=0).run(8)
    r = random_search(toy_aeb, 8)
    save(tmp_path / "f.json", f, r)
    d = json.loads((tmp_path / "f.json").read_text())
    assert d["falsifier"]["summary"]["trials"] == 8 and d["random"]["summary"]["trials"] == 8
