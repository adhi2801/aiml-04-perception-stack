import numpy as np

from perception.eval_detection import average_precision, box_iou, map_from_frames, match_frame


def frame(pb, pc, ps, gb, gc):
    pb, gb = np.array(pb, float).reshape(-1, 4), np.array(gb, float).reshape(-1, 4)
    pc, ps, gc = np.array(pc, int), np.array(ps, float), np.array(gc, int)
    return {"tp": match_frame(pb, pc, ps, gb, gc), "ps": ps, "pc": pc, "gc": gc}


def test_iou_identity_and_disjoint():
    a = np.array([[0, 0, 10, 10]], float)
    assert np.isclose(box_iou(a, a)[0, 0], 1.0)
    assert box_iou(a, np.array([[20, 20, 30, 30]], float))[0, 0] == 0.0


def test_perfect_detections_score_one():
    f = frame([[0, 0, 10, 10], [50, 50, 80, 90]], [0, 1], [0.9, 0.8], [[0, 0, 10, 10], [50, 50, 80, 90]], [0, 1])
    m, m50, per = map_from_frames([f])
    assert np.isclose(m, 1.0) and np.isclose(m50, 1.0)


def test_low_confidence_false_positive_does_not_hurt_ap():
    # A false positive ranked below the true positive leaves precision at recall 1.0 unchanged.
    f = frame([[0, 0, 10, 10], [100, 100, 120, 120]], [0, 0], [0.9, 0.1], [[0, 0, 10, 10]], [0])
    assert np.isclose(map_from_frames([f])[1], 1.0)


def test_wrong_class_is_not_a_match():
    f = frame([[0, 0, 10, 10]], [1], [0.9], [[0, 0, 10, 10]], [0])
    assert f["tp"].sum() == 0


def test_duplicate_detection_counts_once():
    f = frame([[0, 0, 10, 10], [0, 0, 10, 10]], [0, 0], [0.9, 0.8], [[0, 0, 10, 10]], [0])
    assert f["tp"][:, 0].tolist() == [True, False]


def test_missing_half_the_objects_halves_ap():
    ap = average_precision(np.ones((1, 10), bool), np.array([0.9]), n_gt=2)
    assert np.allclose(ap, 51 / 101)  # recall reaches 0.5: 51 of 101 recall points have precision 1


def test_loose_box_matches_at_050_not_at_095():
    f = frame([[0, 0, 10, 12]], [0], [0.9], [[0, 0, 10, 10]], [0])  # IoU = 0.833
    assert f["tp"][0, 0] and not f["tp"][0, -1]


def test_pick_conf_is_the_strictest_threshold_reaching_target_recall():
    from perception.eval_detection import pick_conf

    # 10 pedestrians within 30 m whose best matching detection scores are 0.1 .. 1.0, plus one far one (ignored)
    gc = np.zeros(11, int)
    gd = np.array([10.0] * 10 + [60.0])
    best = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0])
    f = {"gc": gc, "gd": gd, "best_conf": best, "ps": np.array([0.95, 0.4]),
         "tp": np.array([[True] * 10, [False] * 10])}
    out = pick_conf([f], target_recall=0.8, max_dist=30)
    assert out["n_objects"] == 10 and out["reached_target"]
    assert out["conf"] == 0.3 and out["recall_at_conf"] == 0.8      # 0.3 keeps 8 of 10; 0.31 would keep 7
    assert out["false_detections_per_frame"] == 1.0                  # the 0.4 detection is a false positive
