"""CPU-only tests for the runtime benchmark's bookkeeping (no GPU, engines or dataset needed)."""

from pathlib import Path

from perception.optimize.benchmark import add_relative, resolve_variants, to_markdown


def _row(name: str, mean_ms: float, map50_95: float) -> dict:
    return {
        "variant": name,
        "size_mb": 1.0,
        "p50_ms": mean_ms,
        "p95_ms": mean_ms,
        "model_p50_ms": mean_ms,
        "fps": round(1000 / mean_ms, 1),
        "mean_ms": mean_ms,
        "map50_95": map50_95,
    }


def test_relative_numbers_are_against_the_first_row():
    rows = [_row("PyTorch FP32", 20.0, 0.70), _row("TensorRT FP16", 10.0, 0.69), _row("TensorRT INT8", 8.0, 0.65)]
    add_relative(rows)
    assert [r["speedup_vs_pytorch_gpu"] for r in rows] == [1.0, 2.0, 2.5]
    assert [r["map_drop_pts"] for r in rows] == [0.0, 1.0, 5.0]


def test_missing_engines_are_skipped_not_exported(tmp_path: Path):
    weights = tmp_path / "det.pt"
    weights.write_bytes(b"x")
    (tmp_path / "det.onnx").write_bytes(b"x")  # ONNX exists, neither engine does
    names = [
        v[0]
        for v in resolve_variants(weights, 640, "data.yaml", allow_export=False, cpu=True, cuda_check=lambda _: True)
    ]
    assert names == ["PyTorch FP32", "ONNX Runtime FP32 (CUDA)", "PyTorch FP32 (CPU)", "ONNX Runtime FP32 (CPU)"]
    assert not (tmp_path / "det.engine").exists()


def test_engines_next_to_the_weights_are_found(tmp_path: Path):
    weights = tmp_path / "det.pt"
    for name in ("det.pt", "det.onnx", "det.engine", "det_int8.engine"):
        (tmp_path / name).write_bytes(b"x")
    variants = resolve_variants(weights, 640, "data.yaml", allow_export=False, cpu=False, cuda_check=lambda _: True)
    assert [(v[0], v[1].name, v[2]) for v in variants] == [
        ("PyTorch FP32", "det.pt", False),
        ("ONNX Runtime FP32 (CUDA)", "det.onnx", False),
        ("TensorRT FP16", "det.engine", True),
        ("TensorRT INT8 (entropy calibration)", "det_int8.engine", False),
    ]


def test_markdown_states_device_and_split():
    rows = [_row("PyTorch FP32", 20.0, 0.70)]
    add_relative(rows)
    md = to_markdown({"device": "GTX 1650", "nvidia_driver": "610.91", "cpu": "x86"}, 640, rows)
    assert "GTX 1650" in md and "610.91" in md and "val split" in md
    assert md.count("\n| ") == 2  # header + one row


def test_onnx_cuda_row_is_dropped_when_it_would_really_run_on_the_cpu(tmp_path: Path):
    weights = tmp_path / "det.pt"
    for name in ("det.pt", "det.onnx"):
        (tmp_path / name).write_bytes(b"x")
    names = [v[0] for v in resolve_variants(weights, 640, "d.yaml", False, cpu=True, cuda_check=lambda _: False)]
    assert "ONNX Runtime FP32 (CUDA)" not in names
    assert "ONNX Runtime FP32 (CPU)" in names  # the honest CPU row stays


def test_per_class_drop_finds_the_class_that_pays():
    from perception.optimize.precision_report import per_class_drop

    drop = per_class_drop(
        {
            "fp32": {"pedestrian": 0.25, "car": 0.87},
            "int8": {"pedestrian": 0.10, "car": 0.86},
        }
    )
    assert drop == {"int8": {"pedestrian": 15.0, "car": 1.0}}


def test_block_spread_exposes_a_clock_change_mid_run():
    from perception.optimize.benchmark import block_spread

    steady = [10.0] * 100
    assert block_spread(steady) == [10.0, 10.0]
    throttled = [10.0] * 50 + [20.0] * 50  # the CPU slowed down halfway through
    low, high = block_spread(throttled)
    assert (low, high) == (10.0, 20.0)
