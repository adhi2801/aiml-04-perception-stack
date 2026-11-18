"""Build TensorRT engines (FP16, INT8) from the exported ONNX detector without loading PyTorch.

Ultralytics' own `export(format="engine")` builds inside a process that already holds PyTorch and its CUDA
libraries; on a 16 GB laptop that peaked near 15 GB of RAM and was killed. This script only loads TensorRT, numpy,
OpenCV and onnx, and writes engines that Ultralytics loads directly (same metadata header), so the benchmark and the
pipeline use them unchanged.

  python scripts/build_trt.py --onnx models/yolo11s_carla.onnx                     # FP16 -> models/yolo11s_carla.engine
  python scripts/build_trt.py --onnx models/yolo11s_carla.onnx --int8 \
      --calib data/carla_yolo/images/val                                           # INT8 -> ..._int8.engine

INT8 calibration uses entropy calibration on validation images (never the test town), preprocessed exactly like
Ultralytics (letterbox to 640, pad 114, RGB, /255).
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

ROOT = Path(__file__).resolve().parents[1]
LOGGER = trt.Logger(trt.Logger.WARNING)


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top : top + nh, left : left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(out[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0


class CudaRT:
    """The three cudart calls a calibrator needs, via ctypes (PyTorch ships cudart64_12.dll; nothing imported)."""

    def __init__(self):
        candidates = list((ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib").glob("cudart64_*.dll"))
        candidates += list((ROOT / ".venv" / "lib").glob("python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*"))
        if not candidates:
            raise SystemExit("cudart not found (needed only for INT8 calibration)")
        self.lib = ctypes.CDLL(str(candidates[0]))

    def malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc")
        return ptr.value

    def h2d(self, dst: int, src: np.ndarray) -> None:
        self._check(
            self.lib.cudaMemcpy(
                ctypes.c_void_p(dst), src.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(src.nbytes), 1
            ),
            "cudaMemcpy",
        )

    def free(self, ptr: int) -> None:
        self.lib.cudaFree(ctypes.c_void_p(ptr))

    @staticmethod
    def _check(code: int, what: str) -> None:
        if code != 0:
            raise RuntimeError(f"{what} failed with CUDA error {code}")


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, images: list[Path], size: int, batch: int, cache: Path):
        super().__init__()
        self.images, self.size, self.batch, self.cache = images, size, batch, cache
        self.cuda = CudaRT()
        self.buf = self.cuda.malloc(batch * 3 * size * size * 4)
        self.i = 0

    def get_batch_size(self) -> int:
        return self.batch

    def get_batch(self, names):
        if self.i + self.batch > len(self.images):
            return None
        x = np.stack([letterbox(cv2.imread(str(p)), self.size) for p in self.images[self.i : self.i + self.batch]])
        self.cuda.h2d(self.buf, np.ascontiguousarray(x))
        self.i += self.batch
        if self.i % (self.batch * 10) == 0:
            print(f"  calibrated on {self.i}/{len(self.images)} images", flush=True)
        return [self.buf]

    def read_calibration_cache(self):
        return self.cache.read_bytes() if self.cache.exists() else None

    def write_calibration_cache(self, cache) -> None:
        self.cache.write_bytes(bytes(cache))

    def __del__(self):
        with contextlib.suppress(Exception):
            self.cuda.free(self.buf)


def onnx_metadata(path: Path) -> dict:
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    return {p.key: p.value for p in m.metadata_props}


def keep_fp16(network, patterns: list[str]) -> int:
    """Pin every layer whose name contains one of `patterns` to FP16 (mixed precision); returns how many."""
    compute = {
        trt.LayerType.CONVOLUTION,
        trt.LayerType.ACTIVATION,
        trt.LayerType.ELEMENTWISE,
        trt.LayerType.MATRIX_MULTIPLY,
        trt.LayerType.SOFTMAX,
        trt.LayerType.POOLING,
        trt.LayerType.CONCATENATION,
    }
    pinned = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        # Only math layers with float outputs: shape bookkeeping (Int64 constants, Shape, Gather) cannot be FP16.
        is_float = all(layer.get_output_type(j) in (trt.float32, trt.float16) for j in range(layer.num_outputs))
        if layer.type in compute and is_float and any(p in layer.name for p in patterns):
            layer.precision = trt.float16
            for j in range(layer.num_outputs):
                if layer.get_output_type(j) == trt.float32:
                    layer.set_output_type(j, trt.float16)
            pinned += 1
    return pinned


def build(
    onnx_path: Path,
    out: Path,
    int8: bool,
    calib_dir: Path | None,
    n_calib: int,
    batch: int,
    workspace_gb: float,
    fp16_layers: list[str] | None = None,
    calib_cache: Path | None = None,
) -> Path:
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    config.set_flag(trt.BuilderFlag.FP16)  # INT8 engines keep FP16 for layers that do not quantise well
    size = network.get_input(0).shape[-1]
    if int8:
        images = sorted(calib_dir.glob("*.jpg")) + sorted(calib_dir.glob("*.png"))
        rng = np.random.default_rng(0)
        images = [images[i] for i in sorted(rng.choice(len(images), min(n_calib, len(images)), replace=False))]
        n = len(images) // batch * batch
        print(f"INT8 calibration on {n} validation images from {calib_dir}")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(images[:n], size, batch, calib_cache or out.with_suffix(".calib"))
        if fp16_layers:
            # Mixed precision: the layers named here stay FP16, the rest may run INT8.
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            pinned = keep_fp16(network, fp16_layers)
            if pinned == 0:  # e.g. Git Bash rewrote "/model.23/" into a Windows path
                raise SystemExit(f"--fp16-layers {fp16_layers} matched no layer; nothing would stay FP16")
            print(f"keeping {pinned} of {network.num_layers} layers in FP16 ({', '.join(fp16_layers)})")
        # the calibrator feeds a fixed batch; build the profile for it and for batch-1 inference
        if network.get_input(0).shape[0] == -1:
            raise SystemExit("export the ONNX with a fixed batch (dynamic=False)")
    t0 = time.time()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise SystemExit("TensorRT build failed")
    meta = json.dumps(onnx_metadata(onnx_path))
    with open(out, "wb") as f:  # Ultralytics engine format: int32 metadata length, JSON metadata, engine
        f.write(len(meta).to_bytes(4, byteorder="little", signed=True))
        f.write(meta.encode())
        f.write(engine)
    print(f"{out} ({out.stat().st_size / 2**20:.1f} MB) built in {time.time() - t0:.0f} s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib", type=Path, default=Path("data/carla_yolo/images/val"))
    ap.add_argument("--calib-images", type=int, default=400)
    ap.add_argument("--calib-batch", type=int, default=1)
    ap.add_argument("--workspace-gb", type=float, default=1.0)
    ap.add_argument(
        "--fp16-layers",
        nargs="+",
        help="with --int8: keep layers whose name contains any of these in FP16, e.g. /model.23/ (YOLO11 head)",
    )
    args = ap.parse_args()
    suffix = ("_int8" if args.int8 else "") + ("_mixed" if args.int8 and args.fp16_layers else "")
    out = args.onnx.with_name(args.onnx.stem + suffix + ".engine")
    # The calibration scales do not depend on which layers stay FP16, so a mixed build reuses the INT8 cache.
    cache = args.onnx.with_name(args.onnx.stem + "_int8.calib") if args.int8 else None
    build(
        args.onnx,
        out,
        args.int8,
        args.calib,
        args.calib_images,
        args.calib_batch,
        args.workspace_gb,
        args.fp16_layers,
        cache,
    )


if __name__ == "__main__":
    main()
