"""Learned lane-marking segmentation: LR-ASPP MobileNetV3 (torchvision), trained on CARLA labels.

  python -m perception.lanes.seg_model --train data/carla/Town03_clear data/carla/Town03_rain data/carla/Town04_cloudy

Model selection never sees the test town: by default the last --val-frac of every training sequence is held out
(temporal split, so near-duplicate neighbouring frames stay on one side), and the best epoch and the decision threshold
are picked there. Town02 is then scored once with perception.lanes.evaluate.

Real-time-oriented choice: ~3M parameters, exports cleanly to ONNX/TensorRT.
Lane markings are thin (~1-2% of pixels), so the loss is Dice + positively weighted BCE.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = (640, 360)  # W, H at which the network runs


THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


class LaneFrames(Dataset):
    def __init__(self, seq_dir: Path, train: bool, part: str = "all", val_frac: float = 0.15):
        """part: "all", or "train" / "val" = the first (1 - val_frac) / last val_frac of the frames."""
        rgb = sorted((Path(seq_dir) / "rgb").glob("*.jpg"))
        cut = int(len(rgb) * (1 - val_frac))
        self.rgb = {"all": rgb, "train": rgb[:cut], "val": rgb[cut:]}[part]
        self.seq = Path(seq_dir)
        self.train = train

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        img = cv2.resize(cv2.imread(str(self.rgb[i])), SIZE, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(cv2.imread(str(self.seq / "lanes" / (self.rgb[i].stem + ".png")), 0), SIZE,
                          interpolation=cv2.INTER_NEAREST)
        if self.train:
            if np.random.rand() < 0.5:  # lanes are left/right symmetric, unlike chest X-rays
                img, mask = img[:, ::-1], mask[:, ::-1]
            img = colour_jitter(img)
            img = np.clip(img.astype(np.float32) * np.random.uniform(0.6, 1.4) + np.random.uniform(-25, 25), 0, 255)
        return to_tensor(img), torch.from_numpy((mask > 127).astype(np.float32))[None]


def colour_jitter(bgr: np.ndarray) -> np.ndarray:
    """Hue rotation or grayscale, so paint colour is not a shortcut: Town03 lane paint is 0 % yellow,
    Town02 (the test town) 61 %, and a brightness-only model scored val F1 0.001 there."""
    r = np.random.rand()
    if r < 0.4:
        hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + np.random.randint(0, 180)) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if r < 0.65:
        return cv2.cvtColor(cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return bgr


def to_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = np.ascontiguousarray(bgr[..., ::-1], dtype=np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1))


def build(pretrained: bool = True) -> nn.Module:
    m = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
    m.classifier.low_classifier = nn.Conv2d(m.classifier.low_classifier.in_channels, 1, 1)
    m.classifier.high_classifier = nn.Conv2d(m.classifier.high_classifier.in_channels, 1, 1)
    return m


def loss_fn(logits, target, pos_weight=10.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device))
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return bce + dice


class LaneSegmenter:
    """Inference wrapper with the same call signature as the classical detector's mask output."""

    def __init__(self, weights: str | Path, device: str | None = None, threshold: float | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build(pretrained=False)
        self.model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()
        sidecar = Path(weights).with_suffix(".json")  # written by training: threshold picked on validation
        if threshold is None:
            threshold = json.loads(sidecar.read_text())["threshold"] if sidecar.exists() else 0.5
        self.threshold = threshold

    @torch.inference_mode()
    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        x = to_tensor(cv2.resize(bgr, SIZE, interpolation=cv2.INTER_AREA))[None].to(self.device)
        prob = torch.sigmoid(self.model(x)["out"])[0, 0].float().cpu().numpy()
        return (cv2.resize(prob, (w, h)) > self.threshold).astype(np.uint8) * 255


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", type=Path, required=True)
    ap.add_argument("--val", nargs="+", type=Path,
                    help="separate validation sequences; default: the last --val-frac of each --train sequence. "
                         "Never pass the test town here.")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("models/lane_lraspp.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # fixed input size
    # persistent workers: on Windows every epoch would otherwise re-spawn 4 processes that each re-import torch
    train_part = "all" if args.val else "train"
    val_sets = [LaneFrames(d, False) for d in args.val] if args.val else         [LaneFrames(d, False, "val", args.val_frac) for d in args.train]
    train = DataLoader(ConcatDataset([LaneFrames(d, True, train_part, args.val_frac) for d in args.train]), args.batch,
                       shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    val = DataLoader(ConcatDataset(val_sets), args.batch, num_workers=4, persistent_workers=True)
    model = build().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    run = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        run = wandb.init(project="aiml04-perception", job_type="lane-seg", config=vars(args) | {"train": [str(p) for p in args.train]})

    best = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        for x, y in train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x)["out"], y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        model.eval()
        counts = np.zeros((len(THRESHOLDS), 3))  # tp, fp, fn per threshold
        with torch.inference_mode():
            for x, y in val:
                prob = torch.sigmoid(model(x.to(device))["out"])
                y = y.to(device) > 0.5
                for k, thr in enumerate(THRESHOLDS):
                    pred = prob > thr
                    counts[k] += [(pred & y).sum().item(), (pred & ~y).sum().item(), (~pred & y).sum().item()]
        f1s = 2 * counts[:, 0] / np.maximum(2 * counts[:, 0] + counts[:, 1] + counts[:, 2], 1)
        k = int(np.argmax(f1s))
        f1, thr = float(f1s[k]), THRESHOLDS[k]
        rec = {"epoch": epoch, "loss": loss.item(), "val_pixel_f1": round(f1, 4), "threshold": thr,
               "epoch_sec": round(time.time() - t0, 1)}
        print(json.dumps(rec))
        if run:
            run.log(rec)
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), args.out)
            args.out.with_suffix(".json").write_text(json.dumps({
                "threshold": thr, "epoch": epoch, "val_pixel_f1": round(f1, 4),
                "train": [p.name for p in args.train],
                "val": [p.name for p in args.val] if args.val else f"last {args.val_frac:.0%} of each training sequence"},
                indent=2))
    if run:
        run.summary["best_val_pixel_f1"] = best
        run.finish()


if __name__ == "__main__":
    main()
