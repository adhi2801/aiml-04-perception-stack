"""Learned lane-marking segmentation: LR-ASPP MobileNetV3 (torchvision), trained on CARLA labels.

  python -m perception.lanes.seg_model --train data/carla/Town03_clear data/carla/Town03_rain --val data/carla/Town02_test

Real-time-oriented choice: ~3M parameters, exports cleanly to ONNX/TensorRT.
Lane markings are thin (~1-2% of pixels), so the loss is Dice + positively weighted BCE.
"""
from __future__ import annotations

import argparse
import json
import os
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


class LaneFrames(Dataset):
    def __init__(self, seq_dir: Path, train: bool):
        self.rgb = sorted((Path(seq_dir) / "rgb").glob("*.jpg"))
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
            img = np.clip(img.astype(np.float32) * np.random.uniform(0.6, 1.4) + np.random.uniform(-25, 25), 0, 255)
        return to_tensor(img), torch.from_numpy((mask > 127).astype(np.float32))[None]


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

    def __init__(self, weights: str | Path, device: str | None = None, threshold: float = 0.5):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build(pretrained=False)
        self.model.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()
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
    ap.add_argument("--val", nargs="+", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("models/lane_lraspp.pt"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = DataLoader(ConcatDataset([LaneFrames(d, True) for d in args.train]), args.batch, shuffle=True, num_workers=4, drop_last=True)
    val = DataLoader(ConcatDataset([LaneFrames(d, False) for d in args.val]), args.batch, num_workers=4)
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
        tp = fp = fn = 0
        with torch.inference_mode():
            for x, y in val:
                pred = torch.sigmoid(model(x.to(device))["out"]) > 0.5
                y = y.to(device) > 0.5
                tp += (pred & y).sum().item()
                fp += (pred & ~y).sum().item()
                fn += (~pred & y).sum().item()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        rec = {"epoch": epoch, "loss": loss.item(), "val_pixel_f1": round(f1, 4)}
        print(json.dumps(rec))
        if run:
            run.log(rec)
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), args.out)
    if run:
        run.summary["best_val_pixel_f1"] = best
        run.finish()


if __name__ == "__main__":
    main()
