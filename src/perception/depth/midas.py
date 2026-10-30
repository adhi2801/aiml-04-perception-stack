"""MiDaS monocular depth (Ranftl et al.) via torch.hub, returning relative inverse depth at frame size."""
from __future__ import annotations

import cv2
import numpy as np
import torch


class MiDaS:
    def __init__(self, model_type: str = "MiDaS_small", device: str | None = None, half: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True).to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform if model_type == "MiDaS_small" else transforms.dpt_transform
        self.half = half and self.device.type == "cuda"
        if self.half:
            self.model = self.model.half()

    @torch.inference_mode()
    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(rgb).to(self.device)
        if self.half:
            x = x.half()
        pred = self.model(x)
        pred = torch.nn.functional.interpolate(pred.unsqueeze(1).float(), size=bgr.shape[:2], mode="bicubic",
                                               align_corners=False).squeeze()
        return pred.cpu().numpy()
