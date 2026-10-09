from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import mobilenet_v3_small

from ..crop import HEATMAP_STRIDE

DEFAULT_PEAK_THRESHOLD = 0.3
_HEATMAP_PRIOR_BIAS = -2.19  # sigmoid ~ 0.1 at init, as in CenterNet
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class KeypointHeatmapNet(nn.Module):
    """
    MobileNetV3-small backbone with a small feature-pyramid decoder.
    Outputs one heatmap of logits per keypoint at 1/HEATMAP_STRIDE of the input resolution.
    """

    def __init__(
        self,
        num_keypoints: int,
        backbone: nn.Module | None = None,
        decoder_channels: int = 64,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.stride = HEATMAP_STRIDE
        self.feature_extractor = cast(
            nn.Module,
            mobilenet_v3_small(weights="DEFAULT").features if backbone is None else backbone.features
        )
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self.taps, tap_channels = self._find_taps()
        self.lateral = nn.ModuleList(
            nn.Conv2d(channels, decoder_channels, kernel_size=1) for channels in tap_channels
        )
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, num_keypoints, kernel_size=1),
        )
        nn.init.constant_(self.head[-1].bias, _HEATMAP_PRIOR_BIAS)

        self.__backbone_frozen = False

    def _find_taps(self) -> tuple[list[int], list[int]]:
        """Deepest backbone layer at each stride from HEATMAP_STRIDE up, and its channel count."""
        size = 256
        deepest: dict[int, tuple[int, int]] = {}
        was_training = self.feature_extractor.training
        self.feature_extractor.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, size, size)
            for i, layer in enumerate(self.feature_extractor):
                x = layer(x)
                stride = size // x.shape[-1]
                if stride >= HEATMAP_STRIDE:
                    deepest[stride] = (i, x.shape[1])
        self.feature_extractor.train(was_training)
        if HEATMAP_STRIDE not in deepest:
            raise ValueError(f"Backbone has no layer at stride {HEATMAP_STRIDE}.")
        strides = sorted(deepest)
        return [deepest[s][0] for s in strides], [deepest[s][1] for s in strides]

    @classmethod
    def load(cls, path: str | Path, num_keypoints: int) -> "KeypointHeatmapNet":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(num_keypoints=num_keypoints)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )
        print(f"Loaded keypoint heatmap net from {path} to {device}")
        return model.to(device).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: image batch [B, 3, H, W] with values in [0, 1]
        :return: heatmap logits [B, K, ceil(H / stride), ceil(W / stride)]
        """
        x = (x - self.mean) / self.std
        features = []
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i in self.taps:
                features.append(x)
            if i == self.taps[-1]:
                break

        # Top-down: upsample the coarser map and add the finer layer's projection.
        y = self.lateral[-1](features[-1])
        for lateral, feature in zip(reversed(self.lateral[:-1]), reversed(features[:-1])):
            y = F.interpolate(y, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            y = y + lateral(feature)
        return self.head(y)

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        threshold: float = DEFAULT_PEAK_THRESHOLD,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: input image batch [B, 3, H, W]
        :return: peak coordinates [B, K, 2] in heatmap cells (x, y) and confidences [B, K]
        """
        return heatmap_peaks(self.forward(x).sigmoid(), threshold=threshold)

    def freeze_backbone(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.__backbone_frozen = True

    def unfreeze_backbone(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.__backbone_frozen = False

    @property
    def backbone_frozen(self):
        return self.__backbone_frozen


class HeatmapFocalLoss(nn.Module):
    """
    Focal loss for soft Gaussian heatmap targets (Quality Focal Loss, Li et al. 2020).

    BCE is scaled by |target - prediction|^gamma, so the many easy background pixels
    barely count. Unlike CenterNet's variant it needs no exact 1.0 peak cell, which keeps
    the sub-cell keypoint position in the target. The sum is divided by the number of
    keypoints that have a target peak.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weight = (target - logits.sigmoid()).abs().pow(self.gamma)
        num_peaks = (target.flatten(2).amax(dim=2) > 0).sum().clamp_min(1)
        return (weight * bce).sum() / num_peaks


@torch.no_grad()
def heatmap_peaks(
    heatmaps: torch.Tensor,
    threshold: float = DEFAULT_PEAK_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel peak location (x, y) in heatmap cells and its value. The location is
    refined below one cell by fitting a parabola to the log heatmap around the maximum,
    which is exact for a Gaussian peak.
    """
    batch_size, num_keypoints, height, width = heatmaps.shape
    log_flat = heatmaps.clamp_min(1e-6).log().reshape(batch_size, num_keypoints, -1)
    confidence, index = heatmaps.reshape(batch_size, num_keypoints, -1).max(dim=-1)
    y = index // width
    x = index % width

    def log_value(dx: int, dy: int) -> torch.Tensor:
        xn = (x + dx).clamp(0, width - 1)
        yn = (y + dy).clamp(0, height - 1)
        return log_flat.gather(-1, (yn * width + xn).unsqueeze(-1)).squeeze(-1)

    center = log_value(0, 0)

    def refine(minus: torch.Tensor, plus: torch.Tensor, interior: torch.Tensor) -> torch.Tensor:
        curvature = plus - 2.0 * center + minus
        step = 0.5 * (minus - plus) / curvature
        step = torch.where(interior & (curvature < 0), step, torch.zeros_like(step))
        return step.clamp(-0.5, 0.5)

    dx = refine(log_value(-1, 0), log_value(1, 0), (x > 0) & (x < width - 1))
    dy = refine(log_value(0, -1), log_value(0, 1), (y > 0) & (y < height - 1))
    coordinates = torch.stack([x.float() + dx, y.float() + dy], dim=-1)
    confidence = confidence * (confidence >= threshold).float()
    return coordinates, confidence
