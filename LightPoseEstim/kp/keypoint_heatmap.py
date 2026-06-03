from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import mobilenet_v3_small


class KeypointHeatmapNet(nn.Module):
    def __init__(self, num_keypoints: int, backbone: nn.Module | None = None):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.feature_extractor = cast(
            nn.Module,
            mobilenet_v3_small(weights="DEFAULT").features if backbone is None else backbone.features
        )

        with torch.no_grad():
            example_features = self.feature_extractor(torch.randn(1, 3, 256, 256))

        in_channels = example_features.shape[1]
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, kernel_size=1),
        )

        self.__backbone_frozen = False

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
        height, width = x.shape[-2:]
        x = self.feature_extractor(x)
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        return self.head(x)

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        threshold: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: input image batch [B, 3, H, W]
        :return: peak coordinates [B, K, 2] and confidences [B, K]
        """
        return heatmap_peaks(self.forward(x), threshold=threshold)

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


@torch.no_grad()
def heatmap_peaks(
    heatmaps: torch.Tensor,
    threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-channel peak locations and confidences from heatmaps."""
    batch_size, num_keypoints, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch_size, num_keypoints, -1)
    confidence, index = flat.max(dim=-1)
    y = index // width
    x = index % width
    coordinates = torch.stack([x, y], dim=-1).float()
    confidence = confidence * (confidence >= threshold).float()
    return coordinates, confidence
