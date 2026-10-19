from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import mobilenet_v3_small

from .roi import ROI

# The detector sees every image resized to this (width, height). Boxes are predicted in
# coordinates normalized by the image size, so they map back to any input resolution.
ROI_INPUT_SIZE = (320, 240)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_for_roi(image: torch.Tensor) -> torch.Tensor:
    """Resize a [C, H, W] or [B, C, H, W] image to ROI_INPUT_SIZE (antialiased bilinear)."""
    batch = image if image.ndim == 4 else image.unsqueeze(0)
    width, height = ROI_INPUT_SIZE
    if batch.shape[-2:] != (height, width):
        batch = F.interpolate(
            batch, size=(height, width), mode="bilinear", align_corners=False, antialias=True
        )
    return batch if image.ndim == 4 else batch.squeeze(0)


def normalize_roi(x: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    return roi * roi.new_tensor([1/w, 1/h, 1/w, 1/h])

def denormalize_roi(x: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    return roi * roi.new_tensor([w, h, w, h])


class ROIDetector(nn.Module):
    def __init__(self, backbone: nn.Module | None = None):
        super().__init__()
        self.feature_extractor = cast(
            nn.Module,
            mobilenet_v3_small(weights="DEFAULT").features if backbone is None else backbone.features
        )
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Probe in eval mode so the random input doesn't shift the pretrained BatchNorm statistics.
        was_training = self.feature_extractor.training
        self.feature_extractor.eval()
        with torch.no_grad():
            example_features = self.feature_extractor(torch.randn(1, 3, 224, 224))
        self.feature_extractor.train(was_training)

        self.head = nn.Sequential(
            nn.Linear(example_features.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

        self.__backbone_frozen = False

    @classmethod
    def load(cls, path: str | Path) -> "ROIDetector":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls()
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )
        print(f"Loaded ROI detector from {path} to {device}")
        return model.to(device).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: image batch [B, 3, H, W] with values in [0, 1], any resolution
        :return: normalized ROI tensor
        """
        x = (resize_for_roi(x) - self.mean) / self.std
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> list[ROI]:
        """
        :param x: the input image
        :return: denormalized ROI object
        """
        y = denormalize_roi(x, self.forward(x))
        return [
            ROI(
                float(y[i, 0]),
                float(y[i, 1]),
                float(y[i, 2]),
                float(y[i, 3])
            )
            for i in range(y.shape[0])
        ]

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
