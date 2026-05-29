from typing import cast

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

from .roi import ROI


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
        self.pool = nn.AdaptiveAvgPool2d(1)

        with torch.no_grad():
            example_features = self.feature_extractor(torch.randn(1, 3, 224, 224))

        self.head = nn.Sequential(
            nn.Linear(example_features.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

        self.__backbone_frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: the input image
        :return: normalized ROI tensor
        """
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

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
