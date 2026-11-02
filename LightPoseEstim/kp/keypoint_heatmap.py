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
    Outputs one heatmap of logits per keypoint at 1/HEATMAP_STRIDE of the input resolution,
    and a coarse object rotation from the whole crop. Keypoint heatmaps rely on local
    appearance, so on a near-symmetric object they can swap sides; the global rotation
    picks between a PnP pose and its symmetric twins.

    With ``num_turns`` > 0 (keypoints in symmetric sets, see symmetry.py) the heatmaps and the
    rotation may use any of the labellings the object's turns induce, and a classifier gives the
    turn from the labelling used to the true one. It keeps an 8x8 layout of the decoder's
    features: which side a window or screen is on is what tells the turns apart.
    """

    def __init__(
        self,
        num_keypoints: int,
        backbone: nn.Module | None = None,
        decoder_channels: int = 64,
        num_turns: int = 0,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_turns = num_turns
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
        # Keeps a coarse 4x4 layout: where parts sit in the crop is what gives orientation away.
        self.orientation = nn.Sequential(
            nn.Conv2d(tap_channels[-1], 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6),
        )
        if num_turns > 0:
            self.turn_classifier = nn.Sequential(
                nn.Conv2d(decoder_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, num_turns),
            )

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
    def load(cls, path: str | Path, num_keypoints: int, num_turns: int = 0) -> "KeypointHeatmapNet":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(num_keypoints=num_keypoints, num_turns=num_turns)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )
        print(f"Loaded keypoint heatmap net from {path} to {device}")
        return model.to(device).eval()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        :param x: image batch [B, 3, H, W] with values in [0, 1]
        :return: heatmap logits [B, K, ceil(H / stride), ceil(W / stride)], the
            allocentric object rotation as 6 numbers [B, 6] (see rotation_from_6d) and,
            with ``num_turns``, the turn logits [B, num_turns]
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
        if self.num_turns > 0:
            return self.head(y), self.orientation(features[-1]), self.turn_classifier(y)
        return self.head(y), self.orientation(features[-1])

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        threshold: float = DEFAULT_PEAK_THRESHOLD,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input image batch [B, 3, H, W]
        :return: peak coordinates [B, K, 2] in heatmap cells (x, y), confidences [B, K],
            and allocentric rotation matrices [B, 3, 3]
        """
        logits, rotation = self.forward(x)[:2]
        coordinates, confidence = heatmap_peaks(logits.sigmoid(), threshold=threshold)
        return coordinates, confidence, rotation_from_6d(rotation)

    @torch.no_grad()
    def inference_with_turns(
        self,
        x: torch.Tensor,
        threshold: float = DEFAULT_PEAK_THRESHOLD,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``inference`` plus the turn probabilities [B, num_turns] (needs ``num_turns``)."""
        logits, rotation, turns = self.forward(x)
        coordinates, confidence = heatmap_peaks(logits.sigmoid(), threshold=threshold)
        return coordinates, confidence, rotation_from_6d(rotation), turns.softmax(dim=1)

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


class KeypointLoss(nn.Module):
    """Heatmap focal loss plus squared Frobenius distance between predicted and true rotation."""

    def __init__(self, rotation_weight: float = 1.0):
        super().__init__()
        self.heatmap = HeatmapFocalLoss()
        self.rotation_weight = rotation_weight

    def forward(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor],
        targets: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        logits, rotation = outputs
        heatmaps, rotation_target = targets
        rotation_error = (rotation_from_6d(rotation) - rotation_target).pow(2).sum(dim=(1, 2)).mean()
        return self.heatmap(logits, heatmaps) + self.rotation_weight * rotation_error


class SymmetricKeypointLoss(nn.Module):
    """
    KeypointLoss for keypoints in symmetric sets (symmetry.KeypointSymmetry). Per crop, the
    heatmap loss is taken under each allowed labelling and the lowest one counts, so the
    heatmaps never have to tell turned twins apart. The rotation target is the true rotation
    in that labelling, and the turn classifier learns which labelling was used.

    Targets: true-labelled heatmaps [B, K, H, W], true allocentric rotation [B, 3, 3], the
    labellings each crop may use [B, G] (bool) and, optionally, how well each crop's view can tell
    each turn's twins apart [B, G] (symmetry.TurnDistinctness.informativeness) and a turn-loss
    weight per crop [B] (symmetry.twin_balance_weights). Where a view can't tell twins apart, the
    classifier's target is split evenly between them, so it learns to say "can't tell" instead of
    memorising a guess.
    """

    def __init__(self, symmetry, rotation_weight: float = 1.0, turn_weight: float = 1.0, gamma: float = 2.0):
        super().__init__()
        from ..symmetry import compose_table

        self.register_buffer("permutation", torch.as_tensor(symmetry.permutation, dtype=torch.long), persistent=False)
        self.register_buffer("turns", torch.as_tensor(symmetry.turns, dtype=torch.float32), persistent=False)
        self.register_buffer("compose", torch.as_tensor(compose_table(symmetry.turns), dtype=torch.long), persistent=False)
        self.rotation_weight = rotation_weight
        self.turn_weight = turn_weight
        self.gamma = gamma

    def forward(self, outputs: tuple[torch.Tensor, ...], targets: tuple[torch.Tensor, ...]) -> torch.Tensor:
        logits, rotation, turn_logits = outputs
        heatmaps, rotation_target, allowed = targets[:3]
        informativeness = targets[3] if len(targets) > 3 else None
        turn_weight = targets[4] if len(targets) > 4 else None
        permutation, turns = self.permutation.to(logits.device), self.turns.to(logits.device)
        probability = logits.sigmoid()
        per_turn = []
        for g in range(len(permutation)):
            target = heatmaps[:, permutation[g]]
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            weight = (target - probability).abs().pow(self.gamma)
            per_turn.append((weight * bce).flatten(1).sum(dim=1))
        per_turn = torch.stack(per_turn, dim=1).masked_fill(~allowed.bool(), float("inf"))
        choice = per_turn.argmin(dim=1)
        num_peaks = (heatmaps.flatten(2).amax(dim=2) > 0).sum().clamp_min(1)
        heatmap_loss = per_turn.gather(1, choice[:, None]).sum() / num_peaks
        rotation_target = rotation_target @ turns[choice]
        rotation_error = (rotation_from_6d(rotation) - rotation_target).pow(2).sum(dim=(1, 2)).mean()
        if informativeness is None:
            per_crop = F.cross_entropy(turn_logits, choice, reduction="none")
        else:
            turn_target = soft_turn_targets(choice, informativeness, self.compose.to(logits.device))
            per_crop = -(turn_target * turn_logits.log_softmax(dim=1)).sum(dim=1)
        turn_loss = (per_crop * turn_weight).mean() if turn_weight is not None else per_crop.mean()
        return heatmap_loss + self.rotation_weight * rotation_error + self.turn_weight * turn_loss


def soft_turn_targets(choice: torch.Tensor, informativeness: torch.Tensor, compose: torch.Tensor) -> torch.Tensor:
    """
    Turn-classifier targets [B, G]: all on the labelling used (``choice`` [B]) where the view tells
    the twins apart, split evenly with each turned twin as far as it doesn't (``informativeness``
    [B, G]). At most 3/4 moves off ``choice`` (a uniform target).
    """
    batch, count = informativeness.shape
    rows = torch.arange(batch, device=choice.device)
    moved = 0.5 * (1.0 - informativeness.float())
    moved[:, 0] = 0.0
    moved = moved * (0.75 / moved.sum(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
    target = torch.zeros(batch, count, device=choice.device)
    target[rows, choice] = 1.0
    for g in range(1, count):
        target[rows, choice] -= moved[:, g]
        target[rows, compose[choice, g]] += moved[:, g]
    return target


def rotation_from_6d(x: torch.Tensor) -> torch.Tensor:
    """[B, 6] (first two matrix columns, unnormalized) -> rotation matrices [B, 3, 3] by Gram-Schmidt."""
    a, b = x[:, :3], x[:, 3:]
    first = F.normalize(a, dim=1)
    second = F.normalize(b - (first * b).sum(dim=1, keepdim=True) * first, dim=1)
    return torch.stack([first, second, torch.linalg.cross(first, second)], dim=2)


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
