import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..dataloader import ImageKPHeatmapROIDataset, RecordingSplit, split_by_recording
from .keypoint_heatmap import HeatmapFocalLoss, KeypointHeatmapNet


class KeypointHeatmapTrainer:
    def __init__(
        self,
        dataset: ImageKPHeatmapROIDataset,
        model: KeypointHeatmapNet | type[KeypointHeatmapNet] | None = None,
        split: RecordingSplit | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        criterion: nn.Module = HeatmapFocalLoss(),
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
        device: torch.device | None = None,
        use_tensorboard: bool = False,
        tensorboard_log_dir: str = "runs/keypoint_heatmap",
        augment: bool = True,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size
        self.pin_memory = self.device.type == "cuda"

        if split is None:
            split = split_by_recording(dataset.df)
        # Augment training crops only; validation measures clean crops.
        train_dataset = Subset(dataset.with_augmentation(augment), split.indices(dataset.df, "train"))
        val_dataset = Subset(dataset.with_augmentation(False), split.indices(dataset.df, "val"))
        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
        )
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
        self.train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

        if model is None:
            model = KeypointHeatmapNet(num_keypoints=len(dataset.keypoints))
        elif isinstance(model, type):
            model = model(num_keypoints=len(dataset.keypoints))
        self.model = model

        self.criterion = criterion

        if optimizer_kwargs is None:
            if optimizer_cls == torch.optim.Adam:
                optimizer_kwargs = dict(lr=1e-4)
            else:
                raise ValueError(f"Pass in optimizer_kwargs for {optimizer_cls}")
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)

        self.model.to(self.device)

        self.use_tensorboard = use_tensorboard
        self.tensorboard_log_dir = tensorboard_log_dir

    def training_epoch(self, pbar: tqdm):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        for batch_idx, (images, heatmaps) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=self.pin_memory)
            heatmaps = heatmaps.to(self.device, non_blocking=self.pin_memory)

            predictions = self.model(images)
            loss = self.criterion(predictions, heatmaps)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(
                device=str(self.device),
                batch_size=self.batch_size,
                phase="train",
                batch=f"{batch_idx + 1}/{num_batches}",
                loss=f"{loss.item():.4f}",
            )

        return total_loss / num_batches

    @torch.no_grad()
    def validation_epoch(self, pbar: tqdm):
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        for batch_idx, (images, heatmaps) in enumerate(self.val_loader):
            images = images.to(self.device, non_blocking=self.pin_memory)
            heatmaps = heatmaps.to(self.device, non_blocking=self.pin_memory)

            predictions = self.model(images)
            loss = self.criterion(predictions, heatmaps)

            total_loss += loss.item()
            pbar.set_postfix(
                device=str(self.device),
                batch_size=self.batch_size,
                phase="val",
                batch=f"{batch_idx + 1}/{num_batches}",
                loss=f"{loss.item():.4f}",
            )

        return total_loss / num_batches

    def train(
        self,
        epochs: int,
        save_path: str | Path | None = None,
        freeze_backbone: bool = False,
    ):
        frozen = False
        unfrozen = False
        if freeze_backbone and hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()
            frozen = True
        elif hasattr(self.model, "unfreeze_backbone"):
            self.model.unfreeze_backbone()
            unfrozen = True

        writer = (
            SummaryWriter(log_dir=self.tensorboard_log_dir)
            if self.use_tensorboard
            else None
        )
        best_val_loss = float("inf")
        best_epoch = -1
        checkpoint_path = Path(save_path) if save_path is not None else None

        try:
            pbar = tqdm(range(epochs), unit="epoch")
            for epoch in pbar:
                train_loss = self.training_epoch(pbar)
                val_loss = self.validation_epoch(pbar)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    if checkpoint_path is not None:
                        torch.save(self.model.state_dict(), checkpoint_path)
                        logging.info(
                            "New best val loss %.4f at epoch %d, saved to %s",
                            val_loss,
                            epoch,
                            checkpoint_path,
                        )
                pbar.set_postfix(
                    device=str(self.device),
                    batch_size=self.batch_size,
                    phase="done",
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                    best_val=f"{best_val_loss:.4f}",
                )
                if writer is not None:
                    writer.add_scalar("loss/train", train_loss, epoch)
                    writer.add_scalar("loss/val", val_loss, epoch)
        finally:
            if writer is not None:
                writer.close()

        if checkpoint_path is not None and best_epoch >= 0:
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            )
            logging.info(
                "Restored best checkpoint from epoch %d (val loss %.4f)",
                best_epoch,
                best_val_loss,
            )

        if frozen and hasattr(self.model, "unfreeze_backbone"):
            self.model.unfreeze_backbone()

        if unfrozen and hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()

        return self.model
