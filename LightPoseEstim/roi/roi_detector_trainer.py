from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from tqdm import tqdm

from ..dataloader import ImageROIDataset
from .roi_detector import ROIDetector


class ROIDetectorTrainer:
    def __init__(self,
                 dataset: ImageROIDataset,
                 model: ROIDetector | type[ROIDetector] = ROIDetector,
                 train_size: float = 0.8,
                 batch_size: int = 32,
                 criterion: nn.Module = nn.SmoothL1Loss(),
                 optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
                 optimizer_kwargs: dict | None = None,
                 device: torch.device | None = None,
                 use_tensorboard: bool = False,
                 tensorboard_log_dir: str = "runs/roi_detector",
                 ):

        # Build DataLoaders
        train_size = int(len(dataset) * train_size)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size]
        )
        self.batch_size = batch_size
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Build Model
        self.model = model() if isinstance(model, type) else model

        # Store Criterion
        self.criterion = criterion

        # Build Optimizer
        if optimizer_kwargs is None:
            if optimizer_cls == torch.optim.Adam:
                optimizer_kwargs = dict(lr=1e-4)
            else:
                raise ValueError(f"Pass in optimizer_kwargs for {optimizer_cls}")
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_kwargs)

        # Store Device
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.use_tensorboard = use_tensorboard
        self.tensorboard_log_dir = tensorboard_log_dir

    def training_epoch(self, pbar: tqdm):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        for batch_idx, (images, rois) in enumerate(self.train_loader):
            images = images.to(self.device)
            rois = rois.to(self.device)

            predictions = self.model(images)
            loss = self.criterion(predictions, rois)

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

        for batch_idx, (images, rois) in enumerate(self.val_loader):
            images = images.to(self.device)
            rois = rois.to(self.device)

            predictions = self.model(images)

            loss = self.criterion(predictions, rois)

            total_loss += loss.item()
            pbar.set_postfix(
                device=str(self.device),
                batch_size=self.batch_size,
                phase="val",
                batch=f"{batch_idx + 1}/{num_batches}",
                loss=f"{loss.item():.4f}",
            )

        return total_loss / num_batches

    def train(self, epochs: int, save_path: str | None = None, freeze_backbone: bool = False):
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
        try:
            pbar = tqdm(range(epochs), unit="epoch")
            for epoch in pbar:
                train_loss = self.training_epoch(pbar)
                val_loss = self.validation_epoch(pbar)
                pbar.set_postfix(
                    device=str(self.device),
                    batch_size=self.batch_size,
                    phase="done",
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                )
                if writer is not None:
                    writer.add_scalar("loss/train", train_loss, epoch)
                    writer.add_scalar("loss/val", val_loss, epoch)
        finally:
            if writer is not None:
                writer.close()

        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)

        if frozen and hasattr(self.model, "unfreeze_backbone"):
            self.model.unfreeze_backbone()

        if unfrozen and hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()

        return self.model
