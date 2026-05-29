from torch.utils.data import DataLoader, random_split
import torch
from torch import nn

from LightPoseEstim.dataloader import ImageROIDataset
from .roi_detector import ROIDetector


class ROIDetectorTrainer:
    def __init__(self,
                 dataset: ImageROIDataset,
                 model: ROIDetector | type[ROIDetector] = ROIDetector,
                 train_size: float = 0.8,
                 batch_size: int = 32,
                 criterion: nn.Module = nn.SmoothL1Loss(),
                 optimizer_cls: type[torch.optim.optimizer.Optimizer] = torch.optim.Adam,
                 optimizer_kwargs: dict | None = None,
                 device: torch.device | None = None,
                 ):

        # Build DataLoaders
        train_size = int(len(dataset) * train_size)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size]
        )
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
        self.device = device if device is not None else torch.device.cuda() if torch.cuda.is_available() else torch.device.cpu()

    def training_epoch(self):
        pass

    def validation_epoch(self):
        pass

    def train(self, epochs: int, save_path: str | None = None, freeze_backbone: bool = False):
        pass