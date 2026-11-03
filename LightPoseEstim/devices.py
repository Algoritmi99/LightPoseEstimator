"""The device the package runs on unless told otherwise: the GPU when CUDA is available."""
import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """``device`` as a torch.device; ``None`` is CUDA when it is available, else the CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
