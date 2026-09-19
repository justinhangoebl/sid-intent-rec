import torch


def seed_everything(seed: int) -> None:
    raise NotImplementedError


def resolve_device(device: str, gpu_id: int) -> torch.device:
    """`auto` picks cuda:<gpu_id> if available, else cpu."""
    raise NotImplementedError
