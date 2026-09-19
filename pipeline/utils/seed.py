import random
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str, gpu_id: int) -> torch.device:
    """`auto` picks cuda:<gpu_id> if available, else cpu. `cuda` fails instead of falling back to the cpu."""
    if device == "cpu":
        return torch.device("cpu")
    if device in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    if device == "auto":
        return torch.device("cpu")
    raise SystemExit(f"run.device={device!r} but CUDA is not available")
