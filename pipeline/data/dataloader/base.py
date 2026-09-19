"""Turns a Dataset + split into batches of `Interaction`. One subclass per model family."""

import torch
from torch.utils.data import DataLoader

from pipeline.data.dataset.base import Dataset
from pipeline.data.interaction import Interaction


class BaseDataLoader:
    """Model families declare which one they use via `DATALOADER = "<name>"`.

    The data is in memory, so torch's DataLoader only hands out lists of row indices. Subclasses say how
    many rows there are (`n_samples`) and turn an index tensor into a batch (`make_batch`).
    Batches stay on the CPU: move them with `batch.to(device)` in the training / eval loop.
    """

    def __init__(self, dataset: Dataset, split: str, cfg: dict):
        self.dataset = dataset
        self.split = split
        self.cfg = cfg
        self.device = cfg.get("run", {}).get("device", "cpu")
        train = split == "train"
        params = cfg.get("model", {}).get("params") or {}
        self.batch_size = cfg.get("model", {}).get("batch_size", 512) if train else cfg["eval"]["batch_size"]
        self.max_len = params.get("max_len", 50)  # only used by the sequential loaders
        self._loader = DataLoader(
            range(self.n_samples()),
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=cfg.get("run", {}).get("num_workers", 0),
            collate_fn=self._collate,
        )

    def n_samples(self) -> int:
        """Number of rows (train) or users (eval)."""
        raise NotImplementedError

    def make_batch(self, idx: torch.Tensor) -> Interaction:
        """Index tensor -> batch."""
        raise NotImplementedError

    def _collate(self, idx: list[int]) -> Interaction:
        return self.make_batch(torch.as_tensor(idx))

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)
