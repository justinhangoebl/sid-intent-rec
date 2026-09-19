"""Ordered histories. Used by SASRec."""

import torch

from pipeline.data.dataloader.base import BaseDataLoader
from pipeline.data.dataloader.general import EvalLoader
from pipeline.data.interaction import Interaction


def last_items(flat: torch.Tensor, start: torch.Tensor, end: torch.Tensor, max_len: int):
    """The last `max_len` items of flat[start:end] per row, left-padded with 0. Returns ((B, L) items, (B,) lengths)."""
    idx = end[:, None] - max_len + torch.arange(max_len)
    valid = idx >= start[:, None]
    return torch.where(valid, flat[idx.clamp(min=0)], torch.zeros_like(idx)), valid.sum(1)


class SequentialTrainLoader(BaseDataLoader):
    """Train split -> {"user", "history" (B, L) left-padded with 0, "history_len", "target"}.
    Every train interaction with at least one earlier train item is one sample: its predecessors
    (the last L of them) are the history, the item itself is the target."""

    def __init__(self, dataset, split, cfg):
        df = dataset.df[dataset.df["split"] == split]  # sorted by (user, timestamp)
        self.flat = torch.as_tensor(df["item"].to_numpy(), dtype=torch.long)
        self.user_of = torch.as_tensor(df["user"].to_numpy(), dtype=torch.long)
        pos_in_user = torch.as_tensor(df.groupby("user").cumcount().to_numpy(), dtype=torch.long)
        self.row_start = torch.arange(len(df)) - pos_in_user
        self.samples = (pos_in_user >= 1).nonzero().squeeze(1)  # rows that have a history
        super().__init__(dataset, split, cfg)

    def n_samples(self) -> int:
        return len(self.samples)

    def make_batch(self, idx: torch.Tensor) -> Interaction:
        rows = self.samples[idx]
        history, length = last_items(self.flat, self.row_start[rows], rows, self.max_len)
        return Interaction({"user": self.user_of[rows], "history": history, "history_len": length, "target": self.flat[rows]})


class SequentialEvalLoader(EvalLoader):
    """Val/test -> EvalLoader batches plus "history" (B, L) and "history_len": the last L items the model
    may see (train for val, train + val for test), in time order."""

    def __init__(self, dataset, split, cfg):
        super().__init__(dataset, split, cfg)
        visible = ["train"] if split == "val" else ["train", "val"]
        df = dataset.df[dataset.df["split"].isin(visible)]
        self.flat = torch.as_tensor(df["item"].to_numpy(), dtype=torch.long)
        counts = torch.bincount(torch.as_tensor(df["user"].to_numpy(), dtype=torch.long), minlength=dataset.n_users)
        self.end = counts.cumsum(0)
        self.start = self.end - counts

    def make_batch(self, idx: torch.Tensor) -> Interaction:
        batch = super().make_batch(idx)
        users = batch["user"]
        history, length = last_items(self.flat, self.start[users], self.end[users], self.max_len)
        return Interaction({**batch.data, "history": history, "history_len": length})
