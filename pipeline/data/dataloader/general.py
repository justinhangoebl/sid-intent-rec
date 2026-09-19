"""(user, item) style batches, no order. Used by TopPop, BPR and for evaluating any model."""

import torch

from pipeline.data.dataloader.base import BaseDataLoader
from pipeline.data.interaction import CSR, Interaction


class GeneralTrainLoader(BaseDataLoader):
    """Train split -> batches {"user", "pos_item", "neg_item"}. The negative is a uniformly drawn item
    (never the pad id 0) that the user has not interacted with in the train split."""

    def __init__(self, dataset, split, cfg):
        inter = dataset.interactions(split)
        self.users, self.items = inter["user"], inter["item"]
        self.seen = dataset.history("train")
        super().__init__(dataset, split, cfg)

    def n_samples(self) -> int:
        return len(self.users)

    def sample_negatives(self, users: torch.Tensor) -> torch.Tensor:
        n_items, rows = self.dataset.n_items, torch.arange(len(users))
        seen = self.seen.mask(users)
        neg = torch.randint(1, n_items, (len(users),))
        bad = seen[rows, neg]
        while bad.any():
            neg[bad] = torch.randint(1, n_items, (int(bad.sum()),))
            bad = seen[rows, neg]
        return neg

    def make_batch(self, idx: torch.Tensor) -> Interaction:
        users = self.users[idx]
        return Interaction({"user": users, "pos_item": self.items[idx], "neg_item": self.sample_negatives(users)})


class EvalLoader(BaseDataLoader):
    """Val/test split -> batches of users {"user", "seen", "target"}.
    seen:   (B, n_items) bool, items the model may see for that user (masked according to eval.seen_mode, see Evaluator)
    target: (B, n_items) bool, the held-out items of the split (multi-hot, so temporal/random splits work too)
    Iterates over users, not interactions, so the Evaluator can call model.generate on a whole batch."""

    def __init__(self, dataset, split, cfg):
        targets = dataset.interactions(split)
        self.users = targets["user"].unique()
        self.targets = CSR.from_interaction(targets, dataset.n_users, dataset.n_items)
        self.seen = dataset.history(split)
        super().__init__(dataset, split, cfg)

    def n_samples(self) -> int:
        return len(self.users)

    def make_batch(self, idx: torch.Tensor) -> Interaction:
        users = self.users[idx]
        return Interaction({"user": users, "seen": self.seen.mask(users), "target": self.targets.mask(users)})
