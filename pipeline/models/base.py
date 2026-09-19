import torch
from torch import nn
from math import inf

from pipeline.config import with_defaults


class Recommender(nn.Module):
    """`forward(batch)` returns (B, n_items) scores; `generate` turns them into the top-k items.
    Neural models also implement `loss(batch)`. Models without scores (TIGER) override `generate`."""

    DEFAULTS = {}            # hyper-parameters; model.params is checked against these
    DATALOADER = "general"   # which loader family the run builds
    NEURAL = False           # False: run calls fit() once, True: the Trainer runs

    def __init__(self, cfg, dataset):
        super().__init__()
        self.params = with_defaults(cfg["model"]["params"], self.DEFAULTS, "model.params")
        self.n_users, self.n_items = dataset.n_users, dataset.n_items

    def fit(self, loader):
        """Non-neural models learn here from the train loader."""

    def forward(self, batch):
        raise NotImplementedError

    def loss(self, batch):
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, batch, k, filter_seen=True):
        s = self(batch).clone()
        s[:, 0] = -inf                          # never recommend the pad id
        if filter_seen:
            s[batch["seen"]] = -inf
        return s.topk(k).indices
