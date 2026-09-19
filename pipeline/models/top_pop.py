import torch

from pipeline.models.base import Recommender


class TopPop(Recommender):
    """Recommends the most popular items of the train split, the same for every user."""

    DATALOADER = "general"
    NEURAL = False

    def __init__(self, cfg: dict, dataset):
        super().__init__(cfg, dataset)
        self.register_buffer("counts", torch.zeros(self.n_items))

    def fit(self, loader):
        self.counts.copy_(torch.bincount(loader.items, minlength=self.n_items).float())

    def forward(self, batch):
        return self.counts.expand(len(batch["user"]), -1)
