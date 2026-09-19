"""Per-user ranking metrics.

Each takes `hits` (B, K), 1 where the item at that rank is a target, and `n_targets` (B,),
and returns a (B,) tensor for cutoff k. Register new ones in `METRICS`.
"""

import torch


def recall(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    raise NotImplementedError

def precision(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    raise NotImplementedError

def ndcg(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    raise NotImplementedError

def hit(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    raise NotImplementedError

def mrr(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    raise NotImplementedError

"""Beyond Accuracy: coverage, novelty, diversity, serendipity, fairness, ..."""

def item_coverage(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    raise NotImplementedError

def catalog_coverage(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    raise NotImplementedError

def novelty(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    raise NotImplementedError

def diversity(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    raise NotImplementedError

def serendipity(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    raise NotImplementedError

def fairness(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    raise NotImplementedError
