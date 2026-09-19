"""Per-user ranking metrics.

Each takes `hits` (B, K), 1 where the item at that rank is a target, and `n_targets` (B,),
and returns a (B,) tensor for cutoff k. Register new ones in `METRICS`.
"""

import torch


def recall(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    return hits[:, :k].sum(1) / n_targets.clamp(min=1)

def precision(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    return hits[:, :k].sum(1) / k

def ndcg(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    discount = 1.0 / torch.log2(torch.arange(2, k + 2, device=hits.device, dtype=hits.dtype))
    dcg = (hits[:, :k] * discount).sum(1)
    idcg = discount.cumsum(0)[n_targets.clamp(min=1, max=k) - 1]  # all targets ranked first
    return dcg / idcg

def hit(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    return (hits[:, :k].sum(1) > 0).to(hits.dtype)

def mrr(hits: torch.Tensor, n_targets: torch.Tensor, k: int) -> torch.Tensor:
    ranks = torch.arange(1, k + 1, device=hits.device, dtype=hits.dtype)
    return (hits[:, :k] / ranks).max(1).values  # the first hit has the largest 1/rank


METRICS = {"recall": recall, "precision": precision, "ndcg": ndcg, "hit": hit, "mrr": mrr}

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
