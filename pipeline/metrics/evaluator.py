from dataclasses import dataclass

import numpy as np

from pipeline.data.interactions import CSR
from pipeline.models.base import Recommender


@dataclass
class EvalResult:
    metrics: dict[str, float]  # "<metric>@<k>" and coverage
    users: np.ndarray | None = None
    recs: np.ndarray | None = None  # (n_users, max_k) item ids


class Evaluator:
    """Ranks with `model.generate` and averages the [eval] metrics over users."""

    def __init__(self, cfg: dict, device):
        raise NotImplementedError

    def evaluate(self, model: Recommender, loader, targets: CSR, history: CSR, keep_recs: bool = False) -> EvalResult:
        raise NotImplementedError
