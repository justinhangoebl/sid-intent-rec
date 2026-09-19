from dataclasses import dataclass

import numpy as np
import torch

from pipeline.data.interaction import Interaction
from pipeline.metrics.ranking import METRICS
from pipeline.models.base import Recommender


@dataclass
class EvalResult:
    metrics: dict[str, float]  # "<metric>@<k>"
    users: np.ndarray | None = None
    recs: np.ndarray | None = None  # (n_users, max_k) item ids


SEEN_MODES = ("all", "except_target", "none")


class Evaluator:
    """Ranks with `model.generate` and averages the [eval] metrics over users.

    eval.seen_mode says which items the user has already seen are masked from the ranking:
      all            every seen item (the honest setting, harsh when users repeat items)
      except_target  every seen item except the held-out one (RecBole / hopwise `repeatable: False`)
      none           nothing is masked
    """

    def __init__(self, cfg: dict, device):
        self.ks = sorted(cfg["eval"]["ks"])
        self.metrics = cfg["eval"]["metrics"]
        self.seen_mode = cfg["eval"]["seen_mode"]
        if self.seen_mode not in SEEN_MODES:
            raise SystemExit(f"eval.seen_mode must be one of {SEEN_MODES}, got {self.seen_mode!r}")
        self.device = device
        unknown = [m for m in self.metrics if m not in METRICS]
        if unknown:
            raise SystemExit(f"Unknown metrics {unknown}; available: {sorted(METRICS)}")
        # with one held-out item per user hit@k and recall@k are the same number: report them as one "recall/hit"
        merge = {"hit", "recall"} <= set(self.metrics) and cfg["split"]["strategy"] == "leave_one_out"
        self.names = list(dict.fromkeys("recall/hit" if merge and m in ("hit", "recall") else m for m in self.metrics))

    @torch.no_grad()
    def evaluate(self, model: Recommender, loader, keep_recs: bool = False) -> EvalResult:
        model.eval()
        totals = {f"{m}@{k}": 0.0 for m in self.names for k in self.ks}
        n_users, users, recs = 0, [], []
        for batch in loader:
            batch = batch.to(self.device)
            target = batch["target"]
            if self.seen_mode == "except_target":
                batch = Interaction({**batch.data, "seen": batch["seen"] & ~target})
            items = model.generate(batch, k=self.ks[-1], filter_seen=self.seen_mode != "none")
            hits = target.gather(1, items).float()
            n_targets = target.sum(1)
            for m in self.names:
                fn = METRICS["recall" if m == "recall/hit" else m]
                for k in self.ks:
                    totals[f"{m}@{k}"] += fn(hits, n_targets, k).sum().item()
            n_users += len(items)
            if keep_recs:
                users.append(batch["user"].cpu())
                recs.append(items.cpu())
        result = EvalResult({name: total / n_users for name, total in totals.items()})
        if keep_recs:
            result.users, result.recs = torch.cat(users).numpy(), torch.cat(recs).numpy()
        return result
