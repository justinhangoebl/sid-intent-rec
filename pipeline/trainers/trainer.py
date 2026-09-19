from torch.utils.data import Dataset

from pipeline.models.recommender import Recommender


class Trainer:
    """Gradient training loop for any `Recommender`: DataLoader with `model.collate`, `loss = model(batch)`,
    AdamW, warmup + linear decay, grad clipping. Hyper-parameters come from `model.params`.
    Calls `on_epoch({"epoch", "loss", "lr"})` after every epoch and stops if it returns True."""

    def __init__(self, model: Recommender, dataset: Dataset, device, num_workers: int = 0):
        raise NotImplementedError

    def fit(self, on_epoch=None) -> None:
        raise NotImplementedError


class Validator:
    """`on_epoch` hook: validates every `eval.val_every` epochs, keeps the best weights by `eval.monitor`,
    stops after `eval.patience` validations without improvement. `finish()` restores the best weights
    and returns their validation metrics."""

    def __init__(self, model: Recommender, evaluator, loader, data, cfg: dict, tracker):
        raise NotImplementedError

    def __call__(self, train_metrics: dict) -> bool:
        raise NotImplementedError

    def finish(self) -> dict[str, float]:
        raise NotImplementedError
