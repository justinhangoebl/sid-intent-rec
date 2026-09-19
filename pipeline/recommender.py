"""Recommendation stage: build the dataset, model and loaders, fit, evaluate on val and test."""

import json
from importlib import import_module
from pathlib import Path

from pipeline.data.dataloader import get_dataloader
from pipeline.data.dataset import Dataset
from pipeline.metrics.evaluator import Evaluator
from pipeline.utils.logging import get_logger, log_dataset, log_metrics
from pipeline.utils.wandb import wandb_watch
from pipeline.utils.seed import resolve_device, seed_everything

logger = get_logger(__name__)

MODELS = {
    "TopPop": "pipeline.models.top_pop.TopPop",
    "BPR": "pipeline.models.bpr.BPR",
}


def recommender_factory(model_name: str, cfg: dict, dataset):
    """Create a recommender by name (case-insensitive)."""
    by_lower = {name.lower(): name for name in MODELS}
    if model_name.lower() not in by_lower:
        raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODELS)}")

    module_path, class_name = MODELS[by_lower[model_name.lower()]].rsplit(".", 1)
    return getattr(import_module(module_path), class_name)(cfg, dataset)


def run(cfg: dict, model_name: str, wandb_run=None) -> dict[str, dict[str, float]]:
    """Fit `model_name`, evaluate on val and test, log and save the metrics. Returns {"val": {...}, "test": {...}}."""
    run_cfg = cfg["run"]
    seed_everything(run_cfg["seed"])
    device = resolve_device(run_cfg["device"], run_cfg["gpu_id"])

    dataset = Dataset(cfg)
    log_dataset(dataset.summary(), dataset.split_stats(), title=f"Dataset: {Path(cfg['data']['path']).parent.name}")
    model = recommender_factory(model_name, cfg, dataset).to(device)
    wandb_watch(wandb_run, model, run_cfg)

    train_loader = get_dataloader(model.DATALOADER, train=True)(dataset, "train", cfg)
    if model.NEURAL:
        raise NotImplementedError("Neural models need the Trainer, which is not implemented yet")
    model.fit(train_loader)

    evaluator = Evaluator(cfg, device)
    results = {}
    for split in ("val", "test"):
        loader = get_dataloader(model.DATALOADER, train=False)(dataset, split, cfg)
        results[split] = evaluator.evaluate(model, loader).metrics
        if wandb_run is not None:
            wandb_run.log({f"{split}/{name}": value for name, value in results[split].items()})
    log_metrics(results, evaluator.names, evaluator.ks, title=f"{model_name} results", caption=f"seen_mode: {evaluator.seen_mode}")

    dataset_name = Path(cfg["data"]["path"]).parent.name
    out_dir = Path(run_cfg["out_dir"]) / f"{model_name}_{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": model_name, "dataset": dataset_name, "seen_mode": evaluator.seen_mode, "results": results}
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    return results
