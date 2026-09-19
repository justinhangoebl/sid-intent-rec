"""CLI entry points.

Examples:
    uv run recommender train --model TopPop --config configs/rec.yaml
    uv run recommender infer --model TopPop --config configs/rec.yaml
    uv run recommender optimize --model TopPop --config configs/rec.yaml

    uv run quantizer train --quantizer RKMeans --config configs/quant.yaml
    uv run quantizer infer --quantizer RKMeans --config configs/quant.yaml
    uv run quantizer optimize --quantizer RKMeans --config configs/quant.yaml
"""

from pathlib import Path

import click
from pprint import pformat
from pipeline.config import load_config
from pipeline.recommender import MODELS, run as run_recommender
from pipeline.utils.logging import get_logger, log_cfg
from pipeline.utils.wandb import wandb_init

TASKS = ("train", "infer", "optimize")

logger = get_logger(__name__)


@click.command()
@click.argument("task", type=click.Choice(TASKS))
@click.option(
    "--model",
    required=True,
    type=click.Choice(list(MODELS), case_sensitive=False),
    help="Model name.",
)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="Path to the config file.",
)
def recommender(task: str, model: str, config_path: str) -> None:
    """Run a recommender training, inference, or optimization task."""
    cfg = load_config(config_path)
    log_cfg(cfg)
    dataset_name = Path(cfg["data"]["path"]).parent.name
    wandb_run = wandb_init(
        cfg,
        name=f"{model}_{dataset_name}",
        tags=[model, dataset_name, f"seen:{cfg['eval']['seen_mode']}"],
    )

    logger.info(
        "Running recommender task=%s model=%s config=%s",
        task,
        model,
        config_path,
    )
    if task != "train":
        raise click.UsageError(f"task {task!r} is not implemented yet")
    try:
        run_recommender(cfg, model, wandb_run)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


@click.command()
@click.argument("task", type=click.Choice(TASKS))
@click.option(
    "--quantizer",
    required=True,
    help="Quantizer name (e.g., RKMeans).",
)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="Path to the config file.",
)
def quantizer(task: str, quantizer: str, config_path: str) -> None:
    """Run a quantizer training, inference, or optimization task."""
    cfg = load_config(config_path)
    log_cfg(cfg)
    wandb_run = wandb_init(cfg, name=quantizer, tags=[quantizer])

    logger.info(
        "Running quantizer task=%s quantizer=%s config=%s",
        task,
        quantizer,
        config_path,
    )