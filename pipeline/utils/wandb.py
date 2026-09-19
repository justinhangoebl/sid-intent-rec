import wandb


def wandb_init(cfg: dict, name: str | None = None, tags: list[str] | None = None):
    """Start a wandb run when `run.wandb` is true, else return None. The whole config is stored with the run."""
    run_cfg = cfg["run"]
    if not run_cfg.get("wandb", False):
        return None
    config = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return wandb.init(project=run_cfg.get("wandb_project", "default-project"), name=name, tags=tags, config=config)


def wandb_watch(wandb_run, model, run_cfg: dict) -> None:
    """Log gradients / parameters of a model with weights, as set by `run.wandb_watch*`."""
    if wandb_run is None or not run_cfg.get("wandb_watch", False) or not any(True for _ in model.parameters()):
        return
    wandb_run.watch(model, log=run_cfg.get("wandb_watch_log", "gradients"), log_freq=run_cfg.get("wandb_watch_freq", 100))
