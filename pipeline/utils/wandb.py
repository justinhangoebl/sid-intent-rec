import wandb


def wandb_init(cfg: dict) -> None:
    """Initialize wandb with the given config and project name."""
    print(cfg.get("wandb", False))
    if cfg.get("wandb", False):
        return wandb.init(project=cfg.get("wandb_project", "default-project"), config=cfg)
        