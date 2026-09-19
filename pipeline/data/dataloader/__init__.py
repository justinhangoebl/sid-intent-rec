from pipeline.data.dataloader.general import GeneralTrainLoader, EvalLoader
from pipeline.data.dataloader.sequential import SequentialTrainLoader, SequentialEvalLoader

DATALOADERS = {
    "general": (GeneralTrainLoader, EvalLoader),
    "sequential": (SequentialTrainLoader, SequentialEvalLoader),
}


def get_dataloader(name: str, train: bool):
    """Return the train or eval dataloader class for a given model family."""
    if name not in DATALOADERS:
        raise ValueError(f"Unknown dataloader name: {name}.")
    return DATALOADERS[name][0] if train else DATALOADERS[name][1]