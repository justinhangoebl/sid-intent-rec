import torch


class RKMeans:
    """Residual k-means: each level clusters what the previous levels left unexplained."""

    def fit(self, x: torch.Tensor) -> None:
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(n, n_levels) integer codes."""
        raise NotImplementedError

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruction from codes."""
        raise NotImplementedError
