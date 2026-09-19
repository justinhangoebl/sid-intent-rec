"""Batch container and sparse user-item matrix. Models and the evaluator only ever see these."""

import torch


class Interaction:
    """Dict of equally long tensors, e.g. {"user": (N,), "item": (N,)} or {"history": (N, L), ...}."""

    def __init__(self, data: dict[str, torch.Tensor]):
        self.data = data
        assert len(set(len(v) for v in data.values())) <= 1, "All tensors must have the same length"
        assert len(data) > 0, "Data cannot be empty"
    
    def __len__(self):
        return len(next(iter(self.data.values())))
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        else:
            return Interaction({k: v[key] for k, v in self.data.items()})
        
    def to(self, device):
        return Interaction({k: v.to(device) for k, v in self.data.items()})
    
    def keys(self):
        return self.data.keys()


class CSR:
    """User -> items in compressed sparse row form (indptr, indices), built from an Interaction."""

    def __init__(self, indptr: torch.Tensor, indices: torch.Tensor, n_items: int):
        self.indptr = indptr
        self.indices = indices
        self.n_items = n_items
        
    @staticmethod
    def from_interaction(inter: Interaction, n_users: int, n_items: int):
        """Build a CSR from an Interaction with keys "user" and "item"."""
        user = inter["user"]
        item = inter["item"]
        # Create a list of lists, where each list contains the items for each user
        user_items = [[] for _ in range(n_users)]
        for u, i in zip(user, item):
            user_items[u].append(i)
        # Convert to CSR format
        indptr = [0]
        indices = []
        for items in user_items:
            indptr.append(indptr[-1] + len(items))
            indices.extend(items)
        return CSR(torch.tensor(indptr), torch.tensor(indices), n_items)

    def items_of(self, user: int) -> torch.Tensor:
        """Return the items of one user."""
        start = self.indptr[user]
        end = self.indptr[user + 1]
        return self.indices[start:end]

    def mask(self, users: torch.Tensor) -> torch.Tensor:
        """Return a (B, n_items) bool tensor, True where the user has the item (used to mask seen items)."""
        B = len(users)
        mask = torch.zeros((B, self.n_items), dtype=torch.bool)
        for i, user in enumerate(users):
            items = self.items_of(user)
            mask[i, items] = True
        return mask
        
    def counts(self) -> torch.Tensor:
        """Return a (n_items,) tensor with the number of interactions per item."""
        counts = torch.zeros(self.n_items, dtype=torch.long)
        for i in self.indices:
            counts[i] += 1
        return counts
