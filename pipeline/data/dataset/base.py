"""Dataset: files -> filtered, remapped, split interactions. It knows nothing about models."""

import numpy as np
import pandas as pd
import torch

from pipeline.data.interaction import CSR, Interaction
from pipeline.data.utils import k_core, remap_ids, split

SPLITS = ("train", "val", "test")


class Dataset:
    """Loads one interaction file, filters it, remaps ids and splits it.

    Ids: users are 0..n_users-1. Items are 1..n_items-1, and 0 is the padding id, so `n_items`
    includes the pad and can be used directly as the size of an embedding table or score vector.
    """

    def __init__(self, cfg: dict):
        self.data_cfg = cfg.get("data", {})
        self.split_cfg = cfg.get("split", {})
        self.feature_cfg = cfg.get("features", {})

        df = self._load_raw(self.data_cfg["path"])

        min_user, min_item = self.data_cfg.get("min_user_inter"), self.data_cfg.get("min_item_inter")
        if min_user is not None and min_item is not None:
            df = k_core(df, min_user, min_item)

        df, self.user_map, self.item_map = remap_ids(df)
        self.n_users = len(self.user_map)
        self.n_items = len(self.item_map) + 1  # + padding id 0

        strategy = self.split_cfg.get("strategy", "leave_one_out")
        if strategy != "leave_one_out":  # temporal / random need a global time order
            df = df.sort_values("timestamp", kind="stable")
        df = split(df, strategy, self.split_cfg.get("val_ratio"), self.split_cfg.get("test_ratio"))
        self.df = df.sort_values(["user", "timestamp"], kind="stable").reset_index(drop=True)

        self.items = None
        if self.feature_cfg.get("path") is not None:
            self.items = self._load_items(self.feature_cfg["path"])

    def _load_raw(self, path: str) -> pd.DataFrame:
        """Return a DataFrame with columns user, item, timestamp (raw ids, timestamp in seconds)."""
        user_col = self.data_cfg["user_col"]
        item_col = self.data_cfg["item_col"]
        time_col = self.data_cfg["time_col"]

        sep = self.data_cfg.get("sep", "\t")
        df = pd.read_csv(path, sep=sep, usecols=[user_col, item_col, time_col])
        df = df.rename(columns={user_col: "user", item_col: "item", time_col: "timestamp"})

        if not pd.api.types.is_numeric_dtype(df["timestamp"]):  # onion has "2020-03-20 12:59:43"
            df["timestamp"] = (pd.to_datetime(df["timestamp"]) - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        df["timestamp"] = df["timestamp"].astype("int64")
        return df

    def _load_items(self, path: str) -> pd.DataFrame:
        """Item features / metadata, indexed by raw item id."""
        id_col = self.feature_cfg.get("id_col")
        sep = self.feature_cfg.get("sep", "\t")
        drop_cols = self.feature_cfg.get("drop_cols")
        return pd.read_csv(path, sep=sep, index_col=id_col if id_col is not None else 0).drop(columns=drop_cols or [], errors="ignore")

    def summary(self) -> dict[str, float]:
        """Totals of the filtered dataset. `repeats` is the share of rows that repeat an earlier (user, item) pair."""
        n = len(self.df)
        pairs = self.df[["user", "item"]].drop_duplicates().shape[0]
        return {
            "users": self.n_users,
            "items": self.n_items - 1,
            "interactions": n,
            "avg per user": n / self.n_users,
            "sparsity": 1 - pairs / (self.n_users * (self.n_items - 1)),
            "repeats": 1 - pairs / n,
        }

    def split_stats(self) -> list[dict]:
        """One row per split. `target seen` is the share of val/test interactions whose item is already in the
        history the model sees (train for val, train + val for test): the rows seen_mode makes a difference for."""
        df = self.df
        key = df["user"].to_numpy(dtype=np.int64) * self.n_items + df["item"].to_numpy(dtype=np.int64)
        is_split = {s: (df["split"] == s).to_numpy() for s in SPLITS}
        visible = {"val": is_split["train"], "test": is_split["train"] | is_split["val"]}
        rows = []
        for s in SPLITS:
            m = is_split[s]
            rows.append({
                "split": s,
                "interactions": int(m.sum()),
                "share": m.mean(),
                "users": int(df.loc[m, "user"].nunique()),
                "items": int(df.loc[m, "item"].nunique()),
                "target seen": np.isin(key[m], key[visible[s]]).mean() if s in visible else None,
            })
        return rows

    def interactions(self, split: str) -> Interaction:
        """{"user", "item", "timestamp"} of one split."""
        assert split in SPLITS, f"split must be one of {SPLITS}"
        part = self.df[self.df["split"] == split]
        return Interaction({col: torch.as_tensor(part[col].to_numpy().copy(), dtype=torch.long) for col in ("user", "item", "timestamp")})

    def history(self, split: str) -> CSR:
        """What the model may see when predicting `split`: train for val, train + val for test."""
        visible = {"train": ["train"], "val": ["train"], "test": ["train", "val"]}[split]
        part = self.df[self.df["split"].isin(visible)]
        inter = Interaction({col: torch.as_tensor(part[col].to_numpy().copy(), dtype=torch.long) for col in ("user", "item")})
        return CSR.from_interaction(inter, self.n_users, self.n_items)

    def item_features(self, cols=None) -> torch.Tensor:
        """(n_items, d) float tensor aligned to item ids (row 0 is the padding item), for the quantizer."""
        assert self.items is not None, "set features.path in the config"
        cols = cols or self.feature_cfg.get("cols")
        feats = self.items
        if self.feature_cfg.get("drop_cols"):
            feats = feats.drop(columns=self.feature_cfg.get("drop_cols") or [], errors="ignore")
        feats = feats[cols] if cols else feats
        feats = feats.select_dtypes("number").reindex(list(self.item_map))  # item_map is ordered by new id
        assert not feats.isna().any().any(), "some items have no features"

        x = torch.as_tensor(feats.to_numpy(), dtype=torch.float32)
        if self.feature_cfg.get("normalize") == "standardize":
            x = (x - x.mean(0)) / x.std(0).clamp_min(1e-8)
        return torch.cat([torch.zeros(1, x.shape[1]), x])
