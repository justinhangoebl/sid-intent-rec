"""Helpers for Dataset: filtering, id remapping, splitting."""

import pandas as pd


def k_core(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Drop users/items below the thresholds, repeat until nothing changes."""
    copy_df = df.copy()
    while True:
        n_users_before = copy_df["user"].nunique()
        n_items_before = copy_df["item"].nunique()

        user_counts = copy_df["user"].value_counts()
        item_counts = copy_df["item"].value_counts()

        copy_df = copy_df[
            (copy_df["user"].isin(user_counts[user_counts >= min_user].index))
            & (copy_df["item"].isin(item_counts[item_counts >= min_item].index))
        ]

        n_users_after = copy_df["user"].nunique()
        n_items_after = copy_df["item"].nunique()

        if n_users_before == n_users_after and n_items_before == n_items_after:
            break

    return copy_df


def remap_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Map raw user/item ids to 0..n-1. Returns (df, user_map, item_map).
    Reserve item id 0 for padding (real items start at 1) so sequence models can pad with 0."""
    user_map = {raw_id: new_id for new_id, raw_id in enumerate(df["user"].unique())}
    item_map = {raw_id: new_id + 1 for new_id, raw_id in enumerate(df["item"].unique())}

    df["user"] = df["user"].map(user_map)
    df["item"] = df["item"].map(item_map)

    return df, user_map, item_map


def split(df: pd.DataFrame, strategy: str, val_ratio=None, test_ratio=None) -> pd.DataFrame:
    """Adds a `split` column ("train" / "val" / "test") to a df sorted by (user, timestamp).
    leave_one_out: last item -> test, second to last -> val, rest -> train.
    temporal: global time cut by ratios. random: random rows by ratios."""
    if strategy == "leave_one_out":
        df["rank"] = df.groupby("user")["timestamp"].rank(method="first", ascending=False)
        df["split"] = "train"
        df.loc[df["rank"] == 1, "split"] = "test"
        df.loc[df["rank"] == 2, "split"] = "val"
        df.drop(columns=["rank"], inplace=True)
    elif strategy == "temporal":
        assert val_ratio is not None and test_ratio is not None
        total_rows = len(df)
        val_cutoff = int(total_rows * (1 - val_ratio - test_ratio))
        test_cutoff = int(total_rows * (1 - test_ratio))
        df["split"] = "train"
        df.iloc[val_cutoff:test_cutoff, df.columns.get_loc("split")] = "val"
        df.iloc[test_cutoff:, df.columns.get_loc("split")] = "test"
    elif strategy == "random":
        assert val_ratio is not None and test_ratio is not None
        total_rows = len(df)
        val_cutoff = int(total_rows * (1 - val_ratio - test_ratio))
        test_cutoff = int(total_rows * (1 - test_ratio))
        shuffled_indices = df.sample(frac=1, random_state=42).index
        df.loc[shuffled_indices[:val_cutoff], "split"] = "train"
        df.loc[shuffled_indices[val_cutoff:test_cutoff], "split"] = "val"
        df.loc[shuffled_indices[test_cutoff:], "split"] = "test"
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    return df
