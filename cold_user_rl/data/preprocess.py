import json
import os
import pickle
import random

import numpy as np
import pandas as pd


def load_raw_ratings(data_dir):
    path = os.path.join(data_dir, "ratings.csv")
    df = pd.read_csv(path, dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int})
    df = df.drop_duplicates(subset=["userId", "movieId"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def filter_users(df, min_interactions):
    counts = df.groupby("userId").size()
    valid = counts[counts >= min_interactions].index
    return df[df["userId"].isin(valid)].reset_index(drop=True)


def remap_ids(df):
    """Remap userId and movieId to contiguous 0-indexed integers."""
    users = sorted(df["userId"].unique())
    items = sorted(df["movieId"].unique())
    user_map = {uid: i for i, uid in enumerate(users)}
    item_map = {mid: i for i, mid in enumerate(items)}
    df = df.copy()
    df["userId"] = df["userId"].map(user_map)
    df["movieId"] = df["movieId"].map(item_map)
    return df, user_map, item_map


def split_warm_cold(df, cold_user_fraction, min_cold_user_interactions, seed):
    """Randomly assign cold_user_fraction of users as cold users.

    Cold users must have >= min_cold_user_interactions ratings so there is
    enough data for both interview pool and test set.
    """
    rng = random.Random(seed)
    user_counts = df.groupby("userId").size()
    eligible = user_counts[user_counts >= min_cold_user_interactions].index.tolist()
    n_cold = max(1, int(len(eligible) * cold_user_fraction))
    cold_ids = set(rng.sample(eligible, n_cold))
    warm_df = df[~df["userId"].isin(cold_ids)].reset_index(drop=True)
    cold_df = df[df["userId"].isin(cold_ids)].reset_index(drop=True)
    return warm_df, cold_df


def split_cold_user_data(cold_user_df, seed):
    """For each cold user split interactions into interview_pool and test_set.

    Guarantees test_set has at least 5 items.
    Returns dict: {user_id: {"interview_pool": [...], "test_set": [...]}}
    """
    rng = random.Random(seed)
    result = {}
    for uid, grp in cold_user_df.groupby("userId"):
        records = list(grp[["movieId", "rating", "timestamp"]].itertuples(index=False, name=None))
        rng.shuffle(records)
        n_test = max(5, int(len(records) * 0.2))
        test_set = records[:n_test]
        interview_pool = records[n_test:]
        result[uid] = {
            "interview_pool": [(r[0], r[1]) for r in interview_pool],
            "test_set": [(r[0], r[1]) for r in test_set],
        }
    return result


def temporal_split(warm_df, val_frac, test_frac):
    """Per-user temporal split: oldest -> train, newest val_frac -> val, test_frac -> test."""
    train_rows, val_rows, test_rows = [], [], []
    for _, grp in warm_df.groupby("userId"):
        grp_sorted = grp.sort_values("timestamp")
        n = len(grp_sorted)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test_rows.append(grp_sorted.iloc[-n_test:])
        val_rows.append(grp_sorted.iloc[-(n_test + n_val):-n_test])
        train_rows.append(grp_sorted.iloc[:-(n_test + n_val)])

    train = pd.concat(train_rows).reset_index(drop=True)
    val = pd.concat(val_rows).reset_index(drop=True)
    test = pd.concat(test_rows).reset_index(drop=True)
    return train, val, test


def save_processed(output_dir, train_df, val_df, test_df,
                   warm_user_ids, cold_split_dict,
                   user_map, item_map, n_users, n_items,
                   normalize=True):
    """Save processed data using parquet (memory-efficient for large datasets).

    Dense numpy matrices are NOT used — a 200K×84K float matrix is ~67 GB.
    DataFrames are stored as parquet files; cold_split as pickle.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalize ratings before saving so everything downstream uses [0,1] scale
    def _maybe_norm(df):
        if normalize:
            df = df.copy()
            df["rating"] = df["rating"] / 5.0
        return df

    _maybe_norm(train_df).to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    _maybe_norm(val_df).to_parquet(os.path.join(output_dir, "val.parquet"), index=False)

    np.save(os.path.join(output_dir, "warm_user_ids.npy"), np.array(warm_user_ids))

    with open(os.path.join(output_dir, "user_map.pkl"), "wb") as f:
        pickle.dump(user_map, f)
    with open(os.path.join(output_dir, "item_map.pkl"), "wb") as f:
        pickle.dump(item_map, f)
    with open(os.path.join(output_dir, "cold_split.pkl"), "wb") as f:
        pickle.dump(cold_split_dict, f)

    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "n_warm_users": len(warm_user_ids),
        "n_cold_users": len(cold_split_dict),
        "n_train_ratings": len(train_df),
        "n_val_ratings": len(val_df),
        "normalized": normalize,
    }
    # dataset_stats.json is the sentinel file checked to detect complete preprocessing
    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved processed data to {output_dir}")
    print(json.dumps(stats, indent=2))


def load_processed(data_dir):
    result = {}
    result["train_df"] = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    result["val_df"] = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
    result["warm_user_ids"] = np.load(os.path.join(data_dir, "warm_user_ids.npy"), allow_pickle=False)

    with open(os.path.join(data_dir, "user_map.pkl"), "rb") as f:
        result["user_map"] = pickle.load(f)
    with open(os.path.join(data_dir, "item_map.pkl"), "rb") as f:
        result["item_map"] = pickle.load(f)
    with open(os.path.join(data_dir, "cold_split.pkl"), "rb") as f:
        result["cold_split"] = pickle.load(f)
    with open(os.path.join(data_dir, "dataset_stats.json"), "r") as f:
        result["stats"] = json.load(f)

    return result


def is_processed(data_dir):
    """Return True only when preprocessing completed successfully.

    Checks for dataset_stats.json (written last), not just the directory.
    A partial/interrupted run leaves the directory but may not write this file.
    """
    return os.path.isfile(os.path.join(data_dir, "dataset_stats.json"))


def run_pipeline(config):
    """Full preprocessing pipeline driven by CONFIG dict."""
    data_dir = config["data_path"]
    seed = config.get("random_seed", 42)
    normalize = config.get("normalize_ratings", True)

    print("Loading raw ratings ...")
    df = load_raw_ratings(data_dir)
    print(f"  Raw: {len(df):,} ratings, {df['userId'].nunique():,} users, {df['movieId'].nunique():,} items")

    df = filter_users(df, config["min_user_interactions"])
    print(f"  After filter (>={config['min_user_interactions']} ratings): {df['userId'].nunique():,} users")

    df, user_map, item_map = remap_ids(df)
    n_users = df["userId"].nunique()
    n_items = df["movieId"].nunique()

    warm_df, cold_df = split_warm_cold(
        df,
        config["cold_user_fraction"],
        config["min_cold_user_interactions"],
        seed,
    )
    print(f"  Warm users: {warm_df['userId'].nunique():,} | Cold users: {cold_df['userId'].nunique():,}")

    cold_split_dict = split_cold_user_data(cold_df, seed)
    warm_user_ids = warm_df["userId"].unique().tolist()

    train_df, val_df, test_df = temporal_split(
        warm_df,
        config["train_val_test_split"][1],
        config["train_val_test_split"][2],
    )

    out_dir = os.path.join(data_dir, "processed")
    save_processed(
        out_dir, train_df, val_df, test_df,
        warm_user_ids, cold_split_dict,
        user_map, item_map, n_users, n_items, normalize,
    )
    return out_dir


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import CONFIG
    run_pipeline(CONFIG)
