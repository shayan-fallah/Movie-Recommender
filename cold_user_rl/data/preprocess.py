import json
import os
import pickle
import random

import numpy as np
import pandas as pd


def _save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_kcore_subset(ratings_df, tags_df, config):
    """Return a filtered subset via year filter + iterative k-core co-filtering + hard caps.

    Steps:
      1. (Optional) Drop movies released before config["min_year"] using movieId→year
         extracted from the ml-32m movies.csv title field ("Movie Title (YYYY)").
      2. Iteratively remove users with < min_user_ratings and items with < min_item_ratings
         until the graph stops shrinking (up to kcore_iterations passes).
      3. Enforce hard caps: sample max_users users and max_items items by rating count,
         then keep only rows in the intersection.
      4. Re-filter tags_df to the surviving (userId, movieId) pairs.

    Returns (ratings_df_filtered, tags_df_filtered).
    tags_df_filtered is None if tags_df was None.
    """
    if not config.get("use_subset", False):
        return ratings_df, tags_df

    min_u = config.get("min_user_ratings", 100)
    min_i = config.get("min_item_ratings", 200)
    max_iters = config.get("kcore_iterations", 10)
    min_year = config.get("min_year", None)
    max_users = config.get("max_users", None)
    max_items = config.get("max_items", None)
    rng_seed = config.get("subset_random_seed", 42)
    data_dir = config.get("data_path", ".")

    df = ratings_df.copy()

    # ── Step 1: year filter ────────────────────────────────────────────────────
    if min_year is not None:
        movies_path = os.path.join(data_dir, "movies.csv")
        if os.path.isfile(movies_path):
            movies = pd.read_csv(movies_path, dtype={"movieId": int})
            # Extract year from title like "Toy Story (1995)"
            movies["year"] = movies["title"].str.extract(r'\((\d{4})\)$').astype(float)
            keep_ids = movies.loc[movies["year"] >= min_year, "movieId"]
            before = df["movieId"].nunique()
            df = df[df["movieId"].isin(keep_ids)].reset_index(drop=True)
            print(f"  Year filter (>={min_year}): {before:,} → {df['movieId'].nunique():,} items")
        else:
            print(f"  Warning: movies.csv not found at {movies_path}; skipping year filter")

    # ── Step 2: iterative k-core ───────────────────────────────────────────────
    for iteration in range(max_iters):
        prev_len = len(df)
        user_counts = df.groupby("userId").size()
        df = df[df["userId"].isin(user_counts[user_counts >= min_u].index)]
        item_counts = df.groupby("movieId").size()
        df = df[df["movieId"].isin(item_counts[item_counts >= min_i].index)]
        df = df.reset_index(drop=True)
        if len(df) == prev_len:
            print(f"  K-core converged after {iteration + 1} iteration(s): "
                  f"{df['userId'].nunique():,} users, {df['movieId'].nunique():,} items, "
                  f"{len(df):,} ratings")
            break
    else:
        print(f"  K-core reached max iterations ({max_iters}): "
              f"{df['userId'].nunique():,} users, {df['movieId'].nunique():,} items")

    # ── Step 3: hard caps ──────────────────────────────────────────────────────
    rng = np.random.default_rng(rng_seed)

    if max_users is not None and df["userId"].nunique() > max_users:
        user_rc = df.groupby("userId").size().sort_values(ascending=False)
        keep_users = user_rc.index[:max_users]
        df = df[df["userId"].isin(keep_users)].reset_index(drop=True)
        print(f"  Hard cap: kept top-{max_users:,} users by rating count")

    if max_items is not None and df["movieId"].nunique() > max_items:
        item_rc = df.groupby("movieId").size().sort_values(ascending=False)
        keep_items = item_rc.index[:max_items]
        df = df[df["movieId"].isin(keep_items)].reset_index(drop=True)
        print(f"  Hard cap: kept top-{max_items:,} items by rating count")

    # ── Step 4: filter tags ────────────────────────────────────────────────────
    tags_out = None
    if tags_df is not None and len(tags_df) > 0:
        surviving_users = set(df["userId"].unique())
        surviving_items = set(df["movieId"].unique())
        tags_out = tags_df[
            tags_df["userId"].isin(surviving_users) &
            tags_df["movieId"].isin(surviving_items)
        ].reset_index(drop=True)

    return df, tags_out


def print_density_report(ratings_df, config):
    """Print dataset density statistics after filtering."""
    n_users = ratings_df["userId"].nunique()
    n_items = ratings_df["movieId"].nunique()
    n_ratings = len(ratings_df)
    density = n_ratings / max(n_users * n_items, 1)
    avg_per_user = n_ratings / max(n_users, 1)
    avg_per_item = n_ratings / max(n_items, 1)

    print("\n── Dataset density report ──────────────────────────────────────────")
    print(f"  Users:             {n_users:>10,}")
    print(f"  Items:             {n_items:>10,}")
    print(f"  Ratings:           {n_ratings:>10,}")
    print(f"  Density:           {density:>10.4%}")
    print(f"  Avg ratings/user:  {avg_per_user:>10.1f}")
    print(f"  Avg ratings/item:  {avg_per_item:>10.1f}")

    min_cold = config.get("min_cold_user_interactions", 15)
    if avg_per_user < min_cold * 2:
        print(f"  WARNING: avg ratings/user ({avg_per_user:.1f}) is close to "
              f"min_cold_user_interactions ({min_cold}). "
              f"Consider lowering min_user_ratings or min_cold_user_interactions.")
    if density < 0.001:
        print(f"  WARNING: density ({density:.4%}) is very low. "
              f"MF may struggle to learn useful embeddings.")
    print("────────────────────────────────────────────────────────────────────\n")


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

    # Load tags early so build_kcore_subset can filter them alongside ratings
    tags_path = os.path.join(data_dir, "tags.csv")
    if os.path.isfile(tags_path):
        tags_raw = pd.read_csv(tags_path, dtype={"userId": int, "movieId": int})
        tags_raw = tags_raw[["userId", "movieId"]].drop_duplicates()
    else:
        tags_raw = None

    # Subsample via k-core filtering (no-op when use_subset=False)
    df, tags_raw = build_kcore_subset(df, tags_raw, config)
    print_density_report(df, config)

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
