"""
Standalone diagnostic script — reads existing data/models and prints a report.
Does NOT modify any existing code or files.

Run from the cold_user_rl/ directory:
    python debug/diagnose.py

Output is printed to console and saved to debug/diagnostic_report.txt
"""
import io
import json
import os
import pickle
import random
import sys
import traceback

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # cold_user_rl/
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

from config import CONFIG

# ── Output capture ────────────────────────────────────────────────────────────
_OUTPUT_LINES = []

def _print(*args, **kwargs):
    line = " ".join(str(a) for a in args)
    _OUTPUT_LINES.append(line)
    print(line, **kwargs)

def _sep(title=""):
    w = 78
    if title:
        side = (w - len(title) - 2) // 2
        _print(f"{'─' * side} {title} {'─' * (w - side - len(title) - 2)}")
    else:
        _print("─" * w)

# ── Helpers ───────────────────────────────────────────────────────────────────
PASS_FAIL = {}  # label → True/False

def _mark(label, passed, detail=""):
    PASS_FAIL[label] = passed
    tag = "PASS" if passed else "FAIL"
    _print(f"  [{tag}] {label}" + (f" — {detail}" if detail else ""))


def _load_processed():
    data_dir = os.path.join(CONFIG["data_path"], "processed")
    train_path = os.path.join(data_dir, "train.parquet")
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    cold_path  = os.path.join(data_dir, "cold_split.pkl")

    import pandas as pd

    train_df = pd.read_parquet(train_path) if os.path.isfile(train_path) else None
    stats    = json.load(open(stats_path)) if os.path.isfile(stats_path) else {}
    cold_split = pickle.load(open(cold_path, "rb")) if os.path.isfile(cold_path) else {}
    return train_df, stats, cold_split, data_dir


def _find_mf_checkpoint():
    ckpt_dir = CONFIG["checkpoint_dir"]
    for name in ("mf_hybrid.pt", "mf_base.pt"):
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            return path, name
    return None, None


def _find_episodes_csv():
    log_dir = CONFIG["log_dir"]
    best = None
    best_mtime = 0
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f == "episodes.csv":
                p = os.path.join(root, f)
                mt = os.path.getmtime(p)
                if mt > best_mtime:
                    best_mtime = mt
                    best = p
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1: Rating Scale
# ═══════════════════════════════════════════════════════════════════════════════
def check_rating_scale():
    _sep("CHECK 1: Rating Scale")
    try:
        import pandas as pd
        train_df, stats, _, _ = _load_processed()

        if train_df is None:
            _print("  ERROR: train.parquet not found.")
            _mark("Rating normalization", False, "file missing")
            return

        _print("\n  ratings_df['rating'].describe():")
        desc = train_df["rating"].describe()
        for k, v in desc.items():
            _print(f"    {k:8s}: {v:.6f}")

        if "rating_normalized" in train_df.columns:
            _print("\n  ratings_df['rating_normalized'].describe():")
            desc2 = train_df["rating_normalized"].describe()
            for k, v in desc2.items():
                _print(f"    {k:8s}: {v:.6f}")
        else:
            _print("\n  WARNING: No normalized rating column found.")
            _print("  The 'rating' column itself is the normalized value (preprocess.py")
            _print("  divides by 5.0 before saving, so [0, 1] is expected).")

        _print("\n  First 10 rows (userId, movieId, rating):")
        _print(f"  {'userId':>8}  {'movieId':>8}  {'rating':>8}")
        for _, row in train_df[["userId", "movieId", "rating"]].head(10).iterrows():
            _print(f"  {int(row.userId):>8}  {int(row.movieId):>8}  {row.rating:>8.4f}")

        uniq = sorted(train_df["rating"].round(3).unique())
        _print(f"\n  Unique rating values ({len(uniq)} total): {uniq[:20]}")

        rmin, rmax = train_df["rating"].min(), train_df["rating"].max()
        _print(f"\n  Min rating: {rmin:.6f}")
        _print(f"  Max rating: {rmax:.6f}")

        normalized = stats.get("normalized", True)
        expected_lo, expected_hi = (0.0, 1.0) if normalized else (0.5, 5.0)
        in_range = (rmin >= expected_lo - 0.01) and (rmax <= expected_hi + 0.01)

        if not in_range:
            _print(f"\n  WARNING: Ratings outside expected range [{expected_lo}, {expected_hi}].")
        else:
            _print(f"\n  Ratings are within expected range [{expected_lo:.1f}, {expected_hi:.1f}]. OK.")

        _mark("Rating normalization", in_range,
              f"min={rmin:.3f} max={rmax:.3f} expected=[{expected_lo},{expected_hi}]")

    except Exception:
        _print("  EXCEPTION in Check 1:")
        _print(traceback.format_exc())
        _mark("Rating normalization", False, "exception")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2: MF Model Predictions
# ═══════════════════════════════════════════════════════════════════════════════
def check_mf_predictions():
    _sep("CHECK 2: MF Model Predictions")
    try:
        from models.matrix_factorization import MatrixFactorization

        train_df, stats, _, _ = _load_processed()
        ckpt_path, ckpt_name = _find_mf_checkpoint()

        if ckpt_path is None:
            _print("  ERROR: No MF checkpoint found in", CONFIG["checkpoint_dir"])
            _mark("MF predictions in valid range", False, "checkpoint missing")
            return

        _print(f"\n  Loading checkpoint: {ckpt_name}")
        n_users = stats["n_users"]
        n_items = stats["n_items"]
        k = CONFIG["mf_latent_features"]
        mf = MatrixFactorization(n_users, n_items, k)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        mf.load_state_dict(ckpt["model_state"])
        mf.eval()

        _print(f"  n_users={n_users:,}  n_items={n_items:,}  k={k}")

        # P and Q range
        P = mf.P.weight.data
        Q = mf.Q.weight.data
        _print(f"\n  P matrix: min={P.min():.4f}  max={P.max():.4f}  "
               f"mean={P.mean():.4f}  std={P.std():.4f}")
        _print(f"  Q matrix: min={Q.min():.4f}  max={Q.max():.4f}  "
               f"mean={Q.mean():.4f}  std={Q.std():.4f}")

        # 5 random warm users, 3 items each
        _print("\n  Sample predictions for 5 warm users (3 items each):")
        _print(f"  {'userId':>8}  {'itemId':>8}  {'actual':>8}  {'predicted':>10}  {'error':>8}")
        _print("  " + "-" * 52)

        import pandas as pd
        rng = random.Random(42)
        user_groups = {uid: grp for uid, grp in train_df.groupby("userId")}
        sampled_users = rng.sample(list(user_groups.keys()), min(5, len(user_groups)))

        all_preds = []
        for uid in sampled_users:
            grp = user_groups[uid]
            rows = grp.sample(min(3, len(grp)), random_state=42)
            for _, row in rows.iterrows():
                u_t = torch.tensor([int(row.userId)], dtype=torch.long)
                i_t = torch.tensor([int(row.movieId)], dtype=torch.long)
                with torch.no_grad():
                    pred = float(mf(u_t, i_t).item())
                actual = float(row.rating)
                err = pred - actual
                all_preds.append(pred)
                _print(f"  {int(row.userId):>8}  {int(row.movieId):>8}  "
                       f"{actual:>8.4f}  {pred:>10.4f}  {err:>8.4f}")

        # 1000 random (user, item) pairs
        _print("\n  Sampling 1000 random (user, item) pairs for range check ...")
        rng_np = np.random.default_rng(42)
        u_sample = torch.tensor(rng_np.integers(0, n_users, 1000), dtype=torch.long)
        i_sample = torch.tensor(rng_np.integers(0, n_items, 1000), dtype=torch.long)
        with torch.no_grad():
            preds_1k = mf(u_sample, i_sample).numpy()

        _print(f"  Predictions min={preds_1k.min():.4f}  max={preds_1k.max():.4f}  "
               f"mean={preds_1k.mean():.4f}")

        normalized = stats.get("normalized", True)
        lo, hi = (0.0, 1.0) if normalized else (0.5, 5.0)
        n_out = int(np.sum((preds_1k < lo - 0.3) | (preds_1k > hi + 0.3)))
        in_range = n_out < 100  # tolerate some outliers

        if n_out > 0:
            _print(f"  WARNING: {n_out}/1000 predictions are far outside [{lo:.1f}, {hi:.1f}].")
        else:
            _print(f"  All predictions within expected range [{lo:.1f}, {hi:.1f}]. OK.")

        _mark("MF predictions in valid range", in_range,
              f"{n_out}/1000 samples outside [{lo},{hi}]")

    except Exception:
        _print("  EXCEPTION in Check 2:")
        _print(traceback.format_exc())
        _mark("MF predictions in valid range", False, "exception")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3: Cold User Vector Optimization
# ═══════════════════════════════════════════════════════════════════════════════
def check_cold_vector():
    _sep("CHECK 3: Cold User Vector Optimization")
    try:
        from models.matrix_factorization import MatrixFactorization, ColdUserFinetuner

        _, stats, cold_split, _ = _load_processed()
        ckpt_path, ckpt_name = _find_mf_checkpoint()

        if ckpt_path is None or not cold_split:
            _print("  ERROR: Missing checkpoint or cold_split.")
            _mark("Cold user vector optimizing correctly", False, "data missing")
            return

        n_users = stats["n_users"]
        n_items = stats["n_items"]
        k = CONFIG["mf_latent_features"]
        mf = MatrixFactorization(n_users, n_items, k)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        mf.load_state_dict(ckpt["model_state"])
        mf.eval()

        finetuner = ColdUserFinetuner(mf, CONFIG)
        normalized = stats.get("normalized", True)

        cold_ids = list(cold_split.keys())
        rng = random.Random(42)
        sampled = rng.sample(cold_ids, min(3, len(cold_ids)))

        all_passed = True
        for uid in sampled:
            _sep()
            _print(f"  Cold user {uid}")
            user_data = cold_split[uid]
            pool    = user_data["interview_pool"]   # list of (item_id, rating)
            test    = user_data["test_set"]          # list of (item_id, rating)

            # Step A
            all_ratings = [r for _, r in pool] + [r for _, r in test]
            _print(f"  Total interactions: {len(all_ratings)}")
            _print(f"  Rating values (pool): {sorted(set(round(r, 4) for _, r in pool))}")
            if normalized:
                _print(f"  (Stored as normalized [0,1]; multiply by 5 for original scale)")
            _print(f"  Test set size: {len(test)}")

            # Step B — simulate 5-item interview
            interview_items_5 = pool[:5]
            interview_item_ids = [iid for iid, _ in interview_items_5]

            p_before = torch.zeros(k)
            _print(f"\n  p_cu before optimization: norm = {torch.norm(p_before):.6f}")
            _print(f"  p_cu values (first 5): {p_before[:5].tolist()}")

            p_init = finetuner.initialize_cold_vector(interview_item_ids)
            p_after, final_loss = finetuner.finetune(p_init.clone(), interview_items_5)

            _print(f"\n  p_cu after  optimization: norm = {torch.norm(p_after):.6f}")
            _print(f"  p_cu values (first 5): {[round(x, 5) for x in p_after[:5].tolist()]}")
            _print(f"  Optimization loss: {final_loss:.6f}")

            norm_diff = float(torch.norm(p_after - p_before).item())
            _print(f"  Norm difference (before→after): {norm_diff:.6f}")
            if norm_diff < 0.01:
                _print("  WARNING: Cold user vector not updating. "
                       "Check optimizer and learning rate.")
                all_passed = False

            # Step C — RMSE on test set
            _print(f"\n  RMSE on {len(test)} test items:")
            _print(f"  {'item_id':>8}  {'actual':>8}  {'predicted':>10}  {'sq_err':>10}")
            _print("  " + "-" * 44)

            sq_errors = []
            for item_id, actual in test:
                i_t = torch.tensor([int(item_id)], dtype=torch.long)
                if int(item_id) >= n_items:
                    continue
                with torch.no_grad():
                    pred = finetuner.compute_rmse.__func__ if False else None
                    Q_row = mf.Q.weight.data[int(item_id)]
                    b_i   = float(mf.b_i.weight.data[int(item_id)])
                    gb    = float(mf.global_bias.data)
                    pred_score = gb + float((p_after * Q_row).sum()) + b_i
                sq_err = (pred_score - float(actual)) ** 2
                sq_errors.append(sq_err)
                _print(f"  {item_id:>8}  {actual:>8.4f}  {pred_score:>10.4f}  {sq_err:>10.6f}")

            if sq_errors:
                rmse = float(np.sqrt(np.mean(sq_errors)))
                reward_implied = 1.0 / (rmse + 1e-8)
                _print(f"\n  RMSE = {rmse:.4f}")
                _print(f"  Reward = 1 / RMSE = {reward_implied:.4f}")

        _mark("Cold user vector optimizing correctly", all_passed)

    except Exception:
        _print("  EXCEPTION in Check 3:")
        _print(traceback.format_exc())
        _mark("Cold user vector optimizing correctly", False, "exception")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 4: Reward–RMSE Consistency
# ═══════════════════════════════════════════════════════════════════════════════
def check_reward_rmse():
    _sep("CHECK 4: Reward-RMSE Consistency")
    try:
        import pandas as pd
        csv_path = _find_episodes_csv()
        if csv_path is None:
            _print("  No episodes.csv found in", CONFIG["log_dir"])
            _mark("Reward-RMSE consistency", False, "file missing")
            return

        _print(f"  Loading: {csv_path}")
        df = pd.read_csv(csv_path)
        _print(f"  {len(df)} episodes loaded. Columns: {list(df.columns)}")

        if "reward" not in df.columns or "rmse" not in df.columns:
            _print("  ERROR: Missing 'reward' or 'rmse' columns.")
            _mark("Reward-RMSE consistency", False, "columns missing")
            return

        # Episode reward = SUM of per-step 1/RMSE rewards.
        # Best approximation: per-step avg reward = episode_reward / interview_size
        # → implied_rmse ≈ interview_size / episode_reward
        # Without per-step data we use the simpler 1/reward as a rough check.
        interview_sizes = CONFIG.get("interview_sizes", [10])
        default_size = interview_sizes[0] if interview_sizes else 10

        if "interview_size" in df.columns:
            df["_isize"] = df["interview_size"]
        else:
            df["_isize"] = default_size

        df["implied_rmse"] = df["_isize"] / (df["reward"] + 1e-8)

        # "match" = implied_rmse is within 50% of logged rmse (loose, because
        # the episode reward is a sum of per-step rewards at different RMSE values,
        # not just 1 / final_RMSE)
        df["match"] = ((df["implied_rmse"] - df["rmse"]).abs() / (df["rmse"] + 1e-8) < 0.50)

        _print(f"\n  Note: episode reward = SUM of per-step 1/RMSE values.")
        _print(f"  implied_rmse = interview_size / episode_reward (rough approximation).")
        _print(f"\n  First 20 rows:")
        _print(f"  {'episode':>8}  {'reward':>8}  {'rmse':>8}  {'implied':>10}  match")
        _print("  " + "-" * 50)

        for _, row in df.head(20).iterrows():
            m = "yes" if row["match"] else "no "
            _print(f"  {int(row['episode']):>8}  {row['reward']:>8.3f}  "
                   f"{row['rmse']:>8.4f}  {row['implied_rmse']:>10.4f}  {m}")

        match_rate = df["match"].mean()
        _print(f"\n  Match rate (within 50%): {match_rate:.1%}")

        if match_rate < 0.50:
            _print("\n  WARNING: Reward and RMSE are inconsistent.")
            _print("  This means the RL agent is not being rewarded based on what we measure.")
            passed = False
        else:
            _print("\n  Reward-RMSE relationship looks consistent.")
            passed = True

        _mark("Reward-RMSE consistency", passed, f"match rate {match_rate:.1%}")

    except Exception:
        _print("  EXCEPTION in Check 4:")
        _print(traceback.format_exc())
        _mark("Reward-RMSE consistency", False, "exception")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 5: Learning Trend
# ═══════════════════════════════════════════════════════════════════════════════
def check_learning_trend():
    _sep("CHECK 5: Learning Trend")
    try:
        import pandas as pd
        csv_path = _find_episodes_csv()
        if csv_path is None:
            _print("  No episodes.csv found.")
            _mark("RL learning trend detected", False, "file missing")
            return

        df = pd.read_csv(csv_path)
        if "rmse" not in df.columns:
            _print("  ERROR: 'rmse' column missing.")
            _mark("RL learning trend detected", False, "column missing")
            return

        interview_sizes = CONFIG.get("interview_sizes", [10])
        if "interview_size" not in df.columns:
            df["interview_size"] = interview_sizes[0] if interview_sizes else 10

        any_improving = False
        for isize in df["interview_size"].unique():
            sub = df[df["interview_size"] == isize].reset_index(drop=True)
            if len(sub) < 4:
                continue
            n = len(sub)
            q_size = n // 4
            q1 = sub.iloc[:q_size]["rmse"].mean()
            q2 = sub.iloc[q_size:2*q_size]["rmse"].mean()
            q3 = sub.iloc[2*q_size:3*q_size]["rmse"].mean()
            q4 = sub.iloc[3*q_size:]["rmse"].mean()

            improvement = (q1 - q4) / (q1 + 1e-8)  # positive = improving
            if improvement > 0.05:
                trend = "improving"
                any_improving = True
            elif improvement < -0.05:
                trend = "degrading"
            else:
                trend = "stable"

            _print(f"\n  interview_size={isize}  ({len(sub)} episodes)")
            _print(f"    Q1 mean RMSE: {q1:.4f}")
            _print(f"    Q2 mean RMSE: {q2:.4f}")
            _print(f"    Q3 mean RMSE: {q3:.4f}")
            _print(f"    Q4 mean RMSE: {q4:.4f}")
            _print(f"    Trend: {trend}  (Q1→Q4 change: {improvement:+.1%})")

        if not any_improving:
            _print("\n  WARNING: RL agent shows no learning trend.")
            _print("  The agent may not be training correctly.")

        _mark("RL learning trend detected", any_improving)

    except Exception:
        _print("  EXCEPTION in Check 5:")
        _print(traceback.format_exc())
        _mark("RL learning trend detected", False, "exception")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 6: State Vector
# ═══════════════════════════════════════════════════════════════════════════════
def check_state_vector():
    _sep("CHECK 6: State Vector")
    try:
        from models.matrix_factorization import MatrixFactorization, ColdUserFinetuner
        from training.train_rl import build_env
        from data.feedback_constructor import build_feedback_bundle, load_tags

        _, stats, cold_split, data_dir = _load_processed()
        ckpt_path, ckpt_name = _find_mf_checkpoint()

        if ckpt_path is None or not cold_split:
            _print("  ERROR: Missing checkpoint or cold_split.")
            _mark("State vector valid (no NaN/Inf)", False, "data missing")
            return

        import pandas as pd
        processed_dir = os.path.join(CONFIG["data_path"], "processed")
        train_df = pd.read_parquet(os.path.join(processed_dir, "train.parquet"))
        user_map = pickle.load(open(os.path.join(processed_dir, "user_map.pkl"), "rb"))
        item_map = pickle.load(open(os.path.join(processed_dir, "item_map.pkl"), "rb"))

        n_users = stats["n_users"]
        n_items = stats["n_items"]
        k = CONFIG["mf_latent_features"]
        mf = MatrixFactorization(n_users, n_items, k)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        mf.load_state_dict(ckpt["model_state"])
        mf.eval()

        tags_df = load_tags(CONFIG["data_path"], user_map, item_map)
        feedback_bundle = build_feedback_bundle(train_df, tags_df, n_users, n_items, CONFIG)

        item_counts = np.bincount(train_df["movieId"].values.astype(np.int64), minlength=n_items)
        pool_size = CONFIG.get("action_pool_size", 200)
        action_pool = np.argsort(-item_counts)[:pool_size].astype(np.int64)

        pop_dict = {int(i): int(item_counts[i]) for i in range(n_items)}
        item_embeddings = mf.get_item_matrix().numpy()
        from active_learning.strategies import build_item_metadata
        item_metadata = build_item_metadata(pop_dict, item_embeddings)

        finetuner = ColdUserFinetuner(mf, CONFIG)
        env = build_env(mf, finetuner, feedback_bundle, cold_split,
                        action_pool, item_metadata, CONFIG)

        _print(f"  Environment type: {type(env).__name__}")
        _print(f"  state_dim = {env.state_dim}")

        state = env.reset()
        _print(f"\n  After reset():")
        _print(f"  state.shape = {state.shape}")
        _print(f"  min={state.min():.4f}  max={state.max():.4f}  mean={state.mean():.4f}")

        n_nan = int(np.isnan(state).sum())
        n_inf = int(np.isinf(state).sum())
        _print(f"  NaN count: {n_nan}   Inf count: {n_inf}")

        in_zscore = bool((state.min() >= -10) and (state.max() <= 10))
        _print(f"  Values in [-10, 10] range: {in_zscore}")
        if state.max() <= 1.0 and state.min() >= 0.0:
            _print("  State appears to be a binary/normalized vector [0, 1].")
        elif state.min() >= -3 and state.max() <= 3:
            _print("  State appears z-score normalized (values roughly in [-3, 3]).")

        # One env step
        valid = env.get_valid_actions()
        _print(f"\n  Valid actions at step 0: {len(valid)} available")
        if len(valid) > 0:
            action = int(valid[0])
            next_state, reward, done, info = env.step(action)
            _print(f"\n  After one step (action={action}):")
            _print(f"  next_state.shape = {next_state.shape}")
            _print(f"  reward = {reward:.4f}  done = {done}")
            _print(f"  info RMSE = {info.get('rmse', 'N/A')}")
            n_nan2 = int(np.isnan(next_state).sum())
            n_inf2 = int(np.isinf(next_state).sum())
            _print(f"  NaN count: {n_nan2}   Inf count: {n_inf2}")

        passed = (n_nan == 0) and (n_inf == 0)
        _mark("State vector valid (no NaN/Inf)", passed,
              f"NaN={n_nan} Inf={n_inf}")

    except Exception:
        _print("  EXCEPTION in Check 6:")
        _print(traceback.format_exc())
        _mark("State vector valid (no NaN/Inf)", False, "exception")


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def final_report():
    _sep("FINAL REPORT")
    _print("\n=== DIAGNOSTIC SUMMARY ===\n")

    labels = [
        "Rating normalization",
        "MF predictions in valid range",
        "Cold user vector optimizing correctly",
        "Reward-RMSE consistency",
        "RL learning trend detected",
        "State vector valid (no NaN/Inf)",
    ]

    fails = []
    for label in labels:
        result = PASS_FAIL.get(label, None)
        if result is None:
            tag = "SKIP"
        elif result:
            tag = "PASS"
        else:
            tag = "FAIL"
            fails.append(label)
        _print(f"  [{tag}] {label}")

    _print()
    if fails:
        _print(f"Issues found ({len(fails)}):")
        for f in fails:
            _print(f"  • {f}")
    else:
        _print("  No issues found — all checks passed.")
    _print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    _print("=" * 78)
    _print("  COLD-USER RL DIAGNOSTIC REPORT")
    _print(f"  data_path     : {CONFIG['data_path']}")
    _print(f"  checkpoint_dir: {CONFIG['checkpoint_dir']}")
    _print(f"  log_dir       : {CONFIG['log_dir']}")
    _print("=" * 78)

    check_rating_scale()
    check_mf_predictions()
    check_cold_vector()
    check_reward_rmse()
    check_learning_trend()
    check_state_vector()
    final_report()

    # Save report
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "diagnostic_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_OUTPUT_LINES))
    _print(f"Report saved to: {report_path}")
