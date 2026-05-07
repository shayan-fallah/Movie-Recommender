"""Shared setup logic for all experiment scripts."""
import os
import sys

import numpy as np

# Ensure cold_user_rl package is on the path when running from anywhere
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)


def setup_experiment(config, experiment_name):
    """Full experiment setup: seeds, data, MF model, RL training, evaluation.

    Returns a dict of all objects needed by the caller.
    """
    from training.train_mf import set_all_seeds, load_or_train_mf
    from training.train_rl import train_rl
    from evaluation.evaluate import (
        evaluate_cold_users, compare_no_interview_baseline, generate_full_report
    )
    from data.preprocess import load_processed
    from data.feedback_constructor import build_feedback_bundle, load_tags
    from models.matrix_factorization import MatrixFactorization, ColdUserFinetuner
    from utils.logger import ExperimentLogger
    from active_learning.strategies import build_item_metadata

    set_all_seeds(config.get("random_seed", 42))

    logger = ExperimentLogger(config, experiment_name)
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Config flags: hybrid={config.get('use_hybrid_feedback')}, "
                f"al={config.get('use_personalized_al')}, "
                f"recurrent={config.get('use_recurrent_dqn')}, "
                f"hierarchical={config.get('use_hierarchical_rl')}")

    # ── Data ─────────────────────────────────────────────────────────────────
    processed_dir = os.path.join(config["data_path"], "processed")
    if not os.path.isdir(processed_dir):
        logger.info("Processed data not found — running preprocessing pipeline ...")
        from data.preprocess import run_pipeline
        run_pipeline(config)

    data = load_processed(processed_dir)
    stats = data["stats"]
    n_users = stats["n_users"]
    n_items = stats["n_items"]
    logger.info(f"Dataset: {n_users:,} users, {n_items:,} items")

    tags_df = load_tags(config["data_path"], data["user_map"], data["item_map"])

    # Rebuild training dataframe for feedback bundle
    import pandas as pd
    train_mat = data["train_ratings"]
    u_ids, i_ids = np.where(~np.isnan(train_mat))
    ratings = train_mat[u_ids, i_ids]
    train_df = pd.DataFrame({"userId": u_ids, "movieId": i_ids, "rating": ratings})

    feedback_bundle = build_feedback_bundle(train_df, tags_df, n_users, n_items, config)

    # ── MF Model ─────────────────────────────────────────────────────────────
    k = config["mf_latent_features"]
    mf_model = MatrixFactorization(n_users, n_items, k)

    ckpt_dir = config.get("checkpoint_dir", "./checkpoints")
    mf_ckpt = os.path.join(ckpt_dir, f"mf_{'hybrid' if config.get('use_hybrid_feedback') else 'base'}.pt")
    mf_model = load_or_train_mf(mf_ckpt, mf_model, feedback_bundle, config, logger)
    mf_model.eval()

    # ── Action Pool ───────────────────────────────────────────────────────────
    # Top-N most popular items (by rating count in training data)
    pool_size = config.get("action_pool_size", 200)
    item_counts = np.bincount(i_ids.astype(np.int64), minlength=n_items)
    action_pool = np.argsort(-item_counts)[:pool_size].astype(np.int64)

    # ── Item metadata ─────────────────────────────────────────────────────────
    pop_dict = {int(i): int(item_counts[i]) for i in range(n_items)}
    item_embeddings = mf_model.get_item_matrix().numpy()  # (n_items, k)
    item_metadata = build_item_metadata(pop_dict, item_embeddings)

    # ── Cold split ────────────────────────────────────────────────────────────
    cold_split = data["cold_split"]
    cold_ids = list(cold_split.keys())

    # 80/20 train/eval split among cold users
    rng = np.random.default_rng(config.get("random_seed", 42))
    rng.shuffle(cold_ids)
    n_train_cold = max(1, int(len(cold_ids) * 0.8))
    train_cold_split = {uid: cold_split[uid] for uid in cold_ids[:n_train_cold]}
    eval_cold_split = {uid: cold_split[uid] for uid in cold_ids[n_train_cold:]}

    logger.info(f"Cold users — train: {len(train_cold_split)}, eval: {len(eval_cold_split)}")

    # ── RL Training ───────────────────────────────────────────────────────────
    agent = train_rl(
        mf_model, feedback_bundle, train_cold_split, action_pool,
        item_metadata, config, logger, eval_cold_split=eval_cold_split
    )

    # ── Final Evaluation ──────────────────────────────────────────────────────
    finetuner = ColdUserFinetuner(mf_model, config)
    # Rebuild env for evaluation
    from training.train_rl import build_env
    eval_env = build_env(
        mf_model, finetuner, feedback_bundle, eval_cold_split,
        action_pool, item_metadata, config
    )
    item_metadata["action_pool"] = action_pool

    n_eval = config.get("num_eval_episodes", 200)
    results = evaluate_cold_users(
        agent, eval_env, mf_model, finetuner,
        eval_cold_split, action_pool, item_metadata, config,
        n_users=n_eval
    )

    baseline = compare_no_interview_baseline(mf_model, eval_cold_split, config)

    report = {experiment_name: results, "baseline": baseline}
    report_path = os.path.join(config.get("log_dir", "./logs"), experiment_name, "results.json")
    generate_full_report(report, report_path)

    logger.close()
    return {"agent": agent, "results": results, "baseline": baseline}
