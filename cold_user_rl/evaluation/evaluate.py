import json
import os

import numpy as np
import torch

from evaluation.metrics import compute_all_metrics, run_pairwise_ttests


def evaluate_cold_users(agent, env, mf_model, finetuner, cold_split,
                        action_pool, item_metadata, config,
                        running_mean=None, running_var=None,
                        n_users=None):
    """Evaluate the trained RL agent on cold users with epsilon=0 (greedy).

    Runs each interview size in config['interview_sizes'] and aggregates metrics.

    Args:
        agent: DQNAgent / DRQNAgent / HierarchicalRLAgent
        env: ColdUserEnv or ColdUserEnvExtended
        mf_model: MatrixFactorization
        finetuner: ColdUserFinetuner
        cold_split: dict {user_id: {interview_pool, test_set}}
        action_pool: numpy array of action item IDs
        item_metadata: dict
        config: CONFIG dict
        running_mean, running_var: state normalization statistics (None = no normalization)
        n_users: how many cold users to evaluate (None = all)

    Returns:
        dict with aggregated metrics per interview size
    """
    use_recurrent = config.get("use_recurrent_dqn", False)
    use_hier = config.get("use_hierarchical_rl", False)
    k_values = config.get("eval_k_values", [5, 10, 20])
    interview_sizes = config.get("interview_sizes", [10])

    cold_user_ids = list(cold_split.keys())
    if n_users is not None:
        n_users = min(n_users, len(cold_user_ids))
        cold_user_ids = np.random.choice(cold_user_ids, size=n_users, replace=False).tolist()

    all_item_ids = np.arange(mf_model.Q.weight.shape[0], dtype=np.int64)

    results_by_size = {}

    for interview_size in interview_sizes:
        env.set_interview_size(interview_size)
        size_metrics = []

        for uid in cold_user_ids:
            if use_recurrent:
                agent.reset_hidden()

            state = env.reset(user_id=uid)
            state_norm = _norm(state, running_mean, running_var)

            done = False
            info = {}
            while not done:
                valid_actions = env.get_valid_actions()
                if not len(valid_actions):
                    break
                if use_hier:
                    _, item_pool_idx, _ = agent.select_action(
                        state_norm, valid_actions, 0.0, mf_model, env.p_cold
                    )
                    action = (0, item_pool_idx)
                else:
                    action = agent.select_action(state_norm, valid_actions, 0.0)
                next_state, _, done, info = env.step(action)
                state_norm = _norm(next_state, running_mean, running_var)

            # Compute full metrics after interview
            user_metrics = compute_all_metrics(
                mf_model, env.p_cold,
                cold_split[uid]["test_set"],
                all_item_ids,
                k_values=k_values,
            )
            size_metrics.append(user_metrics)

        # Aggregate across users
        agg = {}
        if size_metrics:
            all_keys = size_metrics[0].keys()
            for key in all_keys:
                vals = [m[key] for m in size_metrics if not np.isnan(m.get(key, float("nan")))]
                agg[f"mean_{key}"] = float(np.mean(vals)) if vals else float("nan")
                agg[f"std_{key}"] = float(np.std(vals)) if vals else float("nan")

        results_by_size[interview_size] = agg

    return results_by_size


def compare_no_interview_baseline(mf_model, cold_split, config, running_mean=None, running_var=None):
    """Evaluate the no-interview baseline: cold user initialized to mean warm-user vector.

    This is the lower-bound comparison — no RL, no interview.

    Returns:
        dict with same structure as evaluate_cold_users
    """
    k_values = config.get("eval_k_values", [5, 10, 20])
    all_item_ids = np.arange(mf_model.Q.weight.shape[0], dtype=np.int64)

    mean_user_vec = mf_model.P.weight.data.mean(dim=0)  # (k,)

    metrics_list = []
    for uid, user_data in cold_split.items():
        user_metrics = compute_all_metrics(
            mf_model, mean_user_vec,
            user_data["test_set"],
            all_item_ids,
            k_values=k_values,
        )
        metrics_list.append(user_metrics)

    agg = {}
    if metrics_list:
        for key in metrics_list[0].keys():
            vals = [m[key] for m in metrics_list if not np.isnan(m.get(key, float("nan")))]
            agg[f"mean_{key}"] = float(np.mean(vals)) if vals else float("nan")
            agg[f"std_{key}"] = float(np.std(vals)) if vals else float("nan")

    return {"no_interview_baseline": agg}


def run_statistical_tests(results_by_method, metric="rmse"):
    """Run pairwise t-tests across all method results.

    Args:
        results_by_method: dict {method_name: {interview_size: {mean_rmse, ...}}}
                           OR {method_name: list of per-user metric values}
        metric: metric name for comparison

    Returns:
        pd.DataFrame with pairwise test results
    """
    import pandas as pd

    # Expect {method_name: [per-user metric values]}
    comparisons = run_pairwise_ttests(results_by_method, metric)

    rows = []
    for (method_a, method_b), test_result in comparisons.items():
        rows.append({
            "method_a": method_a,
            "method_b": method_b,
            **test_result,
        })
    return pd.DataFrame(rows)


def generate_full_report(results_dict, output_path):
    """Write evaluation results to JSON and print a summary table.

    Args:
        results_dict: {experiment_name: {interview_size: {metric: value}}}
        output_path: path to save the JSON report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("EVALUATION REPORT")
    print(f"{'='*70}")
    for exp_name, size_results in results_dict.items():
        print(f"\n{exp_name}:")
        if isinstance(size_results, dict):
            for size, metrics in size_results.items():
                if isinstance(metrics, dict):
                    mean_rmse = metrics.get("mean_rmse", "N/A")
                    ndcg = metrics.get("mean_ndcg@10", "N/A")
                    print(f"  Interview size {size}: RMSE={mean_rmse:.4f}, NDCG@10={ndcg:.4f}")
    print(f"{'='*70}")
    print(f"Full report saved to: {output_path}")


def _norm(state, mean, var):
    if mean is None or var is None:
        return state
    return (state - mean) / (np.sqrt(var) + 1e-8)
