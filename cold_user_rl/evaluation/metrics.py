import math

import numpy as np
import torch
from scipy import stats


def rmse(predictions, targets):
    """RMSE ignoring NaN targets.

    Args:
        predictions: array-like
        targets: array-like (NaN entries are excluded)

    Returns:
        float RMSE
    """
    preds = np.asarray(predictions, dtype=np.float64)
    tgts = np.asarray(targets, dtype=np.float64)
    mask = ~np.isnan(tgts)
    if mask.sum() == 0:
        return float("nan")
    diff = preds[mask] - tgts[mask]
    return float(np.sqrt((diff ** 2).mean()))


def precision_at_k(recommended, relevant, k):
    """Precision@K.

    Args:
        recommended: ordered list of item IDs
        relevant: set/list of ground-truth item IDs

    Returns:
        float in [0, 1]
    """
    relevant_set = set(relevant)
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(recommended, relevant, k):
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(recommended, relevant, k):
    """NDCG@K."""
    relevant_set = set(relevant)
    top_k = recommended[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(top_k) if item in relevant_set)
    ideal_n = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended, relevant, k):
    relevant_set = set(relevant)
    return 1.0 if any(item in relevant_set for item in recommended[:k]) else 0.0


def compute_all_metrics(mf_model, p_cold, test_ratings, all_item_ids,
                        k_values=(5, 10, 20)):
    """Compute RMSE + ranking metrics for a single cold user.

    Args:
        mf_model: MatrixFactorization
        p_cold: (k_latent,) tensor — final cold user vector
        test_ratings: list of (item_id, rating) tuples (held-out items)
        all_item_ids: numpy array of all item indices to score for ranking
        k_values: tuple of K values for ranking metrics

    Returns:
        dict with keys: rmse, precision@K, recall@K, ndcg@K, hit_rate@K
    """
    if not test_ratings:
        return {}

    test_ids = [x[0] for x in test_ratings]
    test_labels = np.array([x[1] for x in test_ratings], dtype=np.float64)

    # RMSE on test items
    test_item_ids_t = torch.LongTensor(test_ids)
    with torch.no_grad():
        test_preds = mf_model.predict_for_user(p_cold, test_item_ids_t).cpu().numpy()
    rmse_val = rmse(test_preds, test_labels)

    # Ranking: score all items, rank by predicted score
    all_ids_t = torch.LongTensor(all_item_ids)
    with torch.no_grad():
        all_preds = mf_model.predict_for_user(p_cold, all_ids_t).cpu().numpy()

    ranked_indices = np.argsort(-all_preds)
    ranked_items = all_item_ids[ranked_indices].tolist()

    # Ground-truth relevant = test items with above-average rating
    mean_label = test_labels.mean()
    relevant = [iid for iid, r in test_ratings if r >= mean_label]

    results = {"rmse": rmse_val}
    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(ranked_items, relevant, k)
        results[f"recall@{k}"] = recall_at_k(ranked_items, relevant, k)
        results[f"ndcg@{k}"] = ndcg_at_k(ranked_items, relevant, k)
        results[f"hit_rate@{k}"] = hit_rate_at_k(ranked_items, relevant, k)

    return results


def two_sample_ttest(a, b):
    """Two-sample Student's t-test between two score arrays.

    Returns:
        dict with t_stat, p_value, significant_01, significant_05, significant_10
    """
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant_01": bool(p_value < 0.01),
        "significant_05": bool(p_value < 0.05),
        "significant_10": bool(p_value < 0.10),
    }


def run_pairwise_ttests(results_by_method, metric="rmse"):
    """Run pairwise t-tests between all method pairs.

    Args:
        results_by_method: dict {method_name: [metric_value_per_user]}
        metric: name for reporting

    Returns:
        dict of {(method_a, method_b): ttest_result}
    """
    methods = list(results_by_method.keys())
    comparisons = {}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a_name = methods[i]
            b_name = methods[j]
            a_scores = np.array(results_by_method[a_name])
            b_scores = np.array(results_by_method[b_name])
            comparisons[(a_name, b_name)] = two_sample_ttest(a_scores, b_scores)
    return comparisons
