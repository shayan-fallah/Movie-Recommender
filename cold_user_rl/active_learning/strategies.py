import numpy as np
import torch
import torch.nn.functional as F

STRATEGY_NAMES = [
    "popularity",   # 0
    "entropy",      # 1
    "gini",         # 2
    "popent",       # 3
    "popgini",      # 4
    "error",        # 5
    "poperror",     # 6
    "variance",     # 7
    "popvar",       # 8
]


def compute_uncertainty_scores(mf_model, p_cold, candidate_item_ids, n_mc=20):
    """Approximate per-item uncertainty via MC dropout.

    Runs n_mc forward passes with the model in train() mode (activates dropout).
    If dropout_rate=0, falls back to the simpler |pred - 0.5| measure.

    Args:
        mf_model: MatrixFactorization model
        p_cold: (k,) tensor — cold user latent vector
        candidate_item_ids: (N,) array of item indices

    Returns:
        (N,) numpy array of uncertainty scores (higher = more uncertain)
    """
    item_ids = torch.tensor(candidate_item_ids, dtype=torch.long)

    if mf_model.dropout_rate > 0:
        mf_model.train()
        preds_list = []
        with torch.no_grad():
            for _ in range(n_mc):
                preds = mf_model.predict_for_user(p_cold, item_ids)
                preds_list.append(preds.cpu().numpy())
        mf_model.eval()
        preds_arr = np.stack(preds_list, axis=0)  # (n_mc, N)
        uncertainty = preds_arr.var(axis=0)        # (N,)
    else:
        with torch.no_grad():
            preds = mf_model.predict_for_user(p_cold, item_ids).cpu().numpy()
        # Items where prediction is close to 0.5 are most uncertain
        preds_clipped = np.clip(preds, 0.0, 1.0)
        uncertainty = 0.5 - np.abs(preds_clipped - 0.5)

    return uncertainty


def compute_diversity_scores(candidate_item_ids, shown_item_ids, item_embeddings):
    """Diversity of each candidate relative to already-shown items.

    Diversity(i) = min cosine_distance(Q[i], Q[j]) for j in shown_items.
    Higher diversity = item is more different from everything shown.

    Args:
        candidate_item_ids: (N,) array of item indices
        shown_item_ids: array/list of already-shown item indices
        item_embeddings: (n_items, k) numpy array (detached Q.weight.data)

    Returns:
        (N,) numpy array of diversity scores (higher = more diverse)
    """
    if len(shown_item_ids) == 0:
        return np.ones(len(candidate_item_ids), dtype=np.float32)

    Q_cand = item_embeddings[candidate_item_ids]   # (N, k)
    Q_shown = item_embeddings[list(shown_item_ids)]  # (M, k)

    # Normalize for cosine distance
    norm_cand = Q_cand / (np.linalg.norm(Q_cand, axis=-1, keepdims=True) + 1e-8)
    norm_shown = Q_shown / (np.linalg.norm(Q_shown, axis=-1, keepdims=True) + 1e-8)

    # Cosine similarity: (N, M)
    sims = norm_cand @ norm_shown.T
    # Cosine distance: 1 - sim, take min over shown items (nearest shown neighbour)
    distances = 1.0 - sims  # (N, M)
    diversity = distances.min(axis=1)  # (N,)

    return diversity.astype(np.float32)


def _popularity_scores(candidate_item_ids, item_metadata):
    pop = item_metadata.get("popularity", {})
    scores = np.array([pop.get(int(i), 0) for i in candidate_item_ids], dtype=np.float32)
    return scores


def _entropy_scores(mf_model, p_cold, candidate_item_ids):
    item_ids = torch.tensor(candidate_item_ids, dtype=torch.long)
    with torch.no_grad():
        preds = mf_model.predict_for_user(p_cold, item_ids).cpu().numpy()
    preds_c = np.clip(preds, 1e-7, 1 - 1e-7)
    entropy = -(preds_c * np.log(preds_c) + (1 - preds_c) * np.log(1 - preds_c))
    return entropy.astype(np.float32)


def _gini_scores(mf_model, p_cold, candidate_item_ids):
    item_ids = torch.tensor(candidate_item_ids, dtype=torch.long)
    with torch.no_grad():
        preds = mf_model.predict_for_user(p_cold, item_ids).cpu().numpy()
    preds_c = np.clip(preds, 0.0, 1.0)
    gini = 1.0 - preds_c ** 2 - (1 - preds_c) ** 2
    return gini.astype(np.float32)


def _error_scores(mf_model, p_cold, candidate_item_ids):
    return compute_uncertainty_scores(mf_model, p_cold, candidate_item_ids)


def _variance_scores(mf_model, p_cold, candidate_item_ids):
    return compute_uncertainty_scores(mf_model, p_cold, candidate_item_ids)


def apply_strategy(strategy_id, mf_model, p_cold, candidate_item_ids,
                   shown_item_ids, item_metadata, top_k=None):
    """Rank candidate items by the given strategy and return top_k item IDs.

    Args:
        strategy_id: int in [0, 8]
        mf_model: MatrixFactorization
        p_cold: (k,) tensor
        candidate_item_ids: numpy array of item indices to rank
        shown_item_ids: set/list of already-shown item indices (excluded)
        item_metadata: dict with 'popularity', 'item_embeddings', etc.
        top_k: return at most this many items

    Returns:
        numpy array of ranked item IDs (best first)
    """
    # Exclude shown items
    shown_set = set(int(x) for x in shown_item_ids)
    candidates = np.array([i for i in candidate_item_ids if int(i) not in shown_set], dtype=np.int64)

    if len(candidates) == 0:
        return candidates

    item_embeddings = item_metadata.get("item_embeddings", None)

    if strategy_id == 0:   # popularity
        scores = _popularity_scores(candidates, item_metadata)
    elif strategy_id == 1: # entropy
        scores = _entropy_scores(mf_model, p_cold, candidates)
    elif strategy_id == 2: # gini
        scores = _gini_scores(mf_model, p_cold, candidates)
    elif strategy_id == 3: # popent
        pop = _popularity_scores(candidates, item_metadata)
        ent = _entropy_scores(mf_model, p_cold, candidates)
        scores = _normalize(pop) * _normalize(ent)
    elif strategy_id == 4: # popgini
        pop = _popularity_scores(candidates, item_metadata)
        gin = _gini_scores(mf_model, p_cold, candidates)
        scores = _normalize(pop) * _normalize(gin)
    elif strategy_id == 5: # error
        scores = _error_scores(mf_model, p_cold, candidates)
    elif strategy_id == 6: # poperror
        pop = _popularity_scores(candidates, item_metadata)
        err = _error_scores(mf_model, p_cold, candidates)
        scores = _normalize(pop) * _normalize(err)
    elif strategy_id == 7: # variance
        scores = _variance_scores(mf_model, p_cold, candidates)
    elif strategy_id == 8: # popvar
        pop = _popularity_scores(candidates, item_metadata)
        var = _variance_scores(mf_model, p_cold, candidates)
        scores = _normalize(pop) * _normalize(var)
    else:
        raise ValueError(f"Unknown strategy_id: {strategy_id}")

    ranked_indices = np.argsort(-scores)
    ranked_items = candidates[ranked_indices]

    if top_k is not None:
        ranked_items = ranked_items[:top_k]

    return ranked_items


def _normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)


def build_item_metadata(popularity_counts, item_embeddings):
    """Helper to build the item_metadata dict expected by apply_strategy."""
    return {
        "popularity": popularity_counts,  # dict {item_id: count}
        "item_embeddings": item_embeddings,  # (n_items, k) numpy array
    }
