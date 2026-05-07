from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class FeedbackBundle:
    explicit: np.ndarray    # (n_users, n_items) float32, np.nan for missing
    implicit: np.ndarray    # (n_users, n_items) float32, 0/1/nan
    confidence: np.ndarray  # (n_users, n_items) float32, >= 1.0 for rated


def build_explicit_matrix(df, n_users, n_items, normalize=True):
    """E[u,i] = rating/5 if rated, else np.nan."""
    mat = np.full((n_users, n_items), np.nan, dtype=np.float32)
    for row in df.itertuples(index=False):
        r = float(row.rating)
        if normalize:
            r /= 5.0
        mat[int(row.userId), int(row.movieId)] = r
    return mat


def build_implicit_matrix(df, threshold, n_users, n_items):
    """I[u,i] = 1 if rating >= threshold, 0 if rated below threshold, np.nan if never rated."""
    mat = np.full((n_users, n_items), np.nan, dtype=np.float32)
    for row in df.itertuples(index=False):
        mat[int(row.userId), int(row.movieId)] = 1.0 if float(row.rating) >= threshold else 0.0
    return mat


def build_confidence_matrix(df, tags_df, n_users, n_items, tag_bonus=0.2):
    """C[u,i] = 1.0 base for all rated items, + tag_bonus if user also tagged the movie."""
    mat = np.zeros((n_users, n_items), dtype=np.float32)
    for row in df.itertuples(index=False):
        mat[int(row.userId), int(row.movieId)] = 1.0

    if tags_df is not None and len(tags_df) > 0:
        for row in tags_df.itertuples(index=False):
            uid = int(row.userId)
            mid = int(row.movieId)
            if uid < n_users and mid < n_items and mat[uid, mid] > 0:
                mat[uid, mid] += tag_bonus

    return mat


def compute_hybrid_loss(predictions, explicit_targets, implicit_targets,
                        confidence, lambda_e, lambda_i):
    """
    Hybrid MF loss combining explicit and implicit feedback.

    Args:
        predictions: (N,) predicted scores for observed (user, item) pairs
        explicit_targets: (N,) explicit ratings (NaN for pairs with no explicit rating)
        implicit_targets: (N,) binary implicit labels (NaN for unobserved)
        confidence: (N,) confidence weights for implicit term
        lambda_e: weight for explicit loss term
        lambda_i: weight for implicit loss term

    Returns:
        Scalar loss tensor
    """
    loss = torch.tensor(0.0, requires_grad=True)

    # Explicit term — only observed explicit ratings
    explicit_mask = ~torch.isnan(explicit_targets)
    if explicit_mask.any():
        diff_e = predictions[explicit_mask] - explicit_targets[explicit_mask]
        explicit_loss = (diff_e ** 2).mean()
        loss = loss + lambda_e * explicit_loss

    # Implicit term — confidence-weighted, only where implicit label is known
    implicit_mask = ~torch.isnan(implicit_targets)
    if implicit_mask.any():
        diff_i = predictions[implicit_mask] - implicit_targets[implicit_mask]
        conf = confidence[implicit_mask]
        implicit_loss = (conf * diff_i ** 2).mean()
        loss = loss + lambda_i * implicit_loss

    return loss


def build_feedback_bundle(train_df, tags_df, n_users, n_items, config):
    """Construct FeedbackBundle from training dataframe and config."""
    normalize = config.get("normalize_ratings", True)
    threshold = config.get("implicit_threshold", 3.5)
    tag_bonus = config.get("tag_confidence_bonus", 0.2)

    explicit = build_explicit_matrix(train_df, n_users, n_items, normalize)
    implicit = build_implicit_matrix(train_df, threshold, n_users, n_items)
    confidence = build_confidence_matrix(train_df, tags_df, n_users, n_items, tag_bonus)

    return FeedbackBundle(explicit=explicit, implicit=implicit, confidence=confidence)


def load_tags(data_dir, user_map, item_map):
    """Load tags.csv and remap IDs. Returns None if tags.csv not present."""
    import os
    path = os.path.join(data_dir, "tags.csv")
    if not os.path.isfile(path):
        return None
    tags = pd.read_csv(path, dtype={"userId": int, "movieId": int})
    tags = tags[tags["userId"].isin(user_map) & tags["movieId"].isin(item_map)].copy()
    tags["userId"] = tags["userId"].map(user_map)
    tags["movieId"] = tags["movieId"].map(item_map)
    return tags
