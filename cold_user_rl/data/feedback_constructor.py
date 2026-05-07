"""Feedback bundle construction for hybrid MF training.

FeedbackBundle stores observations as flat arrays (NOT dense matrices).
For MovieLens 32M, three dense 200K×84K matrices would require ~200 GB RAM;
flat arrays for 32M observations require ~640 MB total.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class FeedbackBundle:
    """Observation-pair arrays for all rated (user, item) pairs in the training set."""
    user_ids: np.ndarray    # (N,) int64
    item_ids: np.ndarray    # (N,) int64
    explicit: np.ndarray    # (N,) float32 — normalized ratings in [0, 1]
    implicit: np.ndarray    # (N,) float32 — binary 0/1 (rating >= threshold → 1)
    confidence: np.ndarray  # (N,) float32 — >= 1.0; higher for tagged items


def build_feedback_bundle(train_df, tags_df, n_users, n_items, config):
    """Build FeedbackBundle from training DataFrame.

    Args:
        train_df: DataFrame with columns [userId, movieId, rating]
                  Ratings must already be normalized to [0, 1] (as saved by preprocess.py).
        tags_df:  DataFrame with columns [userId, movieId] or None.
        n_users, n_items: dataset dimensions (for bounds checking only).
        config:   CONFIG dict.

    Returns:
        FeedbackBundle
    """
    normalize = config.get("normalize_ratings", True)
    threshold = config.get("implicit_threshold", 3.5)
    tag_bonus = config.get("tag_confidence_bonus", 0.2)

    # Ratings in train_df are already normalized by preprocess.py
    # Adjust threshold to the same scale
    thr = threshold / 5.0 if normalize else threshold

    user_ids = train_df["userId"].values.astype(np.int64)
    item_ids = train_df["movieId"].values.astype(np.float64)  # temp float for map
    # Ensure integer after any float conversion from parquet
    item_ids = train_df["movieId"].values.astype(np.int64)
    ratings = train_df["rating"].values.astype(np.float32)

    implicit = (ratings >= thr).astype(np.float32)

    confidence = np.ones(len(ratings), dtype=np.float32)
    if tags_df is not None and len(tags_df) > 0:
        # Build a set of (userId, movieId) pairs that have tags
        tagged = set(zip(tags_df["userId"].values, tags_df["movieId"].values))
        for i in range(len(user_ids)):
            if (int(user_ids[i]), int(item_ids[i])) in tagged:
                confidence[i] += tag_bonus

    return FeedbackBundle(
        user_ids=user_ids,
        item_ids=item_ids,
        explicit=ratings,
        implicit=implicit,
        confidence=confidence,
    )


def compute_hybrid_loss(predictions, explicit_targets, implicit_targets,
                        confidence, lambda_e, lambda_i):
    """Hybrid MF loss combining explicit and implicit feedback.

    All inputs are (N,) tensors for the observed batch.

    Args:
        predictions:      (N,) predicted scores
        explicit_targets: (N,) explicit normalized ratings
        implicit_targets: (N,) binary 0/1 implicit labels
        confidence:       (N,) confidence weights
        lambda_e:         weight for explicit loss term
        lambda_i:         weight for implicit loss term

    Returns:
        Scalar loss tensor
    """
    diff_e = predictions - explicit_targets
    explicit_loss = (diff_e ** 2).mean()

    diff_i = predictions - implicit_targets
    implicit_loss = (confidence * diff_i ** 2).mean()

    return lambda_e * explicit_loss + lambda_i * implicit_loss


def load_tags(data_dir, user_map, item_map):
    """Load tags.csv and remap IDs. Returns None if file not present."""
    path = os.path.join(data_dir, "tags.csv")
    if not os.path.isfile(path):
        return None
    tags = pd.read_csv(path, dtype={"userId": int, "movieId": int})
    tags = tags[tags["userId"].isin(user_map) & tags["movieId"].isin(item_map)].copy()
    tags["userId"] = tags["userId"].map(user_map)
    tags["movieId"] = tags["movieId"].map(item_map)
    return tags[["userId", "movieId"]]
