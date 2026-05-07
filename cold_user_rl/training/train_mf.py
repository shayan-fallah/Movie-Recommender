import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from data.feedback_constructor import compute_hybrid_loss


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _make_obs_pairs(rating_matrix):
    """Return arrays of (user_ids, item_ids, ratings) for observed entries."""
    mask = ~np.isnan(rating_matrix)
    user_ids, item_ids = np.where(mask)
    ratings = rating_matrix[user_ids, item_ids].astype(np.float32)
    return user_ids.astype(np.int64), item_ids.astype(np.int64), ratings


def train_mf(mf_model, feedback_bundle, config, logger):
    """Train the MF model on warm users.

    Args:
        mf_model: MatrixFactorization instance
        feedback_bundle: FeedbackBundle (explicit/implicit/confidence)
        config: CONFIG dict
        logger: ExperimentLogger

    Returns:
        Trained mf_model
    """
    use_hybrid = config.get("use_hybrid_feedback", False)
    lr = config.get("mf_learning_rate", 0.001)
    lam = config.get("mf_regularization", 0.01)
    lam_e = config.get("explicit_weight", 0.7)
    lam_i = config.get("implicit_weight", 0.3)
    n_iter = config.get("mf_iterations", 100)
    batch_size = 1024

    optimizer = torch.optim.Adam(mf_model.parameters(), lr=lr, weight_decay=0.0)
    mf_model.train()

    explicit = feedback_bundle.explicit
    implicit = feedback_bundle.implicit
    confidence = feedback_bundle.confidence

    user_ids, item_ids, ratings = _make_obs_pairs(explicit)
    n_obs = len(user_ids)

    logger.info(f"Training MF: {n_obs:,} observations, {n_iter} iterations, "
                f"use_hybrid={use_hybrid}")

    best_val_loss = float("inf")
    best_state = None

    for epoch in trange(n_iter, desc="MF Training"):
        perm = np.random.permutation(n_obs)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_obs, batch_size):
            idx = perm[start: start + batch_size]
            u_t = torch.LongTensor(user_ids[idx])
            i_t = torch.LongTensor(item_ids[idx])
            r_t = torch.FloatTensor(ratings[idx])

            optimizer.zero_grad()
            preds = mf_model(u_t, i_t)

            if use_hybrid:
                # Gather implicit and confidence for this batch
                imp_vals = implicit[user_ids[idx], item_ids[idx]]
                conf_vals = confidence[user_ids[idx], item_ids[idx]]
                imp_t = torch.FloatTensor(np.where(np.isnan(imp_vals), 0.0, imp_vals))
                # Explicit targets: use actual ratings (not NaN here since we sampled obs pairs)
                loss = compute_hybrid_loss(
                    preds, r_t, imp_t,
                    torch.FloatTensor(conf_vals),
                    lam_e, lam_i
                )
            else:
                loss = nn.functional.mse_loss(preds, r_t)

            # L2 regularization
            reg = lam * (
                mf_model.P.weight.norm(2) ** 2 +
                mf_model.Q.weight.norm(2) ** 2
            )
            (loss + reg).backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_state = {k: v.clone() for k, v in mf_model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(f"MF Epoch {epoch+1}/{n_iter} | Loss {avg_loss:.4f}")
            logger.log_scalar("mf/train_loss", avg_loss, epoch)

    if best_state is not None:
        mf_model.load_state_dict(best_state)

    mf_model.eval()
    logger.info(f"MF training complete. Best loss: {best_val_loss:.4f}")
    return mf_model


def evaluate_mf_val(mf_model, val_matrix, config):
    """RMSE on validation set (observed entries only)."""
    mf_model.eval()
    user_ids, item_ids, ratings = _make_obs_pairs(val_matrix)
    if len(user_ids) == 0:
        return float("inf")
    u_t = torch.LongTensor(user_ids)
    i_t = torch.LongTensor(item_ids)
    r_t = torch.FloatTensor(ratings)
    with torch.no_grad():
        preds = mf_model(u_t, i_t)
    rmse = float(torch.sqrt(nn.functional.mse_loss(preds, r_t)).item())
    return rmse


def load_or_train_mf(checkpoint_path, mf_model, feedback_bundle, config, logger):
    """Load MF checkpoint if it exists, otherwise train and save.

    Args:
        checkpoint_path: path to .pt file
        mf_model: MatrixFactorization instance (architecture must match checkpoint)
        feedback_bundle: FeedbackBundle
        config: CONFIG dict
        logger: ExperimentLogger

    Returns:
        Trained (or loaded) mf_model
    """
    if os.path.isfile(checkpoint_path):
        logger.info(f"Loading MF checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        mf_model.load_state_dict(ckpt["model_state"])
        mf_model.eval()
        return mf_model

    mf_model = train_mf(mf_model, feedback_bundle, config, logger)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({"model_state": mf_model.state_dict()}, checkpoint_path)
    logger.save_checkpoint_meta(checkpoint_path)
    logger.info(f"MF checkpoint saved to {checkpoint_path}")
    return mf_model
