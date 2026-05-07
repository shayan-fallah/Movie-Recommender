import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, k, dropout_rate=0.0):
        super().__init__()
        self.k = k
        self.dropout_rate = dropout_rate

        self.P = nn.Embedding(n_users, k)  # user latent matrix
        self.Q = nn.Embedding(n_items, k)  # item latent matrix
        self.b_u = nn.Embedding(n_users, 1)
        self.b_i = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(p=dropout_rate)

        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        nn.init.zeros_(self.b_u.weight)
        nn.init.zeros_(self.b_i.weight)

    def forward(self, user_ids, item_ids):
        p = self.dropout(self.P(user_ids))
        q = self.dropout(self.Q(item_ids))
        dot = (p * q).sum(dim=-1)
        pred = self.global_bias + self.b_u(user_ids).squeeze(-1) + self.b_i(item_ids).squeeze(-1) + dot
        return pred

    def predict_for_user(self, p_cold_vector, item_ids):
        """Predict for a cold user given their latent vector.

        Args:
            p_cold_vector: (k,) tensor — cold user's latent vector
            item_ids: (N,) long tensor — item indices to score

        Returns:
            (N,) tensor of predicted scores
        """
        Q_frozen = self.Q.weight.data[item_ids]  # (N, k)
        b_i = self.b_i.weight.data[item_ids].squeeze(-1)  # (N,)
        dot = (p_cold_vector.unsqueeze(0) * Q_frozen).sum(dim=-1)  # (N,)
        return self.global_bias.data + dot + b_i

    def get_item_matrix(self):
        """Return Q weight matrix, detached from graph."""
        return self.Q.weight.data

    def get_user_vector(self, user_id):
        return self.P.weight.data[user_id]

    def get_all_user_vectors(self):
        return self.P.weight.data


class ColdUserFinetuner:
    """Optimizes a single cold user's latent vector given frozen P and Q."""

    def __init__(self, mf_model, config):
        self.mf = mf_model
        self.config = config

    def initialize_cold_vector(self, cold_user_items, all_item_ids=None):
        """Initialize cold user vector as mean of top-K similar warm users' vectors.

        Similarity: cosine similarity between the mean item embedding of the
        cold user's interaction set and each warm user's P vector.

        Falls back to random initialization if similarity cannot be computed.

        Args:
            cold_user_items: list of item_ids the cold user has interacted with
            all_item_ids: optional array of all warm user item indices (unused)

        Returns:
            (k,) torch.Tensor — initial cold user vector
        """
        k = self.config["mf_latent_features"]
        top_k = self.config.get("cold_init_top_k_similar", 5)

        if len(cold_user_items) == 0:
            return torch.randn(k) * 0.01

        item_ids = torch.tensor(cold_user_items, dtype=torch.long)
        valid = item_ids[item_ids < self.mf.Q.weight.shape[0]]
        if len(valid) == 0:
            return torch.randn(k) * 0.01

        mean_item_vec = self.mf.Q.weight.data[valid].mean(dim=0)  # (k,)

        P = self.mf.get_all_user_vectors()  # (n_warm, k)
        P_norm = F.normalize(P, dim=-1)
        mean_norm = F.normalize(mean_item_vec.unsqueeze(0), dim=-1)
        sims = (P_norm * mean_norm).sum(dim=-1)  # (n_warm,)

        top_k_actual = min(top_k, P.shape[0])
        top_indices = sims.topk(top_k_actual).indices
        p_init = P[top_indices].mean(dim=0).clone()
        return p_init

    def finetune(self, p_cold_init, observed_items):
        """Optimize cold user vector against frozen item matrix.

        Args:
            p_cold_init: (k,) tensor — starting vector
            observed_items: list of (item_id, rating) tuples

        Returns:
            (p_cold, final_loss): optimized vector (detached) and scalar loss
        """
        if not observed_items:
            return p_cold_init.detach().clone(), 0.0

        use_hybrid = self.config.get("use_hybrid_feedback", False)
        freeze = self.config.get("freeze_warm_mf", True)
        lr = self.config.get("cold_vector_lr", 0.01)
        steps = self.config.get("cold_vector_steps", 50)
        lam = self.config.get("mf_regularization", 0.01)

        item_ids = torch.tensor([x[0] for x in observed_items], dtype=torch.long)
        ratings = torch.tensor([x[1] for x in observed_items], dtype=torch.float32)

        if freeze:
            Q_frozen = self.mf.Q.weight.data[item_ids].detach()  # (N, k)
            b_i_frozen = self.mf.b_i.weight.data[item_ids].squeeze(-1).detach()
            gb = self.mf.global_bias.data.detach()

            p_cold = p_cold_init.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([p_cold], lr=lr)

            for _ in range(steps):
                optimizer.zero_grad()
                preds = gb + (p_cold.unsqueeze(0) * Q_frozen).sum(-1) + b_i_frozen

                if use_hybrid:
                    loss = self._hybrid_loss_single(preds, ratings, item_ids)
                else:
                    loss = F.mse_loss(preds, ratings)

                reg = lam * (p_cold ** 2).sum()
                (loss + reg).backward()
                optimizer.step()

            return p_cold.detach(), loss.item()
        else:
            # Base paper behavior: full MF retraining not practical in inference;
            # fall back to single-vector optimization even when flag is False
            # (full retraining would require full training data, handled in train_mf.py)
            return self.finetune.__wrapped__(self, p_cold_init, observed_items) \
                if hasattr(self.finetune, '__wrapped__') \
                else self._finetune_full(p_cold_init, observed_items)

    def _hybrid_loss_single(self, preds, explicit_targets, item_ids):
        """Compute hybrid loss for a single cold user's interactions."""
        from data.feedback_constructor import compute_hybrid_loss

        threshold = self.config.get("implicit_threshold", 3.5)
        lam_e = self.config.get("explicit_weight", 0.7)
        lam_i = self.config.get("implicit_weight", 0.3)
        normalize = self.config.get("normalize_ratings", True)

        explicit_t = explicit_targets.clone()

        thr = threshold / 5.0 if normalize else threshold
        implicit_t = (explicit_targets >= thr).float()

        confidence = torch.ones_like(explicit_targets)
        return compute_hybrid_loss(preds, explicit_t, implicit_t, confidence, lam_e, lam_i)

    def _finetune_full(self, p_cold_init, observed_items):
        """Fallback for freeze_warm_mf=False: same single-vector optimization."""
        return self.finetune(p_cold_init, observed_items)

    def compute_rmse(self, p_cold, test_items):
        """RMSE on held-out test items.

        Args:
            p_cold: (k,) tensor
            test_items: list of (item_id, rating) tuples

        Returns:
            float RMSE
        """
        if not test_items:
            return float("inf")

        item_ids = torch.tensor([x[0] for x in test_items], dtype=torch.long)
        targets = torch.tensor([x[1] for x in test_items], dtype=torch.float32)

        # Filter out-of-range item IDs
        n_items = self.mf.Q.weight.shape[0]
        valid_mask = item_ids < n_items
        if not valid_mask.any():
            return float("inf")

        item_ids = item_ids[valid_mask]
        targets = targets[valid_mask]

        with torch.no_grad():
            Q_frozen = self.mf.Q.weight.data[item_ids]
            b_i = self.mf.b_i.weight.data[item_ids].squeeze(-1)
            gb = self.mf.global_bias.data
            preds = gb + (p_cold.unsqueeze(0) * Q_frozen).sum(-1) + b_i

        diff = preds - targets
        rmse = float(torch.sqrt((diff ** 2).mean()).item())
        return rmse
