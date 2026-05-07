import numpy as np

from active_learning.strategies import (
    apply_strategy,
    compute_diversity_scores,
    compute_uncertainty_scores,
)


class PersonalizedALSelector:
    """Dynamically recomputes uncertainty and diversity scores after each step.

    Used when config['use_personalized_al'] = True (Module 2).
    """

    def __init__(self, mf_model, config, item_metadata):
        self.mf = mf_model
        self.config = config
        self.item_metadata = item_metadata
        self.top_k = config.get("uncertainty_top_k", 20)
        self.diversity_weight = config.get("diversity_weight", 0.5)

        # Per-episode state
        self.p_cold = None
        self.all_candidate_ids = None
        self.shown_items = set()
        self.uncertainty_scores = None  # (n_candidates,)
        self.diversity_scores = None    # (n_candidates,)
        self._id_to_idx = {}            # item_id -> index in all_candidate_ids

    def reset(self, cold_user_id, p_cold, all_candidate_ids):
        """Initialize for a new episode.

        Args:
            cold_user_id: int (unused here, kept for interface symmetry)
            p_cold: (k,) tensor
            all_candidate_ids: numpy array of candidate item IDs to track
        """
        self.p_cold = p_cold
        self.all_candidate_ids = np.array(all_candidate_ids, dtype=np.int64)
        self.shown_items = set()
        self._id_to_idx = {int(iid): i for i, iid in enumerate(all_candidate_ids)}

        self._recompute_uncertainty()
        # No items shown yet — diversity = 1 for all
        self.diversity_scores = np.ones(len(all_candidate_ids), dtype=np.float32)

    def update(self, shown_item_id, p_cold):
        """Called after each interview step.

        Updates uncertainty (p_cold changed) and diversity (shown set grew).
        This is idempotent: calling with the same item twice does nothing the second time.
        """
        if shown_item_id in self.shown_items:
            return

        self.shown_items.add(shown_item_id)
        self.p_cold = p_cold
        self._recompute_uncertainty()
        self._recompute_diversity()

    def _recompute_uncertainty(self):
        self.uncertainty_scores = compute_uncertainty_scores(
            self.mf, self.p_cold, self.all_candidate_ids
        )

    def _recompute_diversity(self):
        item_embeddings = self.item_metadata.get("item_embeddings", None)
        if item_embeddings is None:
            self.diversity_scores = np.ones(len(self.all_candidate_ids), dtype=np.float32)
            return
        self.diversity_scores = compute_diversity_scores(
            self.all_candidate_ids, list(self.shown_items), item_embeddings
        )

    def get_state_extension(self):
        """Return the additional state features for Module 2.

        Returns:
            numpy array of shape (k + 2 * uncertainty_top_k,)
        """
        include_cu = self.config.get("state_include_cu_vector", True)
        include_unc = self.config.get("state_include_uncertainty", True)
        include_div = self.config.get("state_include_diversity", True)

        parts = []

        if include_cu and self.p_cold is not None:
            parts.append(self.p_cold.detach().cpu().numpy())

        if include_unc and self.uncertainty_scores is not None:
            unc = self.uncertainty_scores[: self.top_k]
            # Pad if fewer candidates than top_k
            if len(unc) < self.top_k:
                unc = np.pad(unc, (0, self.top_k - len(unc)))
            parts.append(unc)

        if include_div and self.diversity_scores is not None:
            div = self.diversity_scores[: self.top_k]
            if len(div) < self.top_k:
                div = np.pad(div, (0, self.top_k - len(div)))
            parts.append(div)

        if not parts:
            return np.array([], dtype=np.float32)

        return np.concatenate(parts).astype(np.float32)

    @property
    def state_extension_dim(self):
        k = self.config.get("mf_latent_features", 10)
        top_k = self.config.get("uncertainty_top_k", 20)
        dim = 0
        if self.config.get("state_include_cu_vector", True):
            dim += k
        if self.config.get("state_include_uncertainty", True):
            dim += top_k
        if self.config.get("state_include_diversity", True):
            dim += top_k
        return dim

    def get_candidate_ranking(self, strategy_id=None, top_k=None):
        """Return ranked candidate items.

        If strategy_id is None, rank by combined uncertainty + diversity score.
        """
        candidates = self.all_candidate_ids
        shown_set = self.shown_items

        if strategy_id is None:
            unc = self.uncertainty_scores if self.uncertainty_scores is not None else np.ones(len(candidates))
            div = self.diversity_scores if self.diversity_scores is not None else np.ones(len(candidates))
            dw = self.diversity_weight
            combined = (1 - dw) * unc + dw * div
            ranked = candidates[np.argsort(-combined)]
        else:
            ranked = apply_strategy(
                strategy_id, self.mf, self.p_cold,
                candidates, shown_set, self.item_metadata,
                top_k=len(candidates),
            )

        # Exclude shown
        ranked = np.array([i for i in ranked if int(i) not in shown_set], dtype=np.int64)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked
