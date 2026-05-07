import numpy as np

from environment.cold_user_env import ColdUserEnv


class ColdUserEnvExtended(ColdUserEnv):
    """Extended environment supporting Modules 2 (Personalized AL) and 4 (Hierarchical RL).

    State = [shown_items_binary | cu_latent_vector | uncertainty_scores | diversity_scores]
    Exact composition controlled by config flags.

    When use_hierarchical_rl=True, step() accepts action = (strategy_id, item_pool_idx)
    instead of a plain integer index.
    """

    def __init__(self, mf_model, finetuner, feedback_bundle, cold_split,
                 action_pool, item_metadata, config, al_selector=None):
        """
        Args:
            al_selector: PersonalizedALSelector instance (None if Module 2 inactive)
            (all other args same as ColdUserEnv)
        """
        super().__init__(
            mf_model, finetuner, feedback_bundle, cold_split,
            action_pool, item_metadata, config
        )
        self.al_selector = al_selector
        self._use_al = config.get("use_personalized_al", False)
        self._use_hier = config.get("use_hierarchical_rl", False)

    @property
    def state_dim(self):
        base = self.action_pool_size
        if self._use_al and self.al_selector is not None:
            return base + self.al_selector.state_extension_dim
        return base

    def reset(self, user_id=None):
        base_state = super().reset(user_id=user_id)

        if self._use_al and self.al_selector is not None:
            self.al_selector.reset(
                self.current_user,
                self.p_cold,
                self.action_pool,
            )

        return self._build_state()

    def step(self, action):
        """Execute one interview step.

        Args:
            action: int (item pool index) OR (strategy_id, item_pool_idx) tuple
                    when use_hierarchical_rl=True

        Returns:
            (next_state, reward, done, info)
        """
        if self._use_hier and isinstance(action, (tuple, list)):
            strategy_id, item_pool_idx = int(action[0]), int(action[1])
        else:
            strategy_id = None
            item_pool_idx = int(action)

        # Execute base step
        _, reward, done, info = super().step(item_pool_idx)

        if strategy_id is not None:
            info["strategy_id"] = strategy_id

        # Update AL selector after base step (p_cold has been updated inside base)
        if self._use_al and self.al_selector is not None:
            item_id = int(self.action_pool[item_pool_idx])
            self.al_selector.update(item_id, self.p_cold)

        return self._build_state(), reward, done, info

    def _build_state(self):
        base_state = self.shown_mask.copy()

        if self._use_al and self.al_selector is not None:
            ext = self.al_selector.get_state_extension()
            return np.concatenate([base_state, ext]).astype(np.float32)

        return base_state
