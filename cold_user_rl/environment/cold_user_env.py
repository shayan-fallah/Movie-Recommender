import numpy as np
import torch


class ColdUserEnv:
    """RL environment for the cold-user interview loop.

    One episode = one cold user's interview.

    State:
        Binary vector of length action_pool_size.
        state[i] = 1.0 if item i has been shown, else 0.0.

    Action:
        Integer in [0, action_pool_size) — index into the action pool.
        Must not already be in shown_items.

    Reward:
        1.0 / (RMSE + 1e-8), clipped to [0, reward_clip_max].

    Terminal:
        When interview_size items have been shown.
    """

    def __init__(self, mf_model, finetuner, feedback_bundle, cold_split,
                 action_pool, item_metadata, config):
        """
        Args:
            mf_model: MatrixFactorization instance (warm weights frozen)
            finetuner: ColdUserFinetuner instance
            feedback_bundle: FeedbackBundle (explicit/implicit/confidence arrays)
            cold_split: dict {user_id: {"interview_pool": [...], "test_set": [...]}}
            action_pool: (action_pool_size,) array of item IDs (top-N popular items)
            item_metadata: dict with popularity, item_embeddings etc.
            config: CONFIG dict
        """
        self.mf = mf_model
        self.finetuner = finetuner
        self.feedback_bundle = feedback_bundle
        self.cold_split = cold_split
        self.action_pool = np.array(action_pool, dtype=np.int64)
        self.action_pool_size = len(action_pool)
        self.item_metadata = item_metadata
        self.config = config

        self.cold_user_ids = list(cold_split.keys())
        self.interview_size = config["interview_sizes"][0]  # default; overridable

        # Episode state
        self.current_user = None
        self.p_cold = None
        self.shown_mask = None     # binary vector (action_pool_size,)
        self.observed_items = None # list of (item_id, rating) from ground truth
        self.interview_pool_set = None  # set of item_ids available for this user
        self.step_count = 0
        self.rmse_trajectory = []

    @property
    def state_dim(self):
        return self.action_pool_size

    def reset(self, user_id=None):
        """Sample a cold user and begin a new interview episode.

        Returns:
            Initial state vector (np.ndarray, shape (state_dim,))
        """
        if user_id is None:
            user_id = np.random.choice(self.cold_user_ids)

        self.current_user = user_id
        user_data = self.cold_split[user_id]
        self.interview_pool_set = set(item_id for item_id, _ in user_data["interview_pool"])
        self.test_set = user_data["test_set"]  # list of (item_id, rating)

        # Initialize cold user vector from similar warm users
        interview_items = [item_id for item_id, _ in user_data["interview_pool"]]
        self.p_cold = self.finetuner.initialize_cold_vector(interview_items)

        self.shown_mask = np.zeros(self.action_pool_size, dtype=np.float32)
        self.observed_items = []
        self.step_count = 0
        self.rmse_trajectory = []

        return self._get_state()

    def step(self, action):
        """Execute one interview step.

        Args:
            action: integer index into action_pool

        Returns:
            (next_state, reward, done, info)
        """
        assert 0 <= action < self.action_pool_size, f"Invalid action {action}"
        assert self.shown_mask[action] == 0.0, f"Item {action} already shown"

        item_id = int(self.action_pool[action])

        # Get user's response from ground truth
        rating, is_explicit = self._get_cold_user_rating(item_id)

        # Record interaction
        self.observed_items.append((item_id, rating))

        # Optimize cold user vector with updated observations
        self.p_cold, _ = self.finetuner.finetune(self.p_cold, self.observed_items)

        # Mark item as shown
        self.shown_mask[action] = 1.0
        self.step_count += 1

        # Compute RMSE on held-out test set
        rmse = self.finetuner.compute_rmse(self.p_cold, self.test_set)
        self.rmse_trajectory.append(rmse)

        reward = self._compute_reward(rmse)
        done = self.step_count >= self.interview_size

        info = {
            "rmse": rmse,
            "item_id": item_id,
            "rating": rating,
            "is_explicit": is_explicit,
            "user_id": self.current_user,
            "step": self.step_count,
        }

        return self._get_state(), reward, done, info

    def _get_cold_user_rating(self, item_id):
        """Retrieve cold user's rating for item_id from ground truth.

        Returns (rating, is_explicit).
        If item is not in the user's explicit interactions, returns (0.0, False)
        as an implicit negative — the user has not engaged with it.
        """
        user_data = self.cold_split[self.current_user]
        # Check interview pool
        for iid, r in user_data["interview_pool"]:
            if iid == item_id:
                return float(r), True
        # Not found — implicit negative
        return 0.0, False

    def _compute_reward(self, rmse):
        reward = 1.0 / (rmse + 1e-8)
        return float(np.clip(reward, 0.0, self.config.get("reward_clip_max", 10.0)))

    def get_valid_actions(self):
        """Return indices of items in action_pool that have not yet been shown
        AND are present in the cold user's interview pool."""
        valid = []
        for idx, item_id in enumerate(self.action_pool):
            if self.shown_mask[idx] == 0.0 and int(item_id) in self.interview_pool_set:
                valid.append(idx)
        if not valid:
            # Fallback: any unshown item (when pool exhausted)
            valid = [i for i in range(self.action_pool_size) if self.shown_mask[i] == 0.0]
        return np.array(valid, dtype=np.int64)

    def _get_state(self):
        return self.shown_mask.copy()

    def set_interview_size(self, n):
        self.interview_size = n
