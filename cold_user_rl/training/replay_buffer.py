import numpy as np


class SumTree:
    """Segment tree for O(log N) priority sampling."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data = [None] * capacity
        self.write_ptr = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] += change
            parent //= 2

    def update(self, data_idx, priority):
        tree_idx = data_idx + self.capacity
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority, data):
        data_idx = self.write_ptr
        self.data[data_idx] = data
        self.update(data_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def get(self, cumulative_sum):
        """Find leaf with given cumulative priority sum."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if cumulative_sum <= self.tree[left]:
                idx = left
            else:
                cumulative_sum -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return data_idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self):
        return self.tree[1]

    def __len__(self):
        return self.n_entries


class PrioritizedReplayBuffer:
    """PER with SumTree for O(log N) sampling."""

    def __init__(self, capacity, alpha, beta_start, beta_end, beta_anneal_steps,
                 per_epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.per_epsilon = per_epsilon
        self._beta_step = (beta_end - beta_start) / max(1, beta_anneal_steps)
        self.tree = SumTree(capacity)
        self._max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        priority = self._max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.tree))
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            idx, priority, data = self.tree.get(value)
            if data is None:
                value = np.random.uniform(0, self.tree.total)
                idx, priority, data = self.tree.get(value)
            indices[i] = idx
            priorities[i] = priority
            s, a, r, ns, d = data
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        n = len(self.tree)
        probs = priorities / (self.tree.total + 1e-10)
        weights = (n * probs) ** (-self.beta)
        weights /= weights.max()

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(weights, dtype=np.float32),
            indices,
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            priority = (abs(float(err)) + self.per_epsilon) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            self.tree.update(idx, priority)

    def anneal_beta(self):
        self.beta = min(self.beta_end, self.beta + self._beta_step)

    def __len__(self):
        return len(self.tree)


class EpisodeReplayBuffer:
    """Episode-level replay buffer for DRQN (Module 3).

    Stores complete episodes as padded sequences.
    """

    def __init__(self, capacity, alpha, beta_start, beta_end, beta_anneal_steps,
                 sequence_length=10, per_epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.sequence_length = sequence_length
        self.per_epsilon = per_epsilon
        self._beta_step = (beta_end - beta_start) / max(1, beta_anneal_steps)
        self.episodes = []
        self.priorities = []
        self.write_ptr = 0
        self._max_priority = 1.0

    def push_episode(self, episode):
        """Store a complete episode.

        Args:
            episode: list of (state, action, reward, next_state, done) tuples
        """
        if not episode:
            return
        priority = self._max_priority ** self.alpha
        if len(self.episodes) < self.capacity:
            self.episodes.append(episode)
            self.priorities.append(priority)
        else:
            self.episodes[self.write_ptr] = episode
            self.priorities[self.write_ptr] = priority
        self.write_ptr = (self.write_ptr + 1) % self.capacity

    def sample_episodes(self, batch_size):
        """Sample batch_size episodes with priority weighting.

        Returns padded tensors of shape (batch, seq_len, ...).
        """
        n = len(self.episodes)
        batch_size = min(batch_size, n)
        prios = np.array(self.priorities[:n], dtype=np.float64)
        probs = prios / prios.sum()
        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        T = self.sequence_length
        # Infer state_dim from first episode
        state_dim = len(self.episodes[0][0][0])

        state_seqs = np.zeros((batch_size, T, state_dim), dtype=np.float32)
        next_state_seqs = np.zeros((batch_size, T, state_dim), dtype=np.float32)
        action_seqs = np.zeros((batch_size, T), dtype=np.int64)
        reward_seqs = np.zeros((batch_size, T), dtype=np.float32)
        done_seqs = np.zeros((batch_size, T), dtype=np.float32)
        pad_masks = np.zeros((batch_size, T), dtype=np.float32)  # 1=valid, 0=pad

        for b, ep_idx in enumerate(indices):
            ep = self.episodes[ep_idx]
            ep_len = min(len(ep), T)
            for t in range(ep_len):
                s, a, r, ns, d = ep[t]
                state_seqs[b, t] = s
                next_state_seqs[b, t] = ns
                action_seqs[b, t] = a
                reward_seqs[b, t] = r
                done_seqs[b, t] = float(d)
                pad_masks[b, t] = 1.0

        weights = (n * probs[indices]) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)

        return (
            state_seqs, action_seqs, reward_seqs,
            next_state_seqs, done_seqs, weights, indices, pad_masks
        )

    def update_episode_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            priority = (abs(float(err)) + self.per_epsilon) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            if idx < len(self.priorities):
                self.priorities[idx] = priority

    def anneal_beta(self):
        self.beta = min(self.beta_end, self.beta + self._beta_step)

    def __len__(self):
        return len(self.episodes)


class UnifiedReplayBuffer:
    """Single interface for train_rl.py regardless of which modules are active."""

    def __init__(self, config):
        alpha = 0.6
        beta_start = 0.4
        beta_end = 1.0
        capacity = config["buffer_size"]
        beta_anneal = capacity
        per_epsilon = 1e-6

        if config.get("use_recurrent_dqn", False):
            self._buffer = EpisodeReplayBuffer(
                capacity, alpha, beta_start, beta_end, beta_anneal,
                sequence_length=config.get("sequence_length", 10),
                per_epsilon=per_epsilon,
            )
            self._mode = "episode"
        else:
            self._buffer = PrioritizedReplayBuffer(
                capacity, alpha, beta_start, beta_end, beta_anneal,
                per_epsilon=per_epsilon,
            )
            self._mode = "transition"

    def push(self, *args):
        if self._mode == "transition":
            self._buffer.push(*args)
        # For episode mode, push_episode is called directly

    def push_episode(self, episode):
        if self._mode == "episode":
            self._buffer.push_episode(episode)

    def sample(self, batch_size):
        return self._buffer.sample(batch_size)

    def sample_episodes(self, batch_size):
        return self._buffer.sample_episodes(batch_size)

    def update_priorities(self, indices, td_errors):
        self._buffer.update_priorities(indices, td_errors)

    def update_episode_priorities(self, indices, td_errors):
        if hasattr(self._buffer, "update_episode_priorities"):
            self._buffer.update_episode_priorities(indices, td_errors)

    def anneal_beta(self):
        self._buffer.anneal_beta()

    @property
    def mode(self):
        return self._mode

    def __len__(self):
        return len(self._buffer)
