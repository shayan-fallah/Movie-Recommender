import numpy as np
import torch
import torch.nn.functional as F

from active_learning.strategies import apply_strategy
from models.dqn_base import DQNAgent, DQNNetwork
from training.replay_buffer import PrioritizedReplayBuffer


class HierarchicalRLAgent:
    """Two-level RL agent: Manager (strategy) → Worker (item) (Module 4).

    Manager selects which of the 9 AL strategies to apply.
    Worker selects which specific item from the top-K candidates of that strategy.

    Manager updates every manager_update_frequency worker steps.
    Worker updates every step.
    """

    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.gamma = config["gamma"]
        self.tau = config.get("tau", 0.005)

        n_strategies = config.get("manager_action_space", 9)
        worker_k = config.get("worker_candidate_k", 10)
        h1 = config.get("hidden_layer_1", 64)
        h2 = config.get("hidden_layer_2", 32)
        lr = config["learning_rate_dqn"]
        buf_cap = config["buffer_size"]

        # Manager
        self.manager_net = DQNNetwork(state_dim, n_strategies, h1, h2)
        self.manager_target = DQNNetwork(state_dim, n_strategies, h1, h2)
        self.manager_target.load_state_dict(self.manager_net.state_dict())
        self.manager_target.eval()
        self.manager_optimizer = torch.optim.Adam(self.manager_net.parameters(), lr=lr)

        # Worker
        self.worker_net = DQNNetwork(state_dim, worker_k, h1, h2)
        self.worker_target = DQNNetwork(state_dim, worker_k, h1, h2)
        self.worker_target.load_state_dict(self.worker_net.state_dict())
        self.worker_target.eval()
        self.worker_optimizer = torch.optim.Adam(self.worker_net.parameters(), lr=lr)

        # Separate replay buffers
        self.manager_buffer = PrioritizedReplayBuffer(
            buf_cap, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_anneal_steps=buf_cap
        )
        self.worker_buffer = PrioritizedReplayBuffer(
            buf_cap, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_anneal_steps=buf_cap
        )

        self.n_strategies = n_strategies
        self.worker_k = worker_k
        self.manager_update_freq = config.get("manager_update_frequency", 5)
        self._worker_step_count = 0

        # Per-episode manager tracking
        self._manager_state = None
        self._manager_strategy = None
        self._manager_cumulative_reward = 0.0
        self.item_metadata = None  # injected by training loop

    def set_item_metadata(self, item_metadata):
        self.item_metadata = item_metadata

    def select_action(self, state, valid_actions, epsilon, mf_model, p_cold):
        """Two-level action selection.

        Args:
            state: (state_dim,) numpy array
            valid_actions: numpy array of valid item pool indices
            epsilon: exploration probability
            mf_model: MatrixFactorization (used by strategy to rank items)
            p_cold: (k,) tensor (current cold user vector)

        Returns:
            (strategy_id, item_pool_idx, top_k_candidates)
            where item_pool_idx is an index into the action_pool
        """
        # Manager selects strategy
        if np.random.random() < epsilon:
            strategy_id = np.random.randint(self.n_strategies)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            self.manager_net.eval()
            with torch.no_grad():
                q_strategies = self.manager_net(state_t).squeeze(0)
            strategy_id = int(q_strategies.argmax().item())

        # Get top-K candidates for this strategy from the valid action pool
        # valid_actions are indices into the action_pool
        if self.item_metadata is not None:
            # Map valid_actions (pool indices) to item_ids
            # This is handled externally; here we pass indices directly
            top_k_candidates = self._get_top_k_candidates(
                strategy_id, valid_actions, mf_model, p_cold
            )
        else:
            top_k_candidates = valid_actions[: self.worker_k]

        if len(top_k_candidates) == 0:
            top_k_candidates = valid_actions[: self.worker_k] if len(valid_actions) > 0 else np.array([0])

        # Worker selects slot in [0, len(top_k_candidates))
        n_slots = len(top_k_candidates)
        if np.random.random() < epsilon:
            slot = np.random.randint(n_slots)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            self.worker_net.eval()
            with torch.no_grad():
                q_slots = self.worker_net(state_t).squeeze(0)  # (worker_k,)

            valid_slots = np.arange(n_slots, dtype=np.int64)
            masked_q = torch.full((self.worker_k,), float("-inf"))
            masked_q[valid_slots] = q_slots[valid_slots]
            slot = int(masked_q.argmax().item())
            slot = min(slot, n_slots - 1)

        item_pool_idx = int(top_k_candidates[slot])

        # Track manager state for later update
        if self._manager_state is None:
            self._manager_state = state.copy()
            self._manager_strategy = strategy_id

        return strategy_id, item_pool_idx, top_k_candidates

    def _get_top_k_candidates(self, strategy_id, valid_action_indices, mf_model, p_cold):
        """Map valid pool indices → item_ids → apply strategy → return top-K pool indices."""
        meta = self.item_metadata
        action_pool = meta.get("action_pool", None)
        if action_pool is None:
            return valid_action_indices[: self.worker_k]

        valid_item_ids = np.array([action_pool[i] for i in valid_action_indices], dtype=np.int64)
        shown_set = set()  # already excluded via valid_actions

        ranked_item_ids = apply_strategy(
            strategy_id, mf_model, p_cold,
            valid_item_ids, shown_set, meta,
            top_k=self.worker_k,
        )

        # Map back to pool indices
        item_id_to_pool_idx = {int(action_pool[i]): i for i in valid_action_indices}
        top_k_pool_indices = np.array(
            [item_id_to_pool_idx[int(iid)] for iid in ranked_item_ids
             if int(iid) in item_id_to_pool_idx],
            dtype=np.int64,
        )
        if len(top_k_pool_indices) == 0:
            top_k_pool_indices = valid_action_indices[: self.worker_k]
        return top_k_pool_indices

    def store_worker_transition(self, state, slot, reward, next_state, done, strategy_id):
        self.worker_buffer.push(state, slot, reward, next_state, done)
        self._manager_cumulative_reward += reward
        self._worker_step_count += 1

    def store_manager_transition(self, state, strategy_id, cumulative_reward, next_state, done):
        self.manager_buffer.push(state, strategy_id, cumulative_reward, next_state, done)
        self._manager_state = None
        self._manager_strategy = None
        self._manager_cumulative_reward = 0.0

    def should_update_manager(self):
        return self._worker_step_count % self.manager_update_freq == 0

    def update_worker(self, batch):
        return self._update_net(
            self.worker_net, self.worker_target, self.worker_optimizer,
            batch, self.worker_k
        )

    def update_manager(self, batch):
        return self._update_net(
            self.manager_net, self.manager_target, self.manager_optimizer,
            batch, self.n_strategies
        )

    def _update_net(self, online_net, target_net, optimizer, batch, action_dim):
        states, actions, rewards, next_states, dones, weights, indices = batch
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)
        weights_t = torch.FloatTensor(weights)

        online_net.train()
        q_current = online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            a_star = online_net(next_states_t).argmax(dim=1)
            q_next = target_net(next_states_t).gather(1, a_star.unsqueeze(1)).squeeze(1)
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        td_errors = (q_target - q_current).detach().cpu().numpy()
        loss = (weights_t * F.huber_loss(q_current, q_target, reduction="none")).mean()

        optimizer.zero_grad()
        loss.backward()
        import torch.nn as nn
        nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
        optimizer.step()

        # Soft target update
        for p, tp in zip(online_net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return float(loss.item()), td_errors

    def flush_manager_transition(self, next_state, done):
        """Store accumulated manager transition at episode end or forced flush."""
        if self._manager_state is not None and self._manager_strategy is not None:
            self.store_manager_transition(
                self._manager_state,
                self._manager_strategy,
                self._manager_cumulative_reward,
                next_state,
                done,
            )

    def save(self, path):
        torch.save({
            "manager_net": self.manager_net.state_dict(),
            "manager_target": self.manager_target.state_dict(),
            "worker_net": self.worker_net.state_dict(),
            "worker_target": self.worker_target.state_dict(),
            "worker_step_count": self._worker_step_count,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.manager_net.load_state_dict(ckpt["manager_net"])
        self.manager_target.load_state_dict(ckpt["manager_target"])
        self.worker_net.load_state_dict(ckpt["worker_net"])
        self.worker_target.load_state_dict(ckpt["worker_target"])
        self._worker_step_count = ckpt.get("worker_step_count", 0)
