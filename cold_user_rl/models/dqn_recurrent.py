import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DRQNNetwork(nn.Module):
    """GRU-based recurrent Q-network (Module 3).

    Architecture:
        Linear(state_dim, rnn_hidden_size) → GRU → Linear(rnn_hidden_size, 32) → tanh → Linear(32, action_dim)

    Why GRU over LSTM: interviews are ≤100 steps; GRU has fewer parameters and
    matches LSTM performance on short sequences.
    """

    def __init__(self, state_dim, action_dim, rnn_hidden_size, rnn_num_layers=1):
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        self.input_proj = nn.Linear(state_dim, rnn_hidden_size)
        self.gru = nn.GRU(
            rnn_hidden_size, rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
        )

        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.input_proj.weight)

    def forward(self, state_seq, hidden=None):
        """Full sequence forward pass.

        Args:
            state_seq: (batch, seq_len, state_dim)
            hidden: (num_layers, batch, rnn_hidden_size) or None

        Returns:
            q_values: (batch, seq_len, action_dim)
            hidden_out: (num_layers, batch, rnn_hidden_size)
        """
        x = F.relu(self.input_proj(state_seq))  # (batch, seq, rnn_hidden)
        gru_out, hidden_out = self.gru(x, hidden)  # (batch, seq, rnn_hidden)
        q_values = self.head(gru_out)              # (batch, seq, action_dim)
        return q_values, hidden_out

    def forward_single(self, state, hidden):
        """Single-step inference.

        Args:
            state: (state_dim,) or (1, state_dim)
            hidden: (num_layers, 1, rnn_hidden_size)

        Returns:
            q_values: (action_dim,)
            hidden_out: (num_layers, 1, rnn_hidden_size)
        """
        state = torch.FloatTensor(state).view(1, 1, -1)  # (1, 1, state_dim)
        q_seq, hidden_out = self.forward(state, hidden)
        return q_seq.squeeze(0).squeeze(0), hidden_out  # (action_dim,), hidden


class DRQNAgent:
    """Deep Recurrent Q-Network with episode-level replay.

    Used when config['use_recurrent_dqn'] = True (Module 3).

    Key design:
    - Inference hidden state (self.current_hidden) persists across steps within
      one episode and is reset to zeros at episode start.
    - Training hidden state is always initialized to zeros — episodes are
      replayed from scratch; inference and training hiddens are never shared.
    """

    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.gamma = config["gamma"]
        self.rnn_hidden_size = config.get("rnn_hidden_size", 64)
        self.rnn_num_layers = config.get("rnn_num_layers", 1)
        self.tau = config.get("tau", 0.005)

        self.online_net = DRQNNetwork(state_dim, action_dim, self.rnn_hidden_size, self.rnn_num_layers)
        self.target_net = DRQNNetwork(state_dim, action_dim, self.rnn_hidden_size, self.rnn_num_layers)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=config["learning_rate_dqn"]
        )

        self.current_hidden = None
        self.reset_hidden()

        self._train_steps = 0

    def reset_hidden(self):
        """Reset hidden state to zeros. Called at the start of each episode."""
        self.current_hidden = torch.zeros(
            self.rnn_num_layers, 1, self.rnn_hidden_size
        )

    def select_action(self, state, valid_actions, epsilon):
        """Epsilon-greedy with GRU hidden state carried over from previous step.

        Args:
            state: (state_dim,) numpy array
            valid_actions: (N,) numpy array of valid action indices
            epsilon: exploration probability

        Returns:
            int — selected action index
        """
        if len(valid_actions) == 0:
            return 0

        if np.random.random() < epsilon:
            return int(np.random.choice(valid_actions))

        self.online_net.eval()
        with torch.no_grad():
            q_values, self.current_hidden = self.online_net.forward_single(
                state, self.current_hidden
            )

        # Mask invalid actions
        masked_q = torch.full((self.action_dim,), float("-inf"))
        masked_q[valid_actions] = q_values[valid_actions]
        return int(masked_q.argmax().item())

    def compute_td_loss(self, episode_batch):
        """DRQN loss on a batch of complete episodes.

        GRU is initialized with hidden=None (zeros) for all episodes in the batch.
        This is correct: training always replays episodes from scratch.

        Args:
            episode_batch: tuple from EpisodeReplayBuffer.sample_episodes()
                (state_seqs, action_seqs, reward_seqs, next_state_seqs,
                 done_seqs, weights, indices, pad_masks)

        Returns:
            (loss tensor, episode-level mean |td_error| per episode)
        """
        (state_seqs, action_seqs, reward_seqs,
         next_state_seqs, done_seqs, weights, indices, pad_masks) = episode_batch

        B, T, _ = state_seqs.shape

        states_t = torch.FloatTensor(state_seqs)       # (B, T, state_dim)
        next_states_t = torch.FloatTensor(next_state_seqs)
        actions_t = torch.LongTensor(action_seqs)       # (B, T)
        rewards_t = torch.FloatTensor(reward_seqs)
        dones_t = torch.FloatTensor(done_seqs)
        weights_t = torch.FloatTensor(weights)          # (B,)
        masks_t = torch.FloatTensor(pad_masks)          # (B, T)

        self.online_net.train()

        # Q(s, a) via online net; hidden=None → zeros initialization
        q_seq, _ = self.online_net(states_t, None)      # (B, T, action_dim)
        q_current = q_seq.gather(2, actions_t.unsqueeze(2)).squeeze(2)  # (B, T)

        # Double DQN: a* from online, value from target
        with torch.no_grad():
            q_online_next, _ = self.online_net(next_states_t, None)
            a_star = q_online_next.argmax(dim=2)  # (B, T)
            q_target_next, _ = self.target_net(next_states_t, None)
            q_next = q_target_next.gather(2, a_star.unsqueeze(2)).squeeze(2)
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        td_errors = (q_target - q_current).detach()  # (B, T)

        # Huber loss, masked to valid steps only
        huber = F.huber_loss(q_current, q_target, reduction="none")  # (B, T)
        masked_huber = (huber * masks_t).sum(dim=1) / (masks_t.sum(dim=1) + 1e-8)  # (B,)
        loss = (weights_t * masked_huber).mean()

        # Per-episode mean |td_error| for PER priority update
        ep_td_errors = ((td_errors.abs() * masks_t).sum(dim=1) / (masks_t.sum(dim=1) + 1e-8))
        return loss, ep_td_errors.cpu().numpy()

    def update(self, episode_batch):
        """One gradient update step.

        Returns:
            (float loss, per-episode td_errors)
        """
        loss, td_errors = self.compute_td_loss(episode_batch)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._soft_update_target()
        self._train_steps += 1

        return float(loss.item()), td_errors

    def _soft_update_target(self):
        for param, target_param in zip(
            self.online_net.parameters(), self.target_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path):
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self._train_steps,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._train_steps = ckpt.get("train_steps", 0)
