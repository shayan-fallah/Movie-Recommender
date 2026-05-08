import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """Q-network: state → Q-values for all actions.

    Architecture matches the base paper:
    Linear(state_dim, 64) → tanh → Linear(64, 32) → tanh → Linear(32, action_dim)
    """

    def __init__(self, state_dim, action_dim, hidden1=64, hidden2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, action_dim),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        return self.net(state)


class DQNAgent:
    """Standard Double DQN with PER and soft target updates.

    Used when use_recurrent_dqn=False AND use_hierarchical_rl=False.
    """

    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.gamma = config["gamma"]
        h1 = config.get("hidden_layer_1", 64)
        h2 = config.get("hidden_layer_2", 32)

        self.online_net = DQNNetwork(state_dim, action_dim, h1, h2)
        self.target_net = DQNNetwork(state_dim, action_dim, h1, h2)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=config["learning_rate_dqn"]
        )
        self.tau = config.get("tau", 0.005)
        self._train_steps = 0

    def select_action(self, state, valid_actions, epsilon):
        """Epsilon-greedy action selection with action masking.

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

        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t).squeeze(0)  # (action_dim,)

        # Mask invalid actions
        masked_q = torch.full((self.action_dim,), float("-inf"))
        masked_q[valid_actions] = q_values[valid_actions]
        return int(masked_q.argmax().item())

    def compute_td_loss(self, batch):
        """Double DQN loss.

        Args:
            batch: (states, actions, rewards, next_states, dones, weights, indices)

        Returns:
            (loss tensor, td_errors numpy array)
        """
        states, actions, rewards, next_states, dones, weights, indices = batch

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)
        weights_t = torch.FloatTensor(weights)

        # Current Q(s, a)
        q_current = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target: a* from online, value from target
        with torch.no_grad():
            a_star = self.online_net(next_states_t).argmax(dim=1)
            q_next = self.target_net(next_states_t).gather(1, a_star.unsqueeze(1)).squeeze(1)
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        td_errors = (q_target - q_current).detach().cpu().numpy()
        loss = (weights_t * F.huber_loss(q_current, q_target, reduction="none")).mean()
        return loss, td_errors

    def update(self, batch):
        """One gradient update step.

        Returns:
            (float loss, td_errors numpy array)
        """
        loss, td_errors = self.compute_td_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
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
