import os

import numpy as np
import torch
from tqdm import trange

from active_learning.personalized_al import PersonalizedALSelector
from environment.cold_user_env import ColdUserEnv
from environment.cold_user_env_extended import ColdUserEnvExtended
from models.dqn_base import DQNAgent
from models.dqn_recurrent import DRQNAgent
from models.hierarchical_rl import HierarchicalRLAgent
from training.replay_buffer import PrioritizedReplayBuffer, UnifiedReplayBuffer


class RunningNormalizer:
    """Normalizes values using a running window mean and std.

    Keeps the last `window` values and z-scores the incoming value.
    Returns the raw value unchanged until at least 10 samples are collected.
    """

    def __init__(self, window=500):
        self.values = []
        self.window = window

    def normalize(self, value):
        self.values.append(value)
        if len(self.values) > self.window:
            self.values.pop(0)
        if len(self.values) < 10:
            return value
        mean = np.mean(self.values)
        std = np.std(self.values) + 1e-8
        return (value - mean) / std


def _make_per_buffer(config):
    """Create a PrioritizedReplayBuffer with standard PER hyperparameters."""
    capacity = config["buffer_size"]
    return PrioritizedReplayBuffer(
        capacity=capacity,
        alpha=0.6,
        beta_start=0.4,
        beta_end=1.0,
        beta_anneal_steps=capacity,
        per_epsilon=1e-6,
    )


def build_env(mf_model, finetuner, feedback_bundle, cold_split,
              action_pool, item_metadata, config):
    """Instantiate the correct environment based on config flags.

    Flag branching is centralised here — environment classes are flag-agnostic.
    """
    use_al = config.get("use_personalized_al", False)
    use_hier = config.get("use_hierarchical_rl", False)

    if use_al or use_hier:
        al_selector = None
        if use_al:
            al_selector = PersonalizedALSelector(mf_model, config, item_metadata)
        env = ColdUserEnvExtended(
            mf_model, finetuner, feedback_bundle, cold_split,
            action_pool, item_metadata, config, al_selector=al_selector
        )
    else:
        env = ColdUserEnv(
            mf_model, finetuner, feedback_bundle, cold_split,
            action_pool, item_metadata, config
        )
    return env


def build_agent(state_dim, action_dim, config):
    """Instantiate the correct agent based on config flags.

    Exactly one of the three module flags should be True (or none for base DQN).
    """
    if config.get("use_recurrent_dqn", False):
        return DRQNAgent(state_dim, action_dim, config)
    elif config.get("use_hierarchical_rl", False):
        return HierarchicalRLAgent(state_dim, action_dim, config)
    else:
        return DQNAgent(state_dim, action_dim, config)


def compute_epsilon(step, config):
    """Linear epsilon decay from epsilon_start to epsilon_end."""
    decay = config.get("epsilon_decay", 0.00046)
    eps_start = config.get("epsilon_start", 1.0)
    eps_end = config.get("epsilon_end", 0.01)
    eps = eps_start - decay * step
    return float(max(eps_end, eps))


def normalize_state(state, running_mean, running_var, update=True):
    """Z-score normalization with running statistics."""
    if update:
        running_mean[:] = 0.99 * running_mean + 0.01 * state
        running_var[:] = 0.99 * running_var + 0.01 * (state - running_mean) ** 2
    std = np.sqrt(running_var + 1e-8)
    return (state - running_mean) / std


def train_rl(mf_model, feedback_bundle, cold_split, action_pool,
             item_metadata, config, logger, eval_cold_split=None):
    """Main RL training loop.

    Args:
        mf_model: trained MatrixFactorization (warm weights frozen)
        feedback_bundle: FeedbackBundle
        cold_split: dict {user_id: {interview_pool, test_set}} for training cold users
        action_pool: (action_pool_size,) array of item IDs
        item_metadata: dict for AL strategies
        config: CONFIG dict
        logger: ExperimentLogger
        eval_cold_split: optional separate cold split for periodic evaluation
    """
    from models.matrix_factorization import ColdUserFinetuner

    finetuner = ColdUserFinetuner(mf_model, config)
    env = build_env(mf_model, finetuner, feedback_bundle, cold_split,
                    action_pool, item_metadata, config)

    item_metadata["action_pool"] = action_pool

    state_dim = env.state_dim
    action_dim = len(action_pool)

    agent = build_agent(state_dim, action_dim, config)

    if isinstance(agent, HierarchicalRLAgent):
        agent.set_item_metadata(item_metadata)

    use_recurrent = config.get("use_recurrent_dqn", False)
    use_hier = config.get("use_hierarchical_rl", False)
    is_quick_test = config.get("quick_test", False)

    n_episodes = config.get("quick_test_episodes", 100) if is_quick_test \
        else config.get("num_episodes", 2000)

    batch_size = config.get("batch_size", 32)
    steps_before_train = config.get("steps_before_training", 50)
    eval_every = config.get("eval_every_n_episodes", 500)
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    interview_sizes = config.get("interview_sizes", [10])

    # ── Replay buffers ────────────────────────────────────────────────────────
    # Separate PER buffer per interview size for standard DQN so that
    # transitions from 10-item and 100-item interviews never mix gradients.
    # DRQN uses a single episode buffer (sequences are naturally per-size).
    # HierRL manages its own internal worker/manager buffers.
    if use_recurrent:
        buffer = UnifiedReplayBuffer(config)  # single episode buffer
        replay_buffers = None
    elif use_hier:
        buffer = None
        replay_buffers = None
    else:
        buffer = None
        replay_buffers = {s: _make_per_buffer(config) for s in interview_sizes}

    # One reward normalizer per interview size
    reward_normalizers = {s: RunningNormalizer(window=500) for s in interview_sizes}

    # Running statistics for state normalization
    running_mean = np.zeros(state_dim, dtype=np.float32)
    running_var = np.ones(state_dim, dtype=np.float32)

    global_step = 0
    best_eval_rmse = float("inf")

    for episode in trange(1, n_episodes + 1, desc="RL Training"):
        interview_size = interview_sizes[episode % len(interview_sizes)]
        env.set_interview_size(interview_size)

        if use_recurrent:
            agent.reset_hidden()

        state = env.reset()
        state_norm = normalize_state(state, running_mean, running_var)

        episode_rewards = []
        episode_transitions = []
        manager_reward_acc = 0.0
        td_error_sum = 0.0
        n_updates = 0

        done = False
        while not done:
            epsilon = compute_epsilon(global_step, config)
            valid_actions = env.get_valid_actions()

            if len(valid_actions) == 0:
                break

            # ── Action selection ──────────────────────────────────────────────
            if use_hier:
                strategy_id, item_pool_idx, top_k_cands = agent.select_action(
                    state_norm, valid_actions, epsilon, mf_model, env.p_cold
                )
                action = (strategy_id, item_pool_idx)
            else:
                action = agent.select_action(state_norm, valid_actions, epsilon)

            # ── Environment step ──────────────────────────────────────────────
            next_state, reward, done, info = env.step(action)
            next_state_norm = normalize_state(next_state, running_mean, running_var)

            episode_rewards.append(reward)

            # Normalize reward before storing (per interview size)
            norm_reward = reward_normalizers[interview_size].normalize(reward)

            # ── Storage ───────────────────────────────────────────────────────
            if use_recurrent:
                actual_action = int(action) if not isinstance(action, tuple) else action[1]
                episode_transitions.append(
                    (state_norm.copy(), actual_action, norm_reward,
                     next_state_norm.copy(), float(done))
                )
            elif use_hier:
                slot = top_k_cands.tolist().index(item_pool_idx) \
                    if item_pool_idx in top_k_cands else 0
                agent.store_worker_transition(
                    state_norm, slot, norm_reward, next_state_norm, done, strategy_id
                )
                manager_reward_acc += norm_reward
            else:
                current_buffer = replay_buffers[interview_size]
                current_buffer.push(state_norm, int(action), norm_reward,
                                    next_state_norm, float(done))

            # ── Training ──────────────────────────────────────────────────────
            if global_step >= steps_before_train:
                if use_recurrent:
                    pass  # DRQN trains at episode end
                elif use_hier:
                    if len(agent.worker_buffer) >= batch_size:
                        w_batch = agent.worker_buffer.sample(batch_size)
                        w_loss, w_td = agent.update_worker(w_batch)
                        agent.worker_buffer.update_priorities(w_batch[6], np.abs(w_td))
                        agent.worker_buffer.anneal_beta()
                        td_error_sum += float(np.abs(w_td).mean())
                        n_updates += 1

                    if agent.should_update_manager() and len(agent.manager_buffer) >= batch_size:
                        m_batch = agent.manager_buffer.sample(batch_size)
                        m_loss, m_td = agent.update_manager(m_batch)
                        agent.manager_buffer.update_priorities(m_batch[6], np.abs(m_td))
                        agent.manager_buffer.anneal_beta()
                else:
                    current_buffer = replay_buffers[interview_size]
                    if len(current_buffer) >= batch_size:
                        batch = current_buffer.sample(batch_size)
                        loss, td_errors = agent.update(batch)
                        current_buffer.update_priorities(batch[6], np.abs(td_errors))
                        current_buffer.anneal_beta()
                        td_error_sum += float(np.abs(td_errors).mean())
                        n_updates += 1

            # ── Hierarchical manager transition storage ───────────────────────
            if use_hier and agent.should_update_manager():
                agent.flush_manager_transition(next_state_norm, done)

            state = next_state
            state_norm = next_state_norm
            global_step += 1

        # ── Episode end ───────────────────────────────────────────────────────
        if use_recurrent and episode_transitions:
            buffer.push_episode(episode_transitions)
            if len(buffer) >= batch_size:
                ep_batch = buffer.sample_episodes(batch_size)
                loss, ep_td = agent.update(ep_batch)
                buffer.update_episode_priorities(ep_batch[6], np.abs(ep_td))
                buffer.anneal_beta()
                td_error_sum += float(np.abs(ep_td).mean())
                n_updates += 1

        if use_hier:
            agent.flush_manager_transition(state_norm, True)

        ep_rmse = info.get("rmse", float("nan"))
        ep_reward = sum(episode_rewards)
        mean_td_error = td_error_sum / max(n_updates, 1)

        logger.log_episode({
            "episode": episode,
            "reward": ep_reward,
            "rmse": ep_rmse,
            "interview_size": interview_size,
            "epsilon": compute_epsilon(global_step, config),
            "global_step": global_step,
            "mean_td_error": mean_td_error,
        })

        # ── Periodic evaluation ───────────────────────────────────────────────
        if episode % eval_every == 0 and eval_cold_split is not None:
            eval_results = _quick_eval(
                agent, env, mf_model, finetuner, eval_cold_split,
                action_pool, item_metadata, config, running_mean, running_var,
                n_users=config.get("num_eval_episodes", 200)
            )
            eval_results["episode"] = episode
            logger.log_eval(eval_results)

            if eval_results.get("mean_rmse", float("inf")) < best_eval_rmse:
                best_eval_rmse = eval_results["mean_rmse"]
                ckpt_path = os.path.join(checkpoint_dir, "best_agent.pt")
                agent.save(ckpt_path)
                logger.save_checkpoint_meta(ckpt_path)

    logger.info(f"RL training complete. Best eval RMSE: {best_eval_rmse:.4f}")
    return agent


def _quick_eval(agent, env, mf_model, finetuner, eval_cold_split,
                action_pool, item_metadata, config, running_mean, running_var,
                n_users=50):
    """Greedy evaluation on a subset of cold users."""
    use_recurrent = config.get("use_recurrent_dqn", False)
    use_hier = config.get("use_hierarchical_rl", False)
    is_quick = config.get("quick_test", False)

    eval_users = list(eval_cold_split.keys())
    if is_quick:
        n_users = min(n_users, config.get("quick_test_users", 20))
    n_users = min(n_users, len(eval_users))
    sampled_users = np.random.choice(eval_users, size=n_users, replace=False)

    # The training env only knows train_cold_split users; swap in eval_cold_split
    # so env.reset(user_id=uid) can find eval users, then restore afterwards.
    _orig_split = env.cold_split
    env.cold_split = eval_cold_split

    rmse_list = []
    for uid in sampled_users:
        if use_recurrent:
            agent.reset_hidden()
        state = env.reset(user_id=uid)
        state_norm = (state - running_mean) / (np.sqrt(running_var) + 1e-8)

        done = False
        while not done:
            valid_actions = env.get_valid_actions()
            if len(valid_actions) == 0:
                break
            if use_hier:
                _, item_pool_idx, _ = agent.select_action(
                    state_norm, valid_actions, 0.0, mf_model, env.p_cold
                )
                action = (0, item_pool_idx)
            else:
                action = agent.select_action(state_norm, valid_actions, 0.0)
            next_state, _, done, info = env.step(action)
            state_norm = (next_state - running_mean) / (np.sqrt(running_var) + 1e-8)

        rmse_list.append(info.get("rmse", float("nan")))

    env.cold_split = _orig_split  # restore training split

    rmse_arr = np.array([r for r in rmse_list if not np.isnan(r)])
    return {
        "mean_rmse": float(rmse_arr.mean()) if len(rmse_arr) > 0 else float("nan"),
        "std_rmse": float(rmse_arr.std()) if len(rmse_arr) > 0 else float("nan"),
        "n_eval_users": n_users,
    }
