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
from training.replay_buffer import UnifiedReplayBuffer


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
    # Decay steps derived from epsilon_decay (rate per step)
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

    # Inject action_pool into item_metadata so HierRL can map pool-idx → item-id
    item_metadata["action_pool"] = action_pool
    if isinstance(env, ColdUserEnvExtended) and config.get("use_hierarchical_rl", False):
        # Inject metadata into Hierarchical agent after creation
        pass  # done below after agent creation

    state_dim = env.state_dim
    action_dim = len(action_pool)

    agent = build_agent(state_dim, action_dim, config)

    # Inject item_metadata into HierRL agent
    if isinstance(agent, HierarchicalRLAgent):
        agent.set_item_metadata(item_metadata)

    buffer = UnifiedReplayBuffer(config)

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

    # Running statistics for state normalization
    running_mean = np.zeros(state_dim, dtype=np.float32)
    running_var = np.ones(state_dim, dtype=np.float32)

    global_step = 0
    best_eval_rmse = float("inf")

    for episode in trange(1, n_episodes + 1, desc="RL Training"):
        # Pick interview size (cycle or use first)
        interview_size = interview_sizes[episode % len(interview_sizes)]
        env.set_interview_size(interview_size)

        # Reset agent hidden state (DRQN)
        if use_recurrent:
            agent.reset_hidden()

        state = env.reset()
        state_norm = normalize_state(state, running_mean, running_var)

        episode_rewards = []
        episode_transitions = []  # for DRQN episode buffer
        manager_reward_acc = 0.0

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
            elif use_recurrent:
                action = agent.select_action(state_norm, valid_actions, epsilon)
            else:
                action = agent.select_action(state_norm, valid_actions, epsilon)

            # ── Environment step ──────────────────────────────────────────────
            next_state, reward, done, info = env.step(action)
            next_state_norm = normalize_state(next_state, running_mean, running_var)

            episode_rewards.append(reward)

            # ── Storage ───────────────────────────────────────────────────────
            if use_recurrent:
                actual_action = int(action) if not isinstance(action, tuple) else action[1]
                episode_transitions.append(
                    (state_norm.copy(), actual_action, reward,
                     next_state_norm.copy(), float(done))
                )
            elif use_hier:
                slot = top_k_cands.tolist().index(item_pool_idx) \
                    if item_pool_idx in top_k_cands else 0
                agent.store_worker_transition(
                    state_norm, slot, reward, next_state_norm, done, strategy_id
                )
                manager_reward_acc += reward
            else:
                buffer.push(state_norm, int(action), reward, next_state_norm, float(done))

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

                    if agent.should_update_manager() and len(agent.manager_buffer) >= batch_size:
                        m_batch = agent.manager_buffer.sample(batch_size)
                        m_loss, m_td = agent.update_manager(m_batch)
                        agent.manager_buffer.update_priorities(m_batch[6], np.abs(m_td))
                        agent.manager_buffer.anneal_beta()
                else:
                    if len(buffer) >= batch_size:
                        batch = buffer.sample(batch_size)
                        loss, td_errors = agent.update(batch)
                        buffer.update_priorities(batch[6], np.abs(td_errors))
                        buffer.anneal_beta()

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

        if use_hier:
            agent.flush_manager_transition(state_norm, True)

        ep_rmse = info.get("rmse", float("nan"))
        ep_reward = sum(episode_rewards)

        logger.log_episode({
            "episode": episode,
            "reward": ep_reward,
            "rmse": ep_rmse,
            "interview_size": interview_size,
            "epsilon": compute_epsilon(global_step, config),
            "global_step": global_step,
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
                ckpt_path = os.path.join(checkpoint_dir, f"best_agent.pt")
                agent.save(ckpt_path)
                logger.save_checkpoint_meta(ckpt_path)

    logger.info(f"RL training complete. Best eval RMSE: {best_eval_rmse:.4f}")
    return agent


def _quick_eval(agent, env, mf_model, finetuner, eval_cold_split,
                action_pool, item_metadata, config, running_mean, running_var,
                n_users=50):
    """Greedy evaluation on a subset of cold users."""
    from models.matrix_factorization import ColdUserFinetuner
    use_recurrent = config.get("use_recurrent_dqn", False)
    use_hier = config.get("use_hierarchical_rl", False)
    is_quick = config.get("quick_test", False)

    eval_users = list(eval_cold_split.keys())
    if is_quick:
        n_users = min(n_users, config.get("quick_test_users", 20))
    n_users = min(n_users, len(eval_users))
    sampled_users = np.random.choice(eval_users, size=n_users, replace=False)

    rmse_list = []
    for uid in sampled_users:
        if use_recurrent:
            agent.reset_hidden()
        env.reset(user_id=uid)
        state = env._get_state()
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

    rmse_arr = np.array([r for r in rmse_list if not np.isnan(r)])
    return {
        "mean_rmse": float(rmse_arr.mean()) if len(rmse_arr) > 0 else float("nan"),
        "std_rmse": float(rmse_arr.std()) if len(rmse_arr) > 0 else float("nan"),
        "n_eval_users": n_users,
    }
