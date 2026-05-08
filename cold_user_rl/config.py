import os

# All paths are anchored to this file so they are correct regardless of the
# current working directory (important for Google Colab and IDE runners).
_HERE = os.path.dirname(os.path.abspath(__file__))

CONFIG = {

    # ─── DATASET ───────────────────────────────────────────────────────────────
    "dataset": "movielens_32m",
    "data_path": os.path.join(_HERE, "data", "ml-32m"),
    "min_user_interactions": 20,
    "cold_user_fraction": 0.25,
    "min_cold_user_interactions": 15,
    "interview_sizes": [10, 25, 50, 100],
    "train_val_test_split": [0.7, 0.1, 0.2],
    "random_seed": 42,

    # ─── DATASET SUBSAMPLING ───────────────────────────────────────────────────
    "use_subset": True,
    "min_user_ratings": 100,
    "min_item_ratings": 200,
    "kcore_iterations": 10,
    "min_year": 2000,
    "max_users": 30000,
    "max_items": 10000,
    "subset_random_seed": 42,

    # ─── HYBRID FEEDBACK (MODULE 1) ────────────────────────────────────────────
    "use_hybrid_feedback": True,
    "explicit_weight": 0.7,
    "implicit_weight": 0.3,
    "implicit_threshold": 3.5,
    "tag_confidence_bonus": 0.2,
    "normalize_ratings": True,

    # ─── MATRIX FACTORIZATION ──────────────────────────────────────────────────
    "mf_latent_features": 10,
    "mf_learning_rate": 0.001,
    "mf_regularization": 0.01,
    "mf_iterations": 100,
    "freeze_warm_mf": True,
    "cold_vector_lr": 0.001,
    "cold_vector_steps": 200,
    "cold_init_top_k_similar": 5,

    # ─── RL / DQN ──────────────────────────────────────────────────────────────
    "rl_approach": "item_based",
    "action_pool_size": 200,
    "num_episodes": 2000,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.00046,
    "learning_rate_dqn": 0.0004,
    "batch_size": 32,
    "buffer_size": 100,
    "target_update_steps": 100,
    "steps_before_training": 50,
    "hidden_layer_1": 64,
    "hidden_layer_2": 32,
    "reward_clip_max": 10.0,

    # ─── MODULE 2: PERSONALIZED AL ─────────────────────────────────────────────
    "use_personalized_al": False,
    "al_strategies": [
        "popularity", "entropy", "gini", "popent", "popgini",
        "error", "poperror", "variance", "popvar"
    ],
    "uncertainty_top_k": 20,
    "diversity_weight": 0.5,
    "state_include_cu_vector": True,
    "state_include_uncertainty": True,
    "state_include_diversity": True,

    # ─── MODULE 3: RECURRENT DQN (GRU) ────────────────────────────────────────
    "use_recurrent_dqn": False,
    "rnn_type": "GRU",
    "rnn_hidden_size": 64,
    "rnn_num_layers": 1,
    "sequence_length": 10,

    # ─── MODULE 4: HIERARCHICAL RL ─────────────────────────────────────────────
    "use_hierarchical_rl": False,
    "manager_action_space": 9,
    "worker_candidate_k": 10,
    "manager_reward_weight": 0.4,
    "worker_reward_weight": 0.6,
    "manager_update_frequency": 5,

    # ─── EVALUATION ────────────────────────────────────────────────────────────
    "eval_k_values": [5, 10, 20],
    "num_eval_episodes": 200,
    "eval_every_n_episodes": 500,
    "checkpoint_dir": os.path.join(_HERE, "checkpoints"),
    "log_dir": os.path.join(_HERE, "logs"),

    # ─── QUICK TEST (smoke testing without full dataset) ───────────────────────
    "quick_test": False,
    "quick_test_episodes": 100,
    "quick_test_users": 20,
}
