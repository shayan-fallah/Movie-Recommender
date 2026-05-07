# Cold-User Recommender System with Hybrid RL + AL

## Overview

This repository implements a research-grade recommender system that addresses the **cold-user problem** — the challenge of providing personalised recommendations to new users who have no prior interaction history.

The system is based on:

> Giannikis, S., Frasincar, F., & Boekestijn, D. (2024). "Reinforcement learning for addressing the cold-user problem in recommender systems." *Knowledge-Based Systems*, 294, 111752.

It extends the base paper with four independently toggleable modules:

1. **Hybrid Feedback** — combines explicit ratings and implicit watched/not-watched signals
2. **Personalized Active Learning** — dynamically recomputes which items are most informative per cold user
3. **Recurrent DQN (GRU)** — the agent retains memory of the full interview trajectory
4. **Hierarchical RL** — a Manager picks a strategy; a Worker picks the specific item

## Architecture

```
Cold User Interview Loop
────────────────────────
        ┌─────────────────────────────────┐
        │  Matrix Factorization (MF)       │
        │  P (warm users, frozen)          │
        │  Q (items, frozen)               │
        │  p_cold (optimised per episode)  │
        └────────────────┬────────────────┘
                         │ RMSE reward
        ┌────────────────▼────────────────┐
        │  Deep Q-Network (DQN)            │
        │  state: shown_items [+ ext]      │
        │  action: which item to ask next  │
        │  reward: 1 / (RMSE + ε)          │
        └─────────────────────────────────┘
```

At each episode step the agent picks an item to show the cold user. The user's response is used to fine-tune their latent vector `p_cold` against the frozen item matrix `Q`. The RMSE on held-out test items defines the reward.

## Modules

| Module | Config Flag | Description | Default |
|--------|-------------|-------------|---------|
| Hybrid Feedback | `use_hybrid_feedback` | Explicit ratings + implicit (watched/not) with confidence weighting | `True` |
| Personalized AL | `use_personalized_al` | Dynamic uncertainty + diversity scores in state | `False` |
| Recurrent DQN | `use_recurrent_dqn` | GRU memory across interview steps (DRQN) | `False` |
| Hierarchical RL | `use_hierarchical_rl` | Manager selects strategy; Worker selects item | `False` |
| Freeze Warm MF | `freeze_warm_mf` | Only optimise cold-user latent vector per episode | `True` |

Setting any flag to `False` reproduces the corresponding base-paper behaviour exactly.

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download the dataset
python data/download_movielens.py ./data

# 2. Preprocess (creates data/ml-32m/processed/)
python data/preprocess.py

# 3. Train MF + RL (base paper — all modules OFF)
python experiments/run_base.py

# 4. Run with all modules active
python experiments/run_full.py
```

## Reproducing Results

| Script | Modules active |
|--------|---------------|
| `experiments/run_base.py` | None (base paper) |
| `experiments/run_module1.py` | Hybrid Feedback only |
| `experiments/run_module2.py` | + Personalized AL |
| `experiments/run_module3.py` | + Recurrent DQN |
| `experiments/run_module4.py` | + Hierarchical RL |
| `experiments/run_full.py` | All four modules |

Each script sets all flags explicitly before any imports and calls `set_all_seeds(CONFIG["random_seed"])` for full reproducibility.

## Configuration

All hyperparameters live in `config.py` as a plain Python dict `CONFIG`. Experiment scripts import this dict and override specific keys before importing any training modules.

Key parameters:

```python
CONFIG = {
    "mf_latent_features": 10,          # latent dimension k
    "freeze_warm_mf": True,             # KEY: only optimise cold user vector
    "cold_vector_steps": 50,            # gradient steps per episode step
    "interview_sizes": [10, 25, 50, 100],
    "action_pool_size": 200,            # top-N popular items as action space
    "num_episodes": 2000,
    "use_hybrid_feedback": True,
    "use_personalized_al": False,
    "use_recurrent_dqn": False,
    "use_hierarchical_rl": False,
    "quick_test": False,                # True for fast smoke testing
    ...
}
```

## Dataset

**MovieLens 32M** — 32 million ratings from 200,000 users on 87,000 movies.

- All users have rated at least 20 movies
- No demographic data (GDPR-compliant by design)
- Download: `python data/download_movielens.py`

### Hybrid Feedback Construction

Two parallel feedback signals are built from the same rating data:

**Explicit matrix E:** `E[u,i] = rating / 5.0` (NaN if not rated)

**Implicit matrix I:** `I[u,i] = 1` if `rating >= 3.5`, else `0` (NaN if not rated)

**Confidence matrix C:** base `1.0` + `tag_confidence_bonus` if the user also tagged the movie

**Loss:**
```
Loss = λ_e × MSE(pred, explicit)[observed]
     + λ_i × mean(C × (pred − implicit)²)[observed]
```

### Cold User Simulation

- 25% of users are randomly designated as cold users
- Their interactions are split into: `interview_pool` (items the agent may ask) and `test_set` (held-out for RMSE)
- `test_set` always contains at least 5 items

## Key Optimisation: Freeze Warm MF

The base paper retrains the entire MF model from scratch at each RL step. This is slow and unstable.

When `freeze_warm_mf=True` (default), only the cold user's latent vector `p_cold` is optimised — in 50 gradient steps against the frozen item matrix `Q`. This is ~100× faster and produces a more stable reward signal.

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.2.0
tensorboard>=2.13.0
tqdm>=4.65.0
matplotlib>=3.7.0
requests>=2.28.0
```
