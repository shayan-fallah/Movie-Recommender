"""Reproduce base paper — all modules OFF."""
import copy
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from experiments._common import setup_experiment

# Override ALL module flags to False (base paper behavior)
config = copy.deepcopy(CONFIG)
config["use_hybrid_feedback"] = False
config["use_personalized_al"] = False
config["use_recurrent_dqn"] = False
config["use_hierarchical_rl"] = False

if __name__ == "__main__":
    setup_experiment(config, "base")
