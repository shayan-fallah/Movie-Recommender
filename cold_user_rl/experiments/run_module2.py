"""Module 1 + Module 2: Hybrid feedback + Personalized AL state."""
import copy
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from experiments._common import setup_experiment

config = copy.deepcopy(CONFIG)
config["use_hybrid_feedback"] = True
config["use_personalized_al"] = True
config["use_recurrent_dqn"] = False
config["use_hierarchical_rl"] = False

if __name__ == "__main__":
    setup_experiment(config, "module2_personalized_al")
