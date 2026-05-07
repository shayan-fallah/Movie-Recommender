"""Module 1 + Module 4: Hybrid feedback + Hierarchical RL."""
import copy
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from experiments._common import setup_experiment

config = copy.deepcopy(CONFIG)
config["use_hybrid_feedback"] = True
config["use_personalized_al"] = False
config["use_recurrent_dqn"] = False
config["use_hierarchical_rl"] = True

if __name__ == "__main__":
    setup_experiment(config, "module4_hierarchical_rl")
