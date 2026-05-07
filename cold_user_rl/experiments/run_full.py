"""Full system: all four modules ON."""
import copy
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG
from experiments._common import setup_experiment

config = copy.deepcopy(CONFIG)
config["use_hybrid_feedback"] = True
config["use_personalized_al"] = True
config["use_recurrent_dqn"] = False   # DRQN and Hierarchical RL are mutually exclusive
config["use_hierarchical_rl"] = True

if __name__ == "__main__":
    setup_experiment(config, "full_system")
