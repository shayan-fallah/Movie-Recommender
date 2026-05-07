import csv
import logging
import os
from datetime import datetime


class ExperimentLogger:
    def __init__(self, config, experiment_name):
        self.config = config
        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(config["log_dir"], f"{experiment_name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, "run.log")),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(experiment_name)

        self._episode_csv_path = os.path.join(self.log_dir, "episodes.csv")
        self._eval_csv_path = os.path.join(self.log_dir, "evaluations.csv")
        self._episode_writer = None
        self._eval_writer = None
        self._episode_file = None
        self._eval_file = None

        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(log_dir=os.path.join(self.log_dir, "tb"))
        except ImportError:
            self._tb = None

    def log_scalar(self, tag, value, step):
        if self._tb is not None:
            self._tb.add_scalar(tag, value, step)

    def log_episode(self, episode_dict):
        if self._episode_writer is None:
            self._episode_file = open(self._episode_csv_path, "w", newline="")
            self._episode_writer = csv.DictWriter(
                self._episode_file, fieldnames=list(episode_dict.keys())
            )
            self._episode_writer.writeheader()
        self._episode_writer.writerow(episode_dict)
        self._episode_file.flush()

        ep = episode_dict.get("episode", "?")
        rmse = episode_dict.get("rmse", float("nan"))
        reward = episode_dict.get("reward", float("nan"))
        eps = episode_dict.get("epsilon", float("nan"))
        self.logger.info(
            f"Episode {ep} | RMSE {rmse:.4f} | Reward {reward:.4f} | Epsilon {eps:.4f}"
        )

        for k, v in episode_dict.items():
            if isinstance(v, (int, float)):
                self.log_scalar(f"episode/{k}", v, ep if isinstance(ep, int) else 0)

    def log_eval(self, eval_dict):
        if self._eval_writer is None:
            self._eval_file = open(self._eval_csv_path, "w", newline="")
            self._eval_writer = csv.DictWriter(
                self._eval_file, fieldnames=list(eval_dict.keys())
            )
            self._eval_writer.writeheader()
        self._eval_writer.writerow(eval_dict)
        self._eval_file.flush()

        step = eval_dict.get("episode", 0)
        mean_rmse = eval_dict.get("mean_rmse", float("nan"))
        self.logger.info(f"[EVAL] Episode {step} | Mean RMSE {mean_rmse:.4f}")

        for k, v in eval_dict.items():
            if isinstance(v, (int, float)):
                self.log_scalar(f"eval/{k}", v, step if isinstance(step, int) else 0)

    def save_checkpoint_meta(self, path):
        self.logger.info(f"Checkpoint saved: {path}")

    def info(self, msg):
        self.logger.info(msg)

    def close(self):
        if self._episode_file:
            self._episode_file.close()
        if self._eval_file:
            self._eval_file.close()
        if self._tb is not None:
            self._tb.close()
