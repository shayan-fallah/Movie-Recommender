import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(log_csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(log_csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "reward" in df.columns:
        axes[0].plot(df["episode"], df["reward"], alpha=0.4, label="raw")
        if len(df) >= 50:
            axes[0].plot(
                df["episode"],
                df["reward"].rolling(50).mean(),
                label="MA-50",
                linewidth=2,
            )
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].set_title("Training Reward")
        axes[0].legend()

    if "rmse" in df.columns:
        axes[1].plot(df["episode"], df["rmse"], alpha=0.4, label="raw")
        if len(df) >= 50:
            axes[1].plot(
                df["episode"],
                df["rmse"].rolling(50).mean(),
                label="MA-50",
                linewidth=2,
            )
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("RMSE")
        axes[1].set_title("Cold-User RMSE")
        axes[1].legend()

    plt.tight_layout()
    out = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def compare_modules(results_dict, metric, output_dir):
    """Overlaid evaluation curves for multiple experiment configs.

    Args:
        results_dict: {experiment_name: pd.DataFrame with columns [episode, <metric>]}
        metric: column name to plot
        output_dir: where to save the figure
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, df in results_dict.items():
        if metric in df.columns and "episode" in df.columns:
            ax.plot(df["episode"], df[metric], label=name)
    ax.set_xlabel("Episode")
    ax.set_ylabel(metric)
    ax.set_title(f"Module Comparison — {metric}")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(output_dir, f"compare_{metric}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_strategy_distribution(strategy_log, output_dir, strategy_names=None):
    """Bar chart of manager strategy selections (Module 4).

    Args:
        strategy_log: list of strategy_id integers
        output_dir: where to save the figure
        strategy_names: optional list of string labels for each strategy ID
    """
    os.makedirs(output_dir, exist_ok=True)
    strategy_log = np.array(strategy_log)
    n_strategies = int(strategy_log.max()) + 1
    counts = np.bincount(strategy_log, minlength=n_strategies)

    labels = strategy_names if strategy_names else [str(i) for i in range(n_strategies)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, counts)
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Selection count")
    ax.set_title("Manager Strategy Distribution")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = os.path.join(output_dir, "strategy_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_interview_trajectories(trajectories, n_examples, output_dir):
    """Visualises RMSE trajectory over the interview for a few cold users.

    Args:
        trajectories: list of lists; each inner list is [rmse_step0, rmse_step1, ...]
        n_examples: how many to plot
        output_dir: where to save
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, traj in enumerate(trajectories[:n_examples]):
        ax.plot(traj, alpha=0.6, label=f"User {i}")
    ax.set_xlabel("Interview step")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Trajectory During Interview")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(output_dir, "interview_trajectories.png")
    plt.savefig(out, dpi=150)
    plt.close()
    return out
