"""
Colab entry-point script.

Run from the cold_user_rl/ directory:

    %cd /content/Movie-Recommender/cold_user_rl
    !python colab_run.py --experiment base

Available experiments: base, module1, module2, module3, module4, full

Optional flags:
    --quick_test        Use small episode/user counts for smoke testing
    --data_path PATH    Override data directory (default: ./data/ml-32m/)
    --download          Download MovieLens 32M before running
    --preprocess_only   Only run preprocessing then exit
"""
import argparse
import copy
import os
import sys

# Make sure imports work regardless of how the script is invoked
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from config import CONFIG


def make_config(experiment, data_path=None, quick_test=False):
    cfg = copy.deepcopy(CONFIG)

    if data_path:
        cfg["data_path"] = data_path

    if quick_test:
        cfg["quick_test"] = True
        cfg["num_episodes"] = cfg["quick_test_episodes"]
        cfg["mf_iterations"] = 5
        cfg["eval_every_n_episodes"] = 20

    # Set module flags per experiment
    flags = {
        "base":    dict(use_hybrid_feedback=False, use_personalized_al=False,
                        use_recurrent_dqn=False, use_hierarchical_rl=False),
        "module1": dict(use_hybrid_feedback=True,  use_personalized_al=False,
                        use_recurrent_dqn=False, use_hierarchical_rl=False),
        "module2": dict(use_hybrid_feedback=True,  use_personalized_al=True,
                        use_recurrent_dqn=False, use_hierarchical_rl=False),
        "module3": dict(use_hybrid_feedback=True,  use_personalized_al=False,
                        use_recurrent_dqn=True,  use_hierarchical_rl=False),
        "module4": dict(use_hybrid_feedback=True,  use_personalized_al=False,
                        use_recurrent_dqn=False, use_hierarchical_rl=True),
        "full":    dict(use_hybrid_feedback=True,  use_personalized_al=True,
                        use_recurrent_dqn=False, use_hierarchical_rl=True),
    }
    cfg.update(flags[experiment])
    return cfg


def download_data(data_path):
    from data.download_movielens import download_movielens_32m, verify_files
    ml_dir = os.path.join(data_path, "ml-32m")
    if verify_files(ml_dir):
        print(f"Dataset already present at {ml_dir}")
    else:
        download_movielens_32m(data_path)


def preprocess(cfg):
    from data.preprocess import is_processed, run_pipeline
    processed_dir = os.path.join(cfg["data_path"], "processed")
    if is_processed(processed_dir):
        print(f"Processed data already exists at {processed_dir} — skipping.")
    else:
        print("Running preprocessing pipeline ...")
        run_pipeline(cfg)
        print("Preprocessing complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="base",
                        choices=["base", "module1", "module2", "module3", "module4", "full"])
    parser.add_argument("--data_path", default=None,
                        help="Path to the ml-32m/ parent directory")
    parser.add_argument("--download", action="store_true",
                        help="Download MovieLens 32M before running")
    parser.add_argument("--preprocess_only", action="store_true",
                        help="Only run preprocessing, then exit")
    parser.add_argument("--quick_test", action="store_true",
                        help="Short run for smoke testing (100 episodes, 20 users)")
    args = parser.parse_args()

    cfg = make_config(args.experiment, args.data_path, args.quick_test)

    if args.download:
        download_data(cfg["data_path"])

    preprocess(cfg)

    if args.preprocess_only:
        print("Preprocessing done. Exiting (--preprocess_only).")
        return

    from experiments._common import setup_experiment
    exp_name = args.experiment + ("_quick" if args.quick_test else "")
    setup_experiment(cfg, exp_name)


if __name__ == "__main__":
    main()
