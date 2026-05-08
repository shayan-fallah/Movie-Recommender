"""
Colab entry-point script.

Run from the cold_user_rl/ directory:

    %cd /content/Movie-Recommender/cold_user_rl
    !python colab_run.py --experiment base

Available experiments: base, module1, module2, module3, module4, full

Optional flags:
    --quick_test        Use small episode/user counts for smoke testing
    --data_path PATH    Override data directory (absolute path to ml-32m/)
    --download          Download MovieLens 32M before running
    --preprocess_only   Only run preprocessing then exit
"""
import argparse
import copy
import os
import shutil
import sys

# Make sure imports work regardless of how the script is invoked
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from config import CONFIG


def make_config(experiment, data_path=None, quick_test=False):
    cfg = copy.deepcopy(CONFIG)

    if data_path:
        cfg["data_path"] = os.path.abspath(data_path)

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

    data_path = os.path.abspath(data_path)

    # ── Case 1: already at the correct location ────────────────────────────────
    if verify_files(data_path):
        print(f"Dataset already present at {data_path}")
        return

    # ── Case 2: double-nested layout from old bug (data_path/ml-32m/ratings.csv)
    nested = os.path.join(data_path, "ml-32m")
    if verify_files(nested):
        print(f"Found data nested at {nested} — moving files up to {data_path} ...")
        os.makedirs(data_path, exist_ok=True)
        for fname in os.listdir(nested):
            shutil.move(os.path.join(nested, fname), os.path.join(data_path, fname))
        try:
            os.rmdir(nested)
        except OSError:
            pass  # non-empty remnants are harmless
        print(f"Files moved. Dataset ready at {data_path}")
        return

    # ── Case 3: not downloaded yet — extract to parent so ml-32m/ lands correctly
    parent_dir = os.path.dirname(data_path)
    os.makedirs(parent_dir, exist_ok=True)
    download_movielens_32m(parent_dir)


def preprocess(cfg, force=False):
    from data.preprocess import is_processed, run_pipeline
    processed_dir = os.path.join(cfg["data_path"], "processed")
    if force:
        # Remove the sentinel so run_pipeline executes unconditionally
        sentinel = os.path.join(processed_dir, "dataset_stats.json")
        if os.path.isfile(sentinel):
            os.remove(sentinel)
            print(f"Removed stale sentinel: {sentinel}")
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
                        help="Absolute path to the ml-32m/ directory (overrides config default)")
    parser.add_argument("--download", action="store_true",
                        help="Download MovieLens 32M before running")
    parser.add_argument("--preprocess_only", action="store_true",
                        help="Only run preprocessing, then exit")
    parser.add_argument("--force_preprocess", action="store_true",
                        help="Force re-preprocessing even if processed data already exists "
                             "(deletes the sentinel file and reruns the full pipeline)")
    parser.add_argument("--quick_test", action="store_true",
                        help="Short run for smoke testing (100 episodes, 20 users)")
    args = parser.parse_args()

    cfg = make_config(args.experiment, args.data_path, args.quick_test)

    if args.download:
        download_data(cfg["data_path"])

    preprocess(cfg, force=args.force_preprocess)

    if args.preprocess_only:
        print("Preprocessing done. Exiting (--preprocess_only).")
        return

    from experiments._common import setup_experiment
    exp_name = args.experiment + ("_quick" if args.quick_test else "")
    setup_experiment(cfg, exp_name)


if __name__ == "__main__":
    main()
