import argparse
import os
import subprocess
import sys


SCRIPT_MAP = {
    "pretrain_sgl": "pretrain.py",
    "pretrain_lightgcn": "pretrain_lightgcn.py",
    "pretrain_simgcl": "pretrain_simgcl.py",
    "unlearn": "unlearn.py",
    "finetune": "fineTune.py",
    "finetune_drop": "fineTune_drop.py",
    "main_drop": "main_drop.py",
    "test_simgcl": "test_simgcl.py",
    "make_noise_dataset": "make_noise_dataset.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UnlearnRec scripts from one unified entry point."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=sorted(SCRIPT_MAP.keys()),
        help="Which training/evaluation script to execute.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the target script. Use '--' before forwarded args.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(repo_root, SCRIPT_MAP[args.stage])

    forwarded = list(args.script_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    cmd = [sys.executable, script_path] + forwarded
    print("[run_experiment] Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=repo_root)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
