#!/usr/bin/env python3
"""Sample a subset of trajectory data for testing."""
import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Sample trajectory data")
    parser.add_argument("--src", type=Path, required=True, help="Source directory")
    parser.add_argument("--dst", type=Path, required=True, help="Destination directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-per-env", type=int, default=10, help="Max trajectories per environment")
    args = parser.parse_args()

    import shutil

    rng = random.Random(args.seed)
    args.dst.mkdir(parents=True, exist_ok=True)

    for env_dir in args.src.iterdir():
        if not env_dir.is_dir():
            continue

        dst_env = args.dst / env_dir.name
        dst_env.mkdir(exist_ok=True)

        # Find all trajectory subdirectories
        traj_dirs = sorted([d for d in env_dir.iterdir() if d.is_dir()])
        selected = rng.sample(traj_dirs, min(args.max_per_env, len(traj_dirs)))

        for traj_dir in selected:
            shutil.copytree(traj_dir, dst_env / traj_dir.name, dirs_exist_ok=True)

    print(f"Sampled data written to {args.dst}")


if __name__ == "__main__":
    main()
