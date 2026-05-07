#!/usr/bin/env python3
"""Config for GridWM-Judge."""

from pathlib import Path

_repo_root = Path(__file__).resolve().parent

# MiniGrid root directory (used by split/composite scripts)
MINIGRID_ROOT = _repo_root / "outputs"
# Temporary files directory
TEMP_ROOT = _repo_root / "scripts" / "02_trajectories" / "tmp"
# Repository root directory
REPO_ROOT = _repo_root
