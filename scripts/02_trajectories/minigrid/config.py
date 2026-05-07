#!/usr/bin/env python3
"""Config for GridWM-Judge minigrid scripts."""

from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent.parent

# MiniGrid root directory
MINIGRID_ROOT = _repo_root / "outputs"
# Temporary files directory
TEMP_ROOT = _repo_root / "scripts" / "02_trajectories" / "tmp"
