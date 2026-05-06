#!/usr/bin/env python3
"""Config for GridWM-Judge minigrid scripts."""

from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent.parent

# MiniGrid 根目录
MINIGRID_ROOT = _repo_root / "outputs"
# 临时文件目录
TEMP_ROOT = _repo_root / "scripts" / "02_trajectories" / "tmp"
