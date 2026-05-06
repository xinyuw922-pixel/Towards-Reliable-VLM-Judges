#!/usr/bin/env python3
"""Config for GridWM-Judge."""

from pathlib import Path

_repo_root = Path(__file__).resolve().parent

# MiniGrid 根目录 (用于 split/composite 等脚本的默认路径)
MINIGRID_ROOT = _repo_root / "outputs"
# 临时文件目录
TEMP_ROOT = _repo_root / "scripts" / "02_trajectories" / "tmp"
# 仓库根目录
REPO_ROOT = _repo_root
