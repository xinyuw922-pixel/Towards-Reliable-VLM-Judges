#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GridWM-Judge central configuration file.

All path configurations should be managed through this file to avoid hardcoding.
Supports environment variable overrides for running on different machines.
"""

from pathlib import Path
import os

# ============================================================
# 1. Repository Root (auto-detected)
# ============================================================
REPO_ROOT = Path(__file__).parent.resolve()

# ============================================================
# 2. Data Directories
# ============================================================
# Default uses datasets directory within the repository
DATA_ROOT = Path(os.environ.get("GRIDWM_DATA_ROOT", REPO_ROOT / "datasets"))

# Canonical datasets
MINIGRID_ROOT = DATA_ROOT / "minigrid"
MINIWORLD_ROOT = DATA_ROOT / "miniworld"
RAW_DATA_ROOT = DATA_ROOT / "raw_data"
GLOBAL_VIEWS_ROOT = DATA_ROOT / "global_views"

# ============================================================
# 3. Output Directories
# ============================================================
OUTPUT_ROOT = Path(os.environ.get("GRIDWM_OUTPUT_ROOT", REPO_ROOT / "outputs"))
TEMP_ROOT = Path(os.environ.get("GRIDWM_TEMP_ROOT", REPO_ROOT / "tmp"))

# MiniWorld specific temp directory
MINIWORLD_TEMP_ROOT = TEMP_ROOT / "miniworld"

# ============================================================
# 4. Model Anonymization Mapping
# ============================================================
MODEL_ALIASES = {
    # OpenAI
    "gpt-4o": "Model-A",
    "gpt-4o-mini": "Model-B",
    "gpt-4-turbo": "Model-C",
    "gpt-3.5-turbo": "Model-D",

    # Anthropic
    "claude-3-5-sonnet-20241022": "Model-E",
    "claude-3-5-sonnet-latest": "Model-E",
    "claude-3-opus": "Model-F",
    "claude-3-sonnet": "Model-G",
    "claude-3-haiku": "Model-H",

    # Google
    "gemini-1.5-pro": "Model-I",
    "gemini-1.5-flash": "Model-J",
    "gemini-1.5-pro-latest": "Model-I",
    "gemini-1.5-flash-latest": "Model-J",
    "gemini-2.0-flash": "Model-K",
    "gemini-2.5-pro": "Model-L",
    "gemini-2.5-flash": "Model-M",

    # Other providers
    "llama-3.1-70b-instruct": "Model-N",
    "llama-3.1-8b-instruct": "Model-O",
    "mistral-large": "Model-P",
    "qwen-vl-max": "Model-Q",
    "qwen-vl-plus": "Model-R",
}

# Reverse mapping (Model-A -> gpt-4o)
ALIAS_TO_MODEL = {v: k for k, v in MODEL_ALIASES.items()}

# ============================================================
# 5. Utility Functions
# ============================================================

def anonymize_model(model_id: str) -> str:
    """Convert model ID to anonymous identifier."""
    if not model_id:
        return "unknown"
    return MODEL_ALIASES.get(model_id, model_id)


def deanonymize_model(alias: str) -> str:
    """Convert anonymous identifier back to original model ID."""
    if not alias:
        return "unknown"
    return ALIAS_TO_MODEL.get(alias, alias)


def get_dataset_path(dataset_type: str) -> Path:
    """Get path for specified dataset type."""
    mapping = {
        "minigrid": MINIGRID_ROOT,
        "miniworld": MINIWORLD_ROOT,
        "raw": RAW_DATA_ROOT,
        "global_views": GLOBAL_VIEWS_ROOT,
    }
    return mapping.get(dataset_type.lower(), DATA_ROOT)


def resolve_output_path(relative_path: str) -> Path:
    """Resolve output file path."""
    return OUTPUT_ROOT / relative_path


# ============================================================
# 6. Backward Compatibility Aliases
# ============================================================
CANONICAL_MINIGRID = MINIGRID_ROOT
CANONICAL_MINIWORLD = MINIWORLD_ROOT
