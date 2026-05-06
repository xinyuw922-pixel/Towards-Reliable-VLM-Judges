#!/usr/bin/env python3
"""
Best-effort project .env loading.

Prefer python-dotenv when installed, but keep scripts bootable when the package
is missing by falling back to a minimal KEY=VALUE parser.
"""

from __future__ import annotations

import os
from pathlib import Path


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(value):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if idx == 0 or value[idx - 1].isspace():
                return value[:idx].rstrip()
    return value.rstrip()


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export "):].lstrip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key or any(ch.isspace() for ch in key):
        return None
    value = _strip_inline_comment(value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def _fallback_load_dotenv(path: Path, override: bool) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if not parsed:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value


def load_project_dotenv(dotenv_path: Path | None = None, override: bool = False) -> str:
    path = Path(dotenv_path) if dotenv_path is not None else Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return "missing"
    try:
        from dotenv import load_dotenv
    except ImportError:
        _fallback_load_dotenv(path, override=override)
        return "fallback"
    load_dotenv(path, override=override)
    return "python-dotenv"
