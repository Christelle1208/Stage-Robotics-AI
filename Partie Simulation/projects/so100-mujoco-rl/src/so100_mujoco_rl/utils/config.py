"""YAML config loading utilities.

All paths resolved relative to the project root (two levels above this file,
i.e. the directory containing ``pyproject.toml``).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# The project root is three directories up from this file:
#   src/so100_mujoco_rl/utils/config.py  ->  project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_root() -> Path:
    """Return the absolute path to the project root."""
    return _PROJECT_ROOT


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    The path may be:
    * Absolute — used as-is.
    * Relative — resolved from the project root.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"(project root = {_PROJECT_ROOT})"
        )

    with path.open() as fh:
        cfg = yaml.safe_load(fh)

    return cfg or {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (deep merge, returns new dict)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root if it is not absolute."""
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p
