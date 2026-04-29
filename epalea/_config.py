"""
Config file loading and merging for CLI commands.
CLI flags override YAML values. All commands support --config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load YAML config file. Returns empty dict if path is None."""
    if config_path is None:
        return {}
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    return data


def merge(config: Dict[str, Any], **cli_overrides) -> Dict[str, Any]:
    """
    Merge config file values with CLI overrides.
    CLI values (non-None) always win over config file values.
    """
    merged = dict(config)
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def get(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get value from merged config, falling back to default."""
    return config.get(key, default)
