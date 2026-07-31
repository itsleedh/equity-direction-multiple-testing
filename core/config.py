from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML config, with PyYAML support when available."""
    config_path = Path(path)
    raw = config_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(raw)
    except ModuleNotFoundError:
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                "PyYAML is not installed and the config is not JSON-compatible YAML. "
                "Install PyYAML or keep config.yaml in JSON-compatible syntax."
            ) from exc

    if not isinstance(loaded, dict):
        raise ConfigError(f"Config at {config_path} must define a mapping/object.")
    return loaded


def resolve_path(config_path: str | Path, configured_path: str | Path) -> Path:
    """Resolve a config-relative path."""
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return Path(config_path).resolve().parent / path
