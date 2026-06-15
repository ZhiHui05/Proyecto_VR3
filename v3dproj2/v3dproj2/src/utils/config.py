"""
Configuration loader for the V3D Drone Control System.
Loads and manages YAML configuration files with nested key access.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """
    Loads and provides access to YAML configuration files.

    Supports nested key access via dot-separated paths
    (e.g. 'mediapipe.min_detection_confidence').
    """

    def __init__(self, config_dir: str = "configs") -> None:
        self._config_dir = Path(config_dir).resolve()
        self._data: dict[str, Any] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self._config_dir.exists():
            msg = f"Configuration directory not found: {self._config_dir}"
            raise FileNotFoundError(msg)
        for yaml_file in sorted(self._config_dir.glob("*.yaml")):
            with open(yaml_file, encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            self._data.update(loaded)
        for yml_file in sorted(self._config_dir.glob("*.yml")):
            with open(yml_file, encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            self._data.update(loaded)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        node: Any = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def get_typed(self, key: str, expected_type: type, default: Any = None) -> Any:
        value = self.get(key, default)
        if not isinstance(value, expected_type):
            return default
        return value

    def __getitem__(self, key: str) -> Any:
        result = self.get(key)
        if result is None:
            raise KeyError(f"Configuration key not found: {key}")
        return result

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def all(self) -> dict[str, Any]:
        return dict(self._data)


# Singleton instance
_config_instance: ConfigLoader | None = None


def get_config(config_dir: str = "configs") -> ConfigLoader:
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_dir)
    return _config_instance


def reload_config(config_dir: str = "configs") -> ConfigLoader:
    global _config_instance
    _config_instance = ConfigLoader(config_dir)
    return _config_instance
