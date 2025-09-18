"""Configuration management for RAGX."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - fallback when PyYAML missing
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "profile": "balanced_efficiency",
    "retrieval": {
        "method": "hybrid",
        "alpha": 0.30,
        "top_k": 10,
    },
    "rerank": {
        "method": "monoT5",
        "top_k": 5,
    },
    "packing": "reverse",
    "summarization": {
        "primary": "recomp",
        "fallback": "longllmlingua",
    },
    "budget": {
        "max_context_tokens": 2500,
    },
    "crag": {
        "min_score": 0.55,
    },
    "selfrag": {
        "max_loops": 2,
    },
    "tensor": {
        "enabled": False,
    },
    "mcp": {
        "servers": [],
    },
}


PROFILE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "best_performance": {
        "retrieval": {"alpha": 0.35, "top_k": 12},
        "rerank": {"top_k": 8},
        "budget": {"max_context_tokens": 3200},
    },
    "balanced_efficiency": {},
    "custom": {},
}


@dataclass
class ConfigManager:
    """Utility that merges YAML, profile defaults and environment overrides."""

    config_path: Optional[Path] = None
    base_config: Dict[str, Any] = field(default_factory=lambda: DEFAULT_CONFIG.copy())

    def load(self, profile: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        profile = profile or self.base_config.get("profile", "balanced_efficiency")
        merged: Dict[str, Any] = {}
        _merge_dicts(merged, self.base_config)

        if self.config_path and self.config_path.exists() and yaml:
            with self.config_path.open("r", encoding="utf-8") as handle:
                yaml_config = yaml.safe_load(handle) or {}
            _merge_dicts(merged, yaml_config)

        profile_overrides = PROFILE_OVERRIDES.get(profile, {})
        _merge_dicts(merged, profile_overrides)

        if overrides:
            _merge_dicts(merged, overrides)

        env_overrides = _env_overrides()
        if env_overrides:
            _merge_dicts(merged, env_overrides)

        merged["profile"] = profile
        return merged


def _merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict):
            node = base.setdefault(key, {})
            if isinstance(node, dict):
                _merge_dicts(node, value)
            else:
                base[key] = value
        else:
            base[key] = value


def _env_overrides() -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    alpha = os.getenv("RAGX_ALPHA")
    if alpha is not None:
        try:
            overrides.setdefault("retrieval", {})["alpha"] = float(alpha)
        except ValueError:
            pass

    top_k = os.getenv("RAGX_TOP_K")
    if top_k is not None:
        try:
            overrides.setdefault("retrieval", {})["top_k"] = int(top_k)
        except ValueError:
            pass

    budget = os.getenv("RAGX_MAX_CONTEXT")
    if budget is not None:
        try:
            overrides.setdefault("budget", {})["max_context_tokens"] = int(budget)
        except ValueError:
            pass

    return overrides


__all__ = ["ConfigManager", "DEFAULT_CONFIG", "PROFILE_OVERRIDES"]
