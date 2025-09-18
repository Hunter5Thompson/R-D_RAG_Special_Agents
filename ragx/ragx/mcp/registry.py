"""Minimal MCP registry managing allowed tool servers."""

from __future__ import annotations

from typing import Dict, List


class MCPRegistry:
    def __init__(self) -> None:
        self.allowed_servers: Dict[str, Dict[str, str]] = {}

    def register(self, name: str, config: Dict[str, str]) -> None:
        self.allowed_servers[name] = config

    def unregister(self, name: str) -> None:
        self.allowed_servers.pop(name, None)

    def list_servers(self) -> List[str]:
        return list(self.allowed_servers.keys())


__all__ = ["MCPRegistry"]
