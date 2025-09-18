"""Compatibility helpers providing light-weight Pydantic fallbacks."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field as dataclass_field
from typing import Any, Dict, Optional


class BaseModel:
    """Minimal stand-in replicating key Pydantic BaseModel APIs."""

    def __init_subclass__(cls, **kwargs: Any) -> None:  # pragma: no cover - simple shim
        dataclass(cls)
        super().__init_subclass__(**kwargs)

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)

    def model_copy(self, update: Optional[Dict[str, Any]] = None):
        data = self.model_dump()
        if update:
            data.update(update)
        return type(self)(**data)


def Field(*, default: Any = MISSING, default_factory: Any = MISSING):  # pragma: no cover - shim
    if default is not MISSING and default_factory is not MISSING:
        raise ValueError("default and default_factory are mutually exclusive")
    if default_factory is not MISSING:
        return dataclass_field(default_factory=default_factory)
    if default is not MISSING:
        return dataclass_field(default=default)
    return dataclass_field()


__all__ = ["BaseModel", "Field"]
