"""Simple data-provider registry."""

from __future__ import annotations

from typing import Callable, Dict, Type

from .interface import DataProvider

PROVIDER_REGISTRY: Dict[str, Type[DataProvider]] = {}


def register_provider(name: str) -> Callable[[Type[DataProvider]], Type[DataProvider]]:
    def _decorator(cls: Type[DataProvider]) -> Type[DataProvider]:
        PROVIDER_REGISTRY[name] = cls
        return cls

    return _decorator


def build_provider(name: str, *args, **kwargs) -> DataProvider:
    if name not in PROVIDER_REGISTRY:
        known = ", ".join(sorted(PROVIDER_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown data provider {name!r}. Registered: {known}")
    return PROVIDER_REGISTRY[name](*args, **kwargs)


def registered_providers() -> list[str]:
    return sorted(PROVIDER_REGISTRY.keys())
