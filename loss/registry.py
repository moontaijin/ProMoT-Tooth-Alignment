"""Loss registry.

Each loss term is a small nn.Module registered by name.
This allows config-driven composition and clean ablations.
"""

from __future__ import annotations

from typing import Any, Dict, Type

import torch.nn as nn


LOSS_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_loss(name: str):
    name = str(name)

    def _wrap(cls):
        if name in LOSS_REGISTRY:
            raise KeyError(f"Duplicate loss name: {name}")
        LOSS_REGISTRY[name] = cls
        return cls

    return _wrap


def build_loss(name: str, params: Dict[str, Any] | None = None) -> nn.Module:
    name = str(name)
    if name not in LOSS_REGISTRY:
        raise KeyError(
            f"Unknown loss term '{name}'. Registered: {sorted(LOSS_REGISTRY.keys())}"
        )
    cls = LOSS_REGISTRY[name]
    params = params or {}
    return cls(**params)
