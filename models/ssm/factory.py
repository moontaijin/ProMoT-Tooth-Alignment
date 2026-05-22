# models/ssm/factory.py
"""SSM factory/registry.

Goal:
- Allow passing *one* dictionary (ssm_cfg) from train/config into the model,
  without scattering `if ssm==...` branches across train.py / teeth_model.py.

Design:
- `build_ssm(ssm_cfg, encoder=..., **common)` picks an SSM class from registry
  and forwards only the kwargs that the target __init__ accepts.
- Includes light canonicalization for config key aliases (e.g., S4 json uses
  n_layers/dropout; our S4RefinementSSM uses s4_layers/s4_dropout (hidden_dim comes from encoder)).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type
import inspect

from .kalman import KalmanSmoothRefinementSSM
from .s4 import S4RefinementSSM


SSM_REGISTRY: Dict[str, Type] = {
    "kalman": KalmanSmoothRefinementSSM,
    "kalman_smoothing": KalmanSmoothRefinementSSM,
    "s4": S4RefinementSSM,
    "s4d": S4RefinementSSM,
    "s4_refine": S4RefinementSSM,
    "s4refine": S4RefinementSSM,
}


def _pop_name(cfg: Dict[str, Any]) -> str:
    # Support several common keys.
    for k in ("name", "type", "ssm", "ssm_type"):
        if k in cfg and cfg[k] is not None:
            v = cfg.pop(k)
            return str(v).lower()
    return "kalman"


def _canonicalize_cfg(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map config aliases to the target class init kwargs."""
    out = dict(cfg)

    if name in ("s4", "s4d", "s4_refine", "s4refine"):
        # Common aliases in config/model/ssm/s4.json
        # NOTE: hidden_dim is owned by the encoder (encoder_out_dim). Ignore any d_model/hidden_dim fields.
        out.pop('d_model', None)
        out.pop('hidden_dim', None)

        # - n_layers/layers -> s4_layers
        if "s4_layers" not in out:
            if "n_layers" in out:
                out["s4_layers"] = out.pop("n_layers")
            elif "layers" in out:
                out["s4_layers"] = out.pop("layers")

        # - dropout -> s4_dropout
        if "s4_dropout" not in out and "dropout" in out:
            out["s4_dropout"] = out.pop("dropout")

        # - n_state/d_state -> s4_state
        if "s4_state" not in out:
            if "n_state" in out:
                out["s4_state"] = out.pop("n_state")
            elif "d_state" in out:
                out["s4_state"] = out.pop("d_state")

        # - ff_mult -> s4_ff_mult
        if "s4_ff_mult" not in out and "ff_mult" in out:
            out["s4_ff_mult"] = out.pop("ff_mult")

        # ignore keys that are not init-args (e.g., return_intermediates)

    return out


def _filter_kwargs(cls: Type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by cls.__init__ unless it has **kwargs."""
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    # If **kwargs present, no need to filter.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs

    allowed = set(params.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in allowed}


def build_ssm(
    ssm_cfg: Optional[Dict[str, Any]],
    *,
    encoder,
    **common,
):
    """Instantiate an SSM module.

    Args:
      ssm_cfg: dict like {name/type: 'kalman'|'s4', ...params...}.
               Also supports nested style {name: 's4', params: {...}}.
      encoder: injected feature encoder.
      common: common kwargs (K, hidden_dim, use_tooth_id_emb, ...)
              These override values from ssm_cfg.
    """
    cfg = dict(ssm_cfg or {})

    # nested params style
    nested = cfg.pop("params", None)
    if isinstance(nested, dict):
        for k, v in nested.items():
            cfg.setdefault(k, v)

    name = _pop_name(cfg)
    cls = SSM_REGISTRY.get(name, None)
    if cls is None:
        raise ValueError(f"Unknown SSM '{name}'. Available: {sorted(SSM_REGISTRY.keys())}")

    # Merge: config first, then common override
    merged = dict(cfg)
    merged.update(common)
    merged["encoder"] = encoder

    merged = _canonicalize_cfg(name, merged)
    merged = _filter_kwargs(cls, merged)
    return cls(**merged)
