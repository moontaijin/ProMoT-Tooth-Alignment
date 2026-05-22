"""Config-driven loss composition.

This module removes hard-coded per-loss variables from train.py.

Usage:
  composer = LossComposer.from_cfg(cfg_dict)
  loss_u, loss_l, logs = composer(pred_u, batch_u, pred_l, batch_l, ctx={...})
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .registry import build_loss


def _as_float(x: torch.Tensor | float | int) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().item())
    return float(x)


class LossComposer(nn.Module):
    """Compose jaw-wise and pair-wise loss terms."""

    def __init__(self, term_cfgs: List[Dict[str, Any]]):
        super().__init__()

        self.term_specs: List[Dict[str, Any]] = []
        self.jaw_terms: nn.ModuleList = nn.ModuleList()
        self.pair_terms: nn.ModuleList = nn.ModuleList()

        for tc in term_cfgs or []:
            if not isinstance(tc, dict) or 'name' not in tc:
                continue
            name = str(tc['name'])
            enabled = bool(tc.get('enabled', True))
            weight = float(tc.get('weight', 1.0))
            params = tc.get('params', None) or {}

            term = build_loss(name, params)
            scope = getattr(term, 'scope', 'jaw')
            if scope not in ('jaw', 'pair'):
                raise ValueError(f"Loss term '{name}' has invalid scope='{scope}'.")

            self.term_specs.append({
                'name': name,
                'enabled': enabled,
                'weight': weight,
                'scope': scope,
            })

            if scope == 'jaw':
                self.jaw_terms.append(term)
            else:
                self.pair_terms.append(term)

    @staticmethod
    def from_cfg(cfg_dict: Dict[str, Any]) -> "LossComposer":
        loss_cfg = (cfg_dict or {}).get('loss', {}) or {}
        term_cfgs = loss_cfg.get('terms', []) or []
        return LossComposer(term_cfgs)

    def forward(
        self,
        pred_u: Dict[str, torch.Tensor],
        batch_u: Dict[str, torch.Tensor],
        pred_l: Dict[str, torch.Tensor],
        batch_l: Dict[str, torch.Tensor],
        *,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:

        ctx = ctx or {}

        loss_u = pred_u['quat'].new_tensor(0.0)
        loss_l = pred_l['quat'].new_tensor(0.0)
        logs: Dict[str, float] = {}

        # --- jaw-wise terms ---
        jaw_i = 0
        for spec in self.term_specs:
            if spec['scope'] != 'jaw':
                continue
            term: nn.Module = self.jaw_terms[jaw_i]
            jaw_i += 1

            if (not spec['enabled']) or (spec['weight'] == 0.0):
                continue

            w = float(spec['weight'])
            name = spec['name']

            lu, log_u = term(pred_u, batch_u, ctx)
            ll, log_l = term(pred_l, batch_l, ctx)

            loss_u = loss_u + (lu * w)
            loss_l = loss_l + (ll * w)

            logs[f"loss/{name}/u"] = _as_float(lu) * w
            logs[f"loss/{name}/l"] = _as_float(ll) * w
            logs[f"loss/{name}"] = 0.5 * (logs[f"loss/{name}/u"] + logs[f"loss/{name}/l"])

            if isinstance(log_u, dict):
                for k, v in log_u.items():
                    if v is None:
                        continue
                    logs[f"loss/{name}/u/{k}"] = float(v)
            if isinstance(log_l, dict):
                for k, v in log_l.items():
                    if v is None:
                        continue
                    logs[f"loss/{name}/l/{k}"] = float(v)

        # --- pair-wise terms (computed once, then added symmetrically) ---
        pair_i = 0
        for spec in self.term_specs:
            if spec['scope'] != 'pair':
                continue
            term: nn.Module = self.pair_terms[pair_i]
            pair_i += 1

            if (not spec['enabled']) or (spec['weight'] == 0.0):
                continue

            w = float(spec['weight'])
            name = spec['name']

            lp, log_p = term(pred_u, batch_u, pred_l, batch_l, ctx)

            loss_u = loss_u + (lp * w)
            loss_l = loss_l + (lp * w)

            logs[f"loss/{name}"] = _as_float(lp) * w
            if isinstance(log_p, dict):
                for k, v in log_p.items():
                    if v is None:
                        continue
                    logs[f"loss/{name}/{k}"] = float(v)

        logs["loss/total/u"] = _as_float(loss_u)
        logs["loss/total/l"] = _as_float(loss_l)
        logs["loss/total"] = 0.5 * (logs["loss/total/u"] + logs["loss/total/l"])

        return loss_u, loss_l, logs
