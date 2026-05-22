"""S4(S4D)-based refinement SSM for ProMoT/STTAlign-style tooth alignment.

This module is designed to be a drop-in replacement for
`KalmanSmoothRefinementSSM` (models/ssm/kalman.py).

Key design choices for this repo:
  - Sequence axis is the *stage index* k = 0..K-1 (K is small, e.g., 8).
  - We implement a lightweight S4D (Diagonal S4) layer with complex diagonal
    state matrix A and per-channel state size `n_state`.
  - We keep everything self-contained (no external S4 packages).

Shapes:
  points_in: (B,N,P,3)
  tooth_id:  (B,N)
  mask:      (B,N) bool
  t:         (B,1)

Outputs match KalmanSmoothRefinementSSM:
  - T: (B,N,4,4)
  - deltas: list length K, each (B,N,6)
  - h_last: (B,N,H)
  - (optional) aux dict with intermediate tensors
"""

from __future__ import annotations

from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.geometry import exp_se3


def _as_complex(re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    return torch.complex(re, im)


class S4DLayer(nn.Module):
    """Diagonal state-space layer (S4D-style).

    Implements a per-channel diagonal complex SSM with ZOH discretization.
    This is intentionally simple and robust for small sequence lengths (K<=16).
    """

    def __init__(
        self,
        d_model: int,
        n_state: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_state = int(n_state)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)

        # A = -exp(log_A_real) + i * A_imag  (stable real part)
        self.log_A_real = nn.Parameter(torch.randn(self.d_model, self.n_state) * 0.02)
        self.A_imag = nn.Parameter(torch.randn(self.d_model, self.n_state) * 0.02)

        # Complex B, C
        self.B_re = nn.Parameter(torch.randn(self.d_model, self.n_state) * 0.02)
        self.B_im = nn.Parameter(torch.randn(self.d_model, self.n_state) * 0.02)
        self.C_re = nn.Parameter(torch.randn(self.d_model, self.n_state) * 0.02)
        self.C_im = nn.Parameter(torch.randn(self.d_model, self.n_state) * 0.02)

        # Skip term D
        self.D = nn.Parameter(torch.zeros(self.d_model))

        # Per-channel dt in (dt_min, dt_max)
        self.logit_dt = nn.Parameter(torch.zeros(self.d_model))

        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def _dt(self, u: torch.Tensor, dt_sample: Optional[torch.Tensor]) -> torch.Tensor:
        # per-channel dt
        dt_ch = torch.sigmoid(self.logit_dt).to(device=u.device, dtype=u.dtype)
        dt_ch = self.dt_min + (self.dt_max - self.dt_min) * dt_ch
        dt_ch = dt_ch.view(1, self.d_model, 1)  # (1,H,1)

        if dt_sample is None:
            return dt_ch

        if dt_sample.dim() == 1:
            dt_sample = dt_sample.view(-1, 1)
        dt_sample = dt_sample.to(device=u.device, dtype=u.dtype).view(-1, 1, 1)
        return dt_ch * dt_sample

    def forward(self, u: torch.Tensor, dt_sample: Optional[torch.Tensor] = None) -> torch.Tensor:
        """u: (B,L,H) -> (B,L,H)."""
        if u.dim() != 3:
            raise ValueError(f"S4DLayer expects (B,L,H), got {tuple(u.shape)}")
        B, L, H = u.shape
        if H != self.d_model:
            raise ValueError(f"d_model mismatch: layer={self.d_model}, input={H}")

        # Build complex parameters in fp32 for stable discretization/scan
        A = _as_complex(-torch.exp(self.log_A_real.float()), self.A_imag.float())  # (H,N)
        Bc = _as_complex(self.B_re.float(), self.B_im.float())
        Cc = _as_complex(self.C_re.float(), self.C_im.float())
        D = self.D.to(device=u.device, dtype=u.dtype).view(1, 1, H)

        dt = self._dt(u, dt_sample).float()   # (B,H,1)
        A_b = A.unsqueeze(0)                 # (1,H,N)
        Ad = torch.exp(A_b * dt)             # (B,H,N)
        A_eps = torch.where(torch.abs(A_b) < 1e-6, torch.ones_like(A_b) * 1e-6, A_b)
        Bd = (Ad - 1.0) / A_eps * Bc.unsqueeze(0)  # (B,H,N)

        x = torch.zeros((B, H, self.n_state), device=u.device, dtype=torch.complex64)
        u_f = u.float()
        ys: List[torch.Tensor] = []
        for k in range(L):
            uk = u_f[:, k].unsqueeze(-1)  # (B,H,1)
            x = Ad * x + Bd * uk
            yk = (Cc.unsqueeze(0) * x).sum(dim=-1).real  # (B,H)
            ys.append(yk)
        y = torch.stack(ys, dim=1).to(dtype=u.dtype)  # (B,L,H)

        y = y + D * u
        y = self.drop(y)
        return y


class S4Block(nn.Module):
    """PreNorm S4D + PreNorm FFN (Transformer-like)."""

    def __init__(
        self,
        d_model: int,
        n_state: int = 64,
        dropout: float = 0.0,
        ff_mult: int = 2,
        dt_min: float = 1e-3,
        dt_max: float = 1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.s4 = S4DLayer(
            d_model=d_model,
            n_state=n_state,
            dt_min=dt_min,
            dt_max=dt_max,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor, dt_sample: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.s4(self.norm1(x), dt_sample=dt_sample)
        x = x + z
        x = x + self.ff(self.norm2(x))
        return x


class S4Stack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_state: int = 64,
        dropout: float = 0.0,
        ff_mult: int = 2,
        dt_min: float = 1e-3,
        dt_max: float = 1.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                S4Block(
                    d_model=d_model,
                    n_state=n_state,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    dt_min=dt_min,
                    dt_max=dt_max,
                )
                for _ in range(int(n_layers))
            ]
        )

    def forward(self, x: torch.Tensor, dt_sample: Optional[torch.Tensor] = None) -> torch.Tensor:
        for blk in self.layers:
            x = blk(x, dt_sample=dt_sample)
        return x


class S4RefinementSSM(nn.Module):
    """S4-based K-step refinement SSM (drop-in for kalman.KalmanSmoothRefinementSSM)."""

    def __init__(
        self,
        hidden_dim: int = 256,
        K: int = 8,
        use_tooth_id_emb: bool = True,
        encoder: Optional[nn.Module] = None,
        # S4 hyperparams
        s4_layers: int = 2,
        s4_state: int = 64,
        s4_dropout: float = 0.0,
        s4_ff_mult: int = 2,
        dt_min: float = 1e-3,
        dt_max: float = 1.0,
    ):
        super().__init__()
        if encoder is None:
            raise ValueError("S4RefinementSSM requires an injected encoder.")
        self.enc = encoder
        self.K = int(K)
        self.hidden_dim = int(hidden_dim)

        feat_dim = int(getattr(self.enc, "out_dim", hidden_dim))

        self.use_tooth_id_emb = bool(use_tooth_id_emb)
        id_dim = 0
        if self.use_tooth_id_emb:
            self.id_emb = nn.Embedding(33, 32)
            id_dim = 32

        self.step_emb = nn.Embedding(self.K, 32)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        obs_in = feat_dim + feat_dim + id_dim + 32 + 32
        self.obs_net = nn.Sequential(
            nn.Linear(obs_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.s4 = S4Stack(
            d_model=hidden_dim,
            n_layers=s4_layers,
            n_state=s4_state,
            dropout=s4_dropout,
            ff_mult=s4_ff_mult,
            dt_min=dt_min,
            dt_max=dt_max,
        )

        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(
        self,
        points_in: torch.Tensor,
        tooth_id: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        return_intermediates: bool = False,
    ):
        B, N, _, _ = points_in.shape
        device, dtype = points_in.device, points_in.dtype

        e = self.enc(points_in, mask=mask)                           # (B,N,C)
        e_masked = e.clone()
        e_masked[~mask] = -1e9
        g = e_masked.max(dim=1).values.unsqueeze(1).repeat(1, N, 1)   # (B,N,C)

        if self.use_tooth_id_emb:
            tid = torch.clamp(tooth_id, 0, 32)
            emb = self.id_emb(tid)
        else:
            emb = None

        # dt per sample (same semantics as kalman.py)
        dt = (t / float(self.K)).to(device=device, dtype=dtype)       # (B,1)
        dt_bn = dt.repeat(1, N).reshape(B * N, 1)                     # (BN,1)

        time_feat = self.time_mlp(t.to(device=device, dtype=dtype))   # (B,32)
        time_feat = time_feat.unsqueeze(1).repeat(1, N, 1)            # (B,N,32)

        y_seq = []
        for k in range(self.K):
            step_vec = self.step_emb.weight[k].view(1, 1, -1).expand(B, N, -1)
            if self.use_tooth_id_emb:
                obs_in = torch.cat([e, g, emb, step_vec, time_feat], dim=-1)
            else:
                obs_in = torch.cat([e, g, step_vec, time_feat], dim=-1)
            y = self.obs_net(obs_in)                                  # (B,N,H)
            y = y * mask.unsqueeze(-1).float()
            y_seq.append(y)

        y_seq = torch.stack(y_seq, dim=0)                             # (K,B,N,H)
        y_bn = y_seq.permute(1, 2, 0, 3).reshape(B * N, self.K, self.hidden_dim)  # (BN,K,H)

        h_bn = self.s4(y_bn, dt_sample=dt_bn)                         # (BN,K,H)
        h_seq = h_bn.reshape(B, N, self.K, self.hidden_dim).permute(2, 0, 1, 3).contiguous()  # (K,B,N,H)

        T = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        deltas: List[torch.Tensor] = []
        T_steps: Optional[List[torch.Tensor]] = [] if return_intermediates else None

        for k in range(self.K):
            h_k = h_seq[k]
            xi = self.delta_head(h_k.reshape(B * N, self.hidden_dim)).reshape(B, N, 6)
            xi = xi * mask.unsqueeze(-1).float()

            # same clamp ranges as kalman.py
            w = torch.clamp(xi[..., :3], -0.2, 0.2)
            v = torch.clamp(xi[..., 3:], -0.5, 0.5)
            xi = torch.cat([w, v], dim=-1)

            dT = exp_se3(xi)
            T = T @ dT
            deltas.append(xi)

            if return_intermediates:
                assert T_steps is not None
                T_steps.append(T)

        h_last = h_seq[-1]

        if not return_intermediates:
            return T, deltas, h_last

        aux: Dict[str, torch.Tensor] = {
            "y_seq": y_seq,                      # (K,B,N,H)
            "h_seq": h_seq,                      # (K,B,N,H)
            "T_steps": torch.stack(T_steps, 0),   # (K,B,N,4,4)
            "deltas": torch.stack(deltas, 0),     # (K,B,N,6)
        }
        return T, deltas, h_last, aux
