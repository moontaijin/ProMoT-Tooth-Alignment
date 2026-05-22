"""Loss terms currently used in the training loop.

We only implement losses that already exist in the current code path:
  - recon (GeometricReconstructionLoss): recon + w_c * c
  - dof  (weighted SmoothL1 on quaternion)
  - trans (weighted SmoothL1 on translation)
  - angle (1 - dot(quat, quat_gt))
  - occlusion (interdental_occlusion_loss), optionally detached (legacy behavior)
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from pytorch3d.transforms import quaternion_to_matrix

from .registry import register_loss
from .sttalign_loss import GeometricReconstructionLoss, interdental_occlusion_loss
from utils.geometry import exp_se3, log_se3, invert_se3, apply_T_points

from contextlib import nullcontext

class _JawLoss(nn.Module):
    scope = 'jaw'


class _PairLoss(nn.Module):
    scope = 'pair'


@register_loss('recon')
class ReconTerm(_JawLoss):
    def __init__(self, w_c: float = 1.0):
        super().__init__()
        self.w_c = float(w_c)
        self._loss = GeometricReconstructionLoss()

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], ctx: Dict[str, Any]):
        assembled = pred['assembled']
        label = batch['label']
        weights = batch['weights']
        device = ctx.get('device', assembled.device)

        recon, c = self._loss(assembled, label, weights, device)
        total = recon + (c * self.w_c)
        logs = {
            'recon': float(recon.detach().item()),
            'c': float(c.detach().item()),
            'w_c': self.w_c,
        }
        return total, logs


@register_loss('dof')
class DofTerm(_JawLoss):
    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], ctx: Dict[str, Any]):
        pdofs = pred['quat']
        gdofs = batch['gdofs']
        rweights = batch['rweights']
        mask_index = batch['mask_index']

        pd = pdofs[mask_index]
        gd = gdofs[mask_index]
        rw = rweights[mask_index]

        dot = (pd * gd).sum(dim=-1, keepdim=True)          # (..,1)
        gd = torch.where(dot < 0, -gd, gd)                 # sign-align
       
        loss = torch.sum(torch.sum(F.smooth_l1_loss(pd, gd, reduction='none'), dim=-1) * rw) / pd.shape[0]
        return loss, None


@register_loss('trans')
class TransTerm(_JawLoss):
    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], ctx: Dict[str, Any]):
        ptrans = pred['trans']
        gtrans = batch['gtrans']
        tweights = batch['tweights']
        mask_index = batch['mask_index']

        pt = ptrans[mask_index]
        gt = gtrans[mask_index]
        tw = tweights[mask_index]

        loss = torch.sum(torch.sum(F.smooth_l1_loss(pt, gt, reduction='none'), dim=-1) * tw) / pt.shape[0]
        return loss, None


@register_loss('angle')
class AngleTerm(_JawLoss):
    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], ctx: Dict[str, Any]):
        pdofs = pred['quat']
        gdofs = batch['gdofs']
        mask_index = batch['mask_index']

        pd = pdofs[mask_index]
        gd = gdofs[mask_index]

        dot = torch.sum(pd * gd, dim=-1)
        loss = torch.mean(1.0 - torch.abs(dot))
        
        return loss, None


@register_loss('occlusion')
class OcclusionTerm(_PairLoss):
    def __init__(
        self,
        detach: bool = True,
        w_cox: float = 1.0,
        w_groove: float = 1.0,
        w_incisors: float = 1.0,
    ):
        super().__init__()
        self.detach = bool(detach)
        self.w_cox = float(w_cox)
        self.w_groove = float(w_groove)
        self.w_incisors = float(w_incisors)

    def forward(
        self,
        pred_u: Dict[str, torch.Tensor], batch_u: Dict[str, torch.Tensor],
        pred_l: Dict[str, torch.Tensor], batch_l: Dict[str, torch.Tensor],
        ctx: Dict[str, Any],
    ):
        assembled_u = pred_u['assembled']
        assembled_l = pred_l['assembled']
        label_u = batch_u['label']
        label_l = batch_l['label']
        device = ctx.get('device', assembled_u.device)

        cox, groove, incisors = interdental_occlusion_loss(assembled_u, label_u, assembled_l, label_l, device)
        total = (cox * self.w_cox) + (groove * self.w_groove) + (incisors * self.w_incisors)
        if self.detach:
            total = total.detach()

        logs = {
            'cox': float(cox.detach().item()),
            'groove': float(groove.detach().item()),
            'incisors': float(incisors.detach().item()),
            'detach': float(self.detach),
        }
        return total, logs

# -----------------------------------------------------------------------------
# New terms (requested):
#   - arch_geodesic: arch-aware geodesic/cross-track/tangent loss using GT arch
#   - movement_balance: encourage teeth to move more evenly (spread) per jaw
# NOTE: These are implemented to work with the CURRENT training data structures
#       (pred['assembled'], batch['label'], pred['quat'], pred['trans'], etc.).
# -----------------------------------------------------------------------------

def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim), min=eps))


def _quat_angle_rad(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return rotation angle (rad) from quaternion (w,x,y,z)."""
    w = torch.clamp(q[..., 0].abs(), 0.0, 1.0)
    return 2.0 * torch.acos(torch.clamp(w, min=eps, max=1.0))


@register_loss('arch_geodesic')
class ArchGeodesicTerm(_JawLoss):
    """Arch-aware geodesic loss (GT arch polyline).

    Build a per-sample GT arch polyline using GT tooth centroids (batch['label']) and compute:
      - geodesic arc-length error: |s_pred - s_gt|
      - cross-track distance: ||c_pred - proj_arch(c_pred)||
      - (optional) tangent alignment: 1 - <axis_pred_xy, tangent_xy>

    Inputs used:
      pred['assembled']: (B,T,P,3)
      pred['quat']:      (B,T,4)
      batch['label']:    (B,T,P,3)
    """

    def __init__(
        self,
        lambda_geo: float = 1.0,
        lambda_perp: float = 0.3,
        lambda_tan: float = 0.0,
        use_xy: bool = True,
        soft_projection: bool = False,
        soft_temp: float = 0.01,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_geo = float(lambda_geo)
        self.lambda_perp = float(lambda_perp)
        self.lambda_tan = float(lambda_tan)
        self.use_xy = bool(use_xy)
        self.soft_projection = bool(soft_projection)
        self.soft_temp = float(soft_temp)
        self.eps = float(eps)

    @staticmethod
    def _centroid(pts: torch.Tensor) -> torch.Tensor:
        return pts.mean(dim=2)  # (B,T,3)

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], ctx: Dict[str, Any]):
        assembled = pred['assembled']  # (B,T,P,3)
        label = batch['label']        # (B,T,P,3)

        c_pred = self._centroid(assembled)
        c_gt = self._centroid(label)

        if self.use_xy:
            c_pred = c_pred.clone()
            c_gt = c_gt.clone()
            c_pred[..., 2] = 0.0
            c_gt[..., 2] = 0.0

        # GT polyline segments
        a = c_gt[:, :-1, :]  # (B,S,3)
        b = c_gt[:, 1:, :]   # (B,S,3)
        seg = b - a          # (B,S,3)
        seg_len = _safe_norm(seg, dim=-1, eps=self.eps)  # (B,S)
        seg_dir = seg / seg_len.unsqueeze(-1)            # (B,S,3)

        # arc-length coordinate for GT nodes (index-based)
        cum = torch.cumsum(seg_len, dim=1)               # (B,S)
        s0 = seg_len.new_zeros((seg_len.shape[0], 1))    # (B,1)
        s_nodes = torch.cat([s0, cum], dim=1)            # (B,T)
        s_gt = s_nodes

        # project each predicted centroid onto GT polyline
        p = c_pred
        S = seg.shape[1]
        ap = p.unsqueeze(2) - a.unsqueeze(1)             # (B,T,S,3)
        seg_e = seg.unsqueeze(1)                         # (B,1,S,3)
        denom = torch.sum(seg * seg, dim=-1).clamp_min(self.eps)  # (B,S)
        u = torch.sum(ap * seg_e, dim=-1) / denom.unsqueeze(1)    # (B,T,S)
        u = torch.clamp(u, 0.0, 1.0)
        proj_all = a.unsqueeze(1) + u.unsqueeze(-1) * seg_e       # (B,T,S,3)
        dist2 = torch.sum((p.unsqueeze(2) - proj_all) ** 2, dim=-1)  # (B,T,S)

        s_seg0 = s_nodes[:, :-1]                                   # (B,S)
        s_proj_all = s_seg0.unsqueeze(1) + u * seg_len.unsqueeze(1) # (B,T,S)

        if self.soft_projection:
            temp = max(self.soft_temp, 1e-6)
            w = torch.softmax(-dist2 / temp, dim=-1)                # (B,T,S)
            s_pred = torch.sum(w * s_proj_all, dim=-1)              # (B,T)
            proj = torch.sum(w.unsqueeze(-1) * proj_all, dim=-2)    # (B,T,3)
            tan = torch.sum(w.unsqueeze(-1) * seg_dir.unsqueeze(1), dim=-2)  # (B,T,3)
            tan = tan / _safe_norm(tan, dim=-1, eps=self.eps).unsqueeze(-1)
        else:
            idx = torch.argmin(dist2, dim=-1)                       # (B,T)
            gather_idx = idx.unsqueeze(-1)
            s_pred = torch.gather(s_proj_all, dim=-1, index=gather_idx).squeeze(-1)  # (B,T)
            proj = torch.gather(
                proj_all,
                dim=2,
                index=gather_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            ).squeeze(2)
            tan = torch.gather(
                seg_dir,
                dim=1,
                index=idx.clamp_max(S - 1).unsqueeze(-1).expand(-1, -1, 3)
            )

        l_geo = torch.mean(torch.abs(s_pred - s_gt))
        l_perp = torch.mean(_safe_norm(p - proj, dim=-1, eps=self.eps))

        l_tan = assembled.new_tensor(0.0)
        if self.lambda_tan != 0.0:
            q = pred['quat']
            R = quaternion_to_matrix(q)          # (B,T,3,3)
            x_axis = torch.tensor([1.0, 0.0, 0.0], device=R.device, dtype=R.dtype)
            axis = torch.matmul(R, x_axis)       # (B,T,3)
            if self.use_xy:
                axis = axis.clone(); axis[..., 2] = 0.0
                tan2 = tan.clone(); tan2[..., 2] = 0.0
            else:
                tan2 = tan
            axis = axis / _safe_norm(axis, dim=-1, eps=self.eps).unsqueeze(-1)
            tan2 = tan2 / _safe_norm(tan2, dim=-1, eps=self.eps).unsqueeze(-1)
            cos = torch.sum(axis * tan2, dim=-1).clamp(-1.0, 1.0)
            l_tan = torch.mean(1.0 - cos)

        total = (self.lambda_geo * l_geo) + (self.lambda_perp * l_perp) + (self.lambda_tan * l_tan)
        logs = {
            'geo': float(l_geo.detach().item()),
            'perp': float(l_perp.detach().item()),
            'tan': float(l_tan.detach().item()) if self.lambda_tan != 0.0 else 0.0,
            'lambda_geo': self.lambda_geo,
            'lambda_perp': self.lambda_perp,
            'lambda_tan': self.lambda_tan,
        }
        return total, logs

@register_loss('movement_balance')
class MovementBalanceTerm(_JawLoss):
    def __init__(
        self,
        alpha_trans: float = 1.0,
        beta_rot: float = 0.1,
        mode: str = 'pairwise_l2',   # 'pairwise_l2' | 'pairwise_skl' | (기존 'l2_uniform' 등)
        use_steps: bool = True,
        eps: float = 1e-6,
        peak_cap: float = 0.0,       # 0이면 비활성. 예: 0.35면 "한 치아가 step 이동량의 35% 초과"만 벌점
        peak_w: float = 0.1,         # peak_cap 가중치
    ):
        super().__init__()
        self.alpha_trans = float(alpha_trans)
        self.beta_rot = float(beta_rot)
        self.mode = str(mode)
        self.use_steps = bool(use_steps)
        self.eps = float(eps)
        self.peak_cap = float(peak_cap)
        self.peak_w = float(peak_w)

    def _to_dist(self, m: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        # m: (B,T), active: (B,T) bool
        m = torch.clamp(m, min=0.0) * active.float()
        s = m.sum(dim=1, keepdim=True) + self.eps
        return m / s  # (B,T)

    def _pairwise_div(self, p0: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        if self.mode == 'pairwise_skl':
            # symmetric KL: KL(p0||p1) + KL(p1||p0)
            kl01 = (p0 * (torch.log(p0 + eps) - torch.log(p1 + eps))).sum(dim=1)
            kl10 = (p1 * (torch.log(p1 + eps) - torch.log(p0 + eps))).sum(dim=1)
            return (kl01 + kl10).mean()
        # default: pairwise_l2
        return ((p0 - p1) ** 2).sum(dim=1).mean()

    def forward(self, pred, batch, ctx):
        # active mask (없으면 전부 True로 처리)
        if 'mask' in batch:
            active = batch['mask'].bool()
        else:
            B, T = pred['trans'].shape[:2]
            active = torch.ones((B, T), device=pred['trans'].device, dtype=torch.bool)

        if self.use_steps and isinstance(pred.get('aux', None), dict) and ('deltas' in pred['aux']):
            deltas = pred['aux']['deltas']  # (K,B,T,6), 너희 컨벤션: [w(3), v(3)]
            if deltas.dim() != 4 or deltas.shape[-1] != 6:
                raise ValueError(f"movement_balance: expected deltas (K,B,T,6), got {tuple(deltas.shape)}")

            # step별 magnitude
            w = _safe_norm(deltas[..., :3], dim=-1, eps=self.eps)   # (K,B,T) rot
            v = _safe_norm(deltas[..., 3:], dim=-1, eps=self.eps)   # (K,B,T) trans
            m = self.alpha_trans * v + self.beta_rot * w            # (K,B,T)

            # (선택) step에서 "너무 몰빵"만 막는 peak cap (uniform 강제 아님)
            peak_pen = 0.0
            if self.peak_cap > 0.0:
                # p_k의 max mass가 peak_cap 초과할 때만 벌점
                p_all = self._to_dist(m.reshape(-1, m.shape[-1]), active.repeat(m.shape[0], 1))
                max_mass = p_all.max(dim=1).values
                peak_pen = F.relu(max_mass - self.peak_cap).mean()

            # 핵심: 인접 step 분포만 smooth하게
            loss = 0.0
            cnt = 0
            for k in range(m.shape[0] - 1):
                p0 = self._to_dist(m[k], active)
                p1 = self._to_dist(m[k + 1], active)
                loss = loss + self._pairwise_div(p0, p1)
                cnt += 1
            loss = loss / max(cnt, 1)

            if self.peak_cap > 0.0:
                loss = loss + self.peak_w * peak_pen

            return loss, {'use_steps': 1.0, 'pairwise': 1.0, 'peak_cap': float(self.peak_cap)}

        # use_steps가 아니면 기존 final 기반으로 fallback (여긴 네 기존 구현 쓰면 됨)
        trans = pred['trans']
        quat = pred['quat']
        m_final = self.alpha_trans * _safe_norm(trans, dim=-1, eps=self.eps) + self.beta_rot * _quat_angle_rad(quat, eps=self.eps)
        p = self._to_dist(m_final, active)
        # final에서도 uniform 강제하고 싶지 않으면 여기서는 0 리턴하거나 아주 작은 가중치로만
        return (p * 0.0).sum(), {'use_steps': 0.0, 'pairwise': 0.0}
    

@register_loss('geodesic_step')
class GeodesicStepTerm(_JawLoss):
    """
    pred['aux']['T_steps'] (K,B,T,4,4)를 이용해 stage-wise geodesic supervision 수행.
    """
    def __init__(
        self,
        rot_w: float = 1.0,
        trans_w: float = 1.0,
        trans_scale: float = 10.0,
        step_weight: str = 'uniform',   # uniform|linear|sqrt|exp
        use_weights: bool = True,
        detach_gt: bool = True,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.rot_w = float(rot_w)
        self.trans_w = float(trans_w)
        self.trans_scale = float(trans_scale)
        self.step_weight = str(step_weight)
        self.use_weights = bool(use_weights)
        self.detach_gt = bool(detach_gt)
        self.eps = float(eps)

    def _build_T(self, quat_wxyz: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        R = quaternion_to_matrix(quat_wxyz)  # (B,T,3,3)
        B, T = trans.shape[:2]
        M = torch.eye(4, device=trans.device, dtype=trans.dtype).view(1, 1, 4, 4).repeat(B, T, 1, 1)
        M[..., :3, :3] = R
        M[..., 3, :3] = trans
        return M

    def _rho(self, K: int, device, dtype) -> torch.Tensor:
        if self.step_weight in ('uniform', 'const', 'none'):
            return torch.ones((K,), device=device, dtype=dtype)
        if self.step_weight == 'linear':
            return torch.linspace(1.0 / K, 1.0, K, device=device, dtype=dtype)
        if self.step_weight == 'sqrt':
            return torch.sqrt(torch.linspace(1.0 / K, 1.0, K, device=device, dtype=dtype))
        if self.step_weight in ('exp', 'exponential'):
            t = torch.linspace(0.0, 1.0, K, device=device, dtype=dtype)
            return torch.exp(t)
        return torch.ones((K,), device=device, dtype=dtype)

    def forward(self, pred, batch, ctx):
        aux = pred.get('aux', None)
        if not isinstance(aux, dict) or ('T_steps' not in aux):
            z = pred['trans'].new_tensor(0.0)
            return z, {'enabled': 0.0}

        T_steps = aux['T_steps']  # (K,B,T,4,4)

        T_gt = self._build_T(batch['gdofs'], batch['gtrans'])
        if self.detach_gt:
            T_gt = T_gt.detach()

        xi_gt = log_se3(T_gt, eps=self.eps)  # (B,T,6)
        if self.detach_gt:
            xi_gt = xi_gt.detach()

        K = int(T_steps.shape[0])
        rho = self._rho(K, xi_gt.device, xi_gt.dtype)

        rw = batch.get('rweights', None) if self.use_weights else None
        tw = batch.get('tweights', None) if self.use_weights else None

        mask_index = batch.get('mask_index', None)
        if mask_index is not None:
            T_steps = T_steps[:, mask_index]
            xi_gt = xi_gt[mask_index]
            if rw is not None: rw = rw[mask_index]
            if tw is not None: tw = tw[mask_index]

        loss_acc = xi_gt.new_tensor(0.0)
        rot_acc = xi_gt.new_tensor(0.0)
        trans_acc = xi_gt.new_tensor(0.0)
        wsum = xi_gt.new_tensor(0.0)

        for k in range(K):
            frac = float(k + 1) / float(K)
            T_gt_k = exp_se3(xi_gt * frac)

            T_pred_k = T_steps[k]  # (B,T,4,4)
            T_err = invert_se3(T_pred_k) @ T_gt_k
            xi_err = log_se3(T_err, eps=self.eps)

            rot = torch.abs(xi_err[..., :3]).sum(dim=-1)
            trans = torch.abs(xi_err[..., 3:]).sum(dim=-1) / max(self.trans_scale, 1e-6)

            if rw is not None: rot = rot * rw
            if tw is not None: trans = trans * tw

            rot_m = rot.mean()
            trans_m = trans.mean()
            step_loss = (self.rot_w * rot_m) + (self.trans_w * trans_m)

            wk = rho[k]
            loss_acc += wk * step_loss
            rot_acc += wk * rot_m
            trans_acc += wk * trans_m
            wsum += wk

        loss = loss_acc / torch.clamp(wsum, min=1e-6)
        logs = {
            'enabled': 1.0,
            'K': float(K),
            'rot': float((rot_acc / torch.clamp(wsum, min=1e-6)).detach().item()),
            'trans': float((trans_acc / torch.clamp(wsum, min=1e-6)).detach().item()),
        }
        return loss, logs
    
# -----------------------------------------------------------------------------
# Additional clinically-motivated intermediate-stage constraints:
#   - interproximal_spacing: discourage excessive opening between adjacent teeth
#   - occlusal_plane: discourage drifting far away from the occlusal plane
#
# Required batch keys (add in train.py / val loss extra_fn):
#   batch['data']   : (B,T,P,3) input/start points
#   batch['center'] : (B,T,1,3) per-tooth centroid of input/start
#   batch['label']  : (B,T,P,3) GT/target points (for building refs)
# -----------------------------------------------------------------------------

def _ensure_center_4d(center: torch.Tensor) -> torch.Tensor:
    """Return (B,T,1,3) center tensor."""
    if center.dim() == 4:
        return center
    if center.dim() == 3:
        return center.unsqueeze(2)
    raise ValueError(f"center must be (B,T,1,3) or (B,T,3), got {tuple(center.shape)}")

def _subsample_points(pts: torch.Tensor, m: int) -> torch.Tensor:
    """Deterministic stride subsample along P dimension.
    pts: (B,T,P,3) -> (B,T,M,3)
    """
    if m <= 0:
        return pts
    B, T, P, _ = pts.shape
    if P <= m:
        return pts
    stride = max(P // m, 1)
    out = pts[:, :, ::stride, :]
    if out.shape[2] > m:
        out = out[:, :, :m, :]
    return out

def _build_T_from_quat_trans(quat_wxyz: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Row-convention SE(3) matrix from (quat, trans).
    quat: (B,T,4), trans:(B,T,3) -> (B,T,4,4)
    """
    R = quaternion_to_matrix(quat_wxyz)  # (B,T,3,3)
    B, T = trans.shape[:2]
    M = torch.eye(4, device=trans.device, dtype=trans.dtype).view(1, 1, 4, 4).repeat(B, T, 1, 1)
    M[..., :3, :3] = R
    M[..., 3, :3] = trans
    return M

def _get_T_steps(pred: Dict[str, torch.Tensor], K_virtual: int, eps: float = 1e-7) -> torch.Tensor:
    """Return (K,B,T,4,4) stage transforms.
    1) use pred['aux']['T_steps'] if present
    2) else: virtual geodesic steps I->T_final(pred)
    """
    aux = pred.get('aux', None)
    if isinstance(aux, dict) and ('T_steps' in aux):
        return aux['T_steps']

    K = int(K_virtual)
    if K <= 0:
        raise ValueError(f"K_virtual must be >0, got {K}")

    T_final = _build_T_from_quat_trans(pred['quat'], pred['trans'])
    xi = log_se3(T_final, eps=eps)  # (B,T,6)
    steps = []
    for k in range(K):
        frac = float(k + 1) / float(K)
        steps.append(exp_se3(xi * frac, eps=eps))
    return torch.stack(steps, dim=0)

def _adjacent_pair_min_dist(pts: torch.Tensor, *, softmin_tau: float = 0.0) -> torch.Tensor:
    """Min surface distance for adjacent pairs (i,i+1).
    pts: (B,T,M,3) -> (B,T-1)  (float32)
    """
    B, T, M, _ = pts.shape
    if T < 2:
        return pts.new_zeros((B, 0))

    A = pts[:, :-1].reshape(B * (T - 1), M, 3).float()
    Bp = pts[:, 1:].reshape(B * (T - 1), M, 3).float()

    dmat = torch.cdist(A, Bp)  # (B*(T-1),M,M)

    if softmin_tau and softmin_tau > 0.0:
        tau = float(softmin_tau)
        flat = dmat.reshape(dmat.shape[0], -1)
        dmin = -tau * torch.logsumexp(-flat / tau, dim=-1)
    else:
        dmin = dmat.amin(dim=(-1, -2))

    return dmin.reshape(B, T - 1)

def _step_weights(K: int, mode: str, device, dtype) -> torch.Tensor:
    mode = (mode or 'uniform').lower()
    if mode in ('uniform', 'const', 'none'):
        return torch.ones((K,), device=device, dtype=dtype)
    if mode in ('middle', 'mid', 'sin'):
        t = torch.linspace(0.0, 1.0, K, device=device, dtype=dtype)
        return torch.sin(torch.pi * t).clamp_min(0.0)
    if mode in ('linear',):
        return torch.linspace(1.0 / max(K, 1), 1.0, K, device=device, dtype=dtype)
    return torch.ones((K,), device=device, dtype=dtype)


@register_loss('interproximal_spacing')
class InterproximalSpacingTerm(_JawLoss):
    """Discourage excessive opening between adjacent teeth at intermediate stages."""

    def __init__(
        self,
        K_virtual: int = 21,
        n_points: int = 128,
        margin_mm: float = 0.5,
        softmin_tau: float = 0.0,
        skip_first: int = 0,
        skip_last: int = 1,
        step_weight: str = 'middle',
        eps: float = 1e-7,
    ):
        super().__init__()
        self.K_virtual = int(K_virtual)
        self.n_points = int(n_points)
        self.margin_mm = float(margin_mm)
        self.softmin_tau = float(softmin_tau)
        self.skip_first = int(skip_first)
        self.skip_last = int(skip_last)
        self.step_weight = str(step_weight)
        self.eps = float(eps)

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], ctx: Dict[str, Any]):
        if ('data' not in batch) or ('center' not in batch) or ('label' not in batch):
            z = pred['trans'].new_tensor(0.0)
            return z, {'enabled': 0.0}

        data0 = batch['data']          # (B,T,P,3)
        label = batch['label']         # (B,T,P,3)
        center = _ensure_center_4d(batch['center'])  # (B,T,1,3)

        data_s = _subsample_points(data0, self.n_points)
        gt_s   = _subsample_points(label, self.n_points)

        d_init = _adjacent_pair_min_dist(data_s, softmin_tau=self.softmin_tau).detach()
        d_gt   = _adjacent_pair_min_dist(gt_s,   softmin_tau=self.softmin_tau).detach()
        cap = torch.maximum(d_init, d_gt) + self.margin_mm   # (B,T-1)

        T_steps = _get_T_steps(pred, self.K_virtual, eps=self.eps)  # (K,B,T,4,4)
        K = int(T_steps.shape[0])
        k0 = max(self.skip_first, 0)
        k1 = K - max(self.skip_last, 0)
        if k1 <= k0:
            z = pred['trans'].new_tensor(0.0)
            return z, {'enabled': 0.0, 'reason': 1.0}

        w = _step_weights(K, self.step_weight, device=data0.device, dtype=data0.dtype)

        center32 = center.float()
        rel_s = (data_s - center).float()   # (B,T,M,3)

        loss_acc = data0.new_tensor(0.0)
        wsum = data0.new_tensor(0.0)
        last_d = None

        for k in range(k0, k1):
            Tk = T_steps[k]
            pts_k = apply_T_points(Tk.float(), rel_s) + center32  # (B,T,M,3)
            d_k = _adjacent_pair_min_dist(pts_k, softmin_tau=self.softmin_tau)  # (B,T-1)
            pen = F.relu(d_k - cap)
            lk = (pen * pen).mean()
            wk = w[k]
            loss_acc = loss_acc + wk * lk
            wsum = wsum + wk
            last_d = d_k

        loss = loss_acc / torch.clamp(wsum, min=1e-6)
        logs = {
            'enabled': 1.0,
            'K': float(K),
            'n_points': float(self.n_points),
            'margin_mm': float(self.margin_mm),
            'softmin_tau': float(self.softmin_tau),
            'd_init_mean': float(d_init.mean().item()) if d_init.numel() else 0.0,
            'd_gt_mean': float(d_gt.mean().item()) if d_gt.numel() else 0.0,
            'd_step_mean': float(last_d.mean().detach().item()) if (last_d is not None and last_d.numel()) else 0.0,
        }
        return loss, logs


@register_loss('occlusal_plane')
class OcclusalPlaneConsistencyTerm(_JawLoss):
    """Keep intermediate stages close to an estimated occlusal plane."""

    def __init__(
        self,
        K_virtual: int = 21,
        plane_from: str = 'gt',   # 'gt' | 'init' | 'avg'
        tol_mm: float = 1.5,
        skip_first: int = 0,
        skip_last: int = 1,
        step_weight: str = 'middle',
        eps: float = 1e-7,
    ):
        super().__init__()
        self.K_virtual = int(K_virtual)
        self.plane_from = str(plane_from)
        self.tol_mm = float(tol_mm)
        self.skip_first = int(skip_first)
        self.skip_last = int(skip_last)
        self.step_weight = str(step_weight)
        self.eps = float(eps)

    @staticmethod
    def _centroid(pts: torch.Tensor) -> torch.Tensor:
        return pts.mean(dim=2)  # (B,T,3)

    def _fit_plane(self, C: torch.Tensor):
        # C: (B,T,3)
        ctx = torch.cuda.amp.autocast(enabled=False) if C.is_cuda else nullcontext()
        with ctx:
            C32 = C.to(torch.float32)

            mu = C32.mean(dim=1, keepdim=True)      # (B,1,3)
            X  = C32 - mu                           # (B,T,3)

            cov = (X.transpose(1, 2) @ X) / max(C32.shape[1] - 1, 1)  # (B,3,3) float32
            evals, evecs = torch.linalg.eigh(cov)    # OK in float32
            n = evecs[..., 0]                        # (B,3) smallest eigenvector
            n = n / (_safe_norm(n, dim=-1, eps=self.eps).unsqueeze(-1))

        # 이후 dist 계산도 float32로 하는 게 안전/정확
        return mu, n

    def forward(self, pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], ctx: Dict[str, Any]):
        if ('center' not in batch) or ('label' not in batch):
            z = pred['trans'].new_tensor(0.0)
            return z, {'enabled': 0.0}

        center = _ensure_center_4d(batch['center'])     # (B,T,1,3)
        C_gt = self._centroid(batch['label'])           # (B,T,3)

        plane_from = self.plane_from.lower()
        if plane_from == 'init':
            C = self._centroid(batch['data']) if ('data' in batch) else C_gt
        elif plane_from == 'avg':
            if 'data' in batch:
                C = 0.5 * (self._centroid(batch['data']) + C_gt)
            else:
                C = C_gt
        else:
            C = C_gt

        mu, n = self._fit_plane(C)

        T_steps = _get_T_steps(pred, self.K_virtual, eps=self.eps)
        K = int(T_steps.shape[0])
        k0 = max(self.skip_first, 0)
        k1 = K - max(self.skip_last, 0)
        if k1 <= k0:
            z = pred['trans'].new_tensor(0.0)
            return z, {'enabled': 0.0, 'reason': 1.0}

        w = _step_weights(K, self.step_weight, device=center.device, dtype=center.dtype)

        c0 = center.squeeze(2).float()  # (B,T,3)
        loss_acc = center.new_tensor(0.0)
        wsum = center.new_tensor(0.0)
        last_mean = None

        for k in range(k0, k1):
            t_k = T_steps[k][..., 3, :3].float()  # (B,T,3)
            c_k = c0 + t_k
            dist = torch.abs(torch.sum((c_k - mu) * n.unsqueeze(1), dim=-1)).float()  # (B,T)
            pen = F.relu(dist - self.tol_mm)
            lk = (pen * pen).mean()
            wk = w[k]
            loss_acc = loss_acc + wk * lk
            wsum = wsum + wk
            last_mean = dist.mean()

        loss = loss_acc / torch.clamp(wsum, min=1e-6)
        logs = {
            'enabled': 1.0,
            'K': float(K),
            'tol_mm': float(self.tol_mm),
            'dist_mean': float(last_mean.detach().item()) if last_mean is not None else 0.0,
        }
        return loss, logs