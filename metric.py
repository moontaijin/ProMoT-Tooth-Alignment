from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union, List

import numpy as np
import torch

from data.utils import quaternion_to_axis_angle

Array = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: Array) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def _check_shapes(points_pred: np.ndarray, points_gt: np.ndarray):
    if points_pred.shape != points_gt.shape:
        raise ValueError(f"shape mismatch: pred={points_pred.shape}, gt={points_gt.shape}")
    if points_pred.ndim not in (3, 4):
        raise ValueError(f"points must be (N,P,3) or (B,N,P,3). got {points_pred.shape}")
    if points_pred.shape[-1] != 3:
        raise ValueError(f"last dim must be 3. got {points_pred.shape}")


def _maybe_expand_batch(x: np.ndarray) -> np.ndarray:
    # make (B,N,P,3)
    if x.ndim == 3:
        return x[None, ...]
    return x


def _apply_mask_bn(x: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Validate and normalize mask to (B,N) bool."""
    if mask is None:
        return x, None
    mask = mask.astype(bool)
    if mask.ndim == 1:
        mask = mask[None, :]
    if mask.shape[0] != x.shape[0] or mask.shape[1] != x.shape[1]:
        raise ValueError(f"mask shape {mask.shape} incompatible with x {x.shape}")
    return x, mask


def tre_correspondence(points_pred: Array,
                       points_gt: Array,
                       mask: Optional[Array] = None,
                       reduce: str = "mean") -> float:
    """Target Registration Error (TRE) with point-to-point correspondence.

    points_pred, points_gt:
      - (N,P,3) or (B,N,P,3)
    mask:
      - (N,) or (B,N) boolean, True for valid teeth.

    Returns scalar TRE:
      mean over points, then mean over teeth(valid only), then mean over batch.
    """
    pp = _to_numpy(points_pred).astype(np.float32)
    pg = _to_numpy(points_gt).astype(np.float32)
    _check_shapes(pp, pg)

    pp = _maybe_expand_batch(pp)
    pg = _maybe_expand_batch(pg)

    mask_np = None if mask is None else _to_numpy(mask)
    _, mask_np = _apply_mask_bn(pp, mask_np)

    d = np.linalg.norm(pp - pg, axis=-1)          # (B,N,P)
    per_tooth = d.mean(axis=-1)                   # (B,N)

    if mask_np is not None:
        valid = mask_np
        denom = np.maximum(valid.sum(axis=1), 1)
        per_case = (per_tooth * valid).sum(axis=1) / denom
    else:
        per_case = per_tooth.mean(axis=1)

    if reduce == "mean":
        return float(per_case.mean())
    if reduce == "median":
        return float(np.median(per_case))
    raise ValueError("reduce must be 'mean' or 'median'")


# -----------------------------
# Pretty printing / aggregation
# -----------------------------
DEFAULT_METRIC_ORDER = [
    "loss",
    "add_mm", "tre_mm",
    "auc",
    "mm_trans_mm", "mm_rot_deg",
    "me_trans_mm", "me_rot_deg",
]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer))


def format_metrics_line(metrics: Dict[str, Any],
                        *,
                        order: Optional[List[str]] = None,
                        exclude: Optional[Set[str]] = None,
                        rename: Optional[Dict[str, str]] = None,
                        precision: int = 4) -> str:
    """Convert metric dict to a single-line string.

    - order keys appear first if present
    - remaining numeric keys are appended in sorted order
    """
    if metrics is None:
        return ""
    order = order or DEFAULT_METRIC_ORDER
    exclude = exclude or set()
    rename = rename or {}

    parts: List[str] = []
    used: Set[str] = set()

    for k in order:
        if k in metrics and k not in exclude and _is_number(metrics[k]):
            kk = rename.get(k, k)
            parts.append(f"{kk}={float(metrics[k]):.{precision}f}")
            used.add(k)

    rest = sorted([k for k, v in metrics.items() if k not in used and k not in exclude and _is_number(v)])
    for k in rest:
        kk = rename.get(k, k)
        parts.append(f"{kk}={float(metrics[k]):.{precision}f}")

    return " ".join(parts)


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, v: float, n: int = 1):
        self.total += float(v) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


@dataclass
class MetricAverager:
    meters: Dict[str, AverageMeter] = field(default_factory=dict)

    def update(self, metrics: Dict[str, Any], n: int = 1, exclude: Optional[Set[str]] = None):
        if metrics is None:
            return
        exclude = exclude or set()
        for k, v in metrics.items():
            if k in exclude:
                continue
            if _is_number(v):
                if k not in self.meters:
                    self.meters[k] = AverageMeter()
                self.meters[k].update(float(v), n=n)

    def as_dict(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}


# -----------------------------
# Main evaluation
# -----------------------------

def _autocast_ctx(device: torch.device):
    # lightweight autocast helper (no dependency on train.py)
    try:
        from torch.amp import autocast as _autocast
        if device.type == "cuda":
            return _autocast(device_type="cuda")
    except Exception:
        pass
    try:
        from torch.cuda.amp import autocast
        if device.type == "cuda":
            return autocast()
    except Exception:
        pass
    # CPU / no AMP
    class _Null:
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc, tb):
            return False
    return _Null()


@torch.no_grad()
def evaluate(
    loader,
    model_u,
    model_l,
    tooth_assembler_u,
    tooth_assembler_l,
    device: Union[str, torch.device],
    *,
    extra_fn: Optional[Callable[..., Dict[str, float]]] = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    if isinstance(device, str):
        device = torch.device(device)

    model_u.eval()
    model_l.eval()

    avg = MetricAverager()
    for batch in loader:
        # --------------------------------------------
        # Support:
        #  - processed: (pack_u, pack_l)
        #  - raw_otf  : (pack_u, pack_l, scales)
        # --------------------------------------------
        scales = None
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            pack_u, pack_l = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 3:
            pack_u, pack_l, scales = batch
        else:
            raise ValueError("loader must yield (pack_u, pack_l) or (pack_u, pack_l, scales)")

        (data_u, label_u, center_u, gdofs_u, gtrans_u, tweights_u, rweights_u, mask_index_u) = pack_u
        (data_l, label_l, center_l, gdofs_l, gtrans_l, tweights_l, rweights_l, mask_index_l) = pack_l

        data_u = data_u.to(device, non_blocking=True).float()
        data_l = data_l.to(device, non_blocking=True).float()
        label_u = label_u.to(device, non_blocking=True).float()
        label_l = label_l.to(device, non_blocking=True).float()
        center_u = center_u.to(device, non_blocking=True).float()
        center_l = center_l.to(device, non_blocking=True).float()
        gdofs_u = gdofs_u.to(device, non_blocking=True).float()
        gdofs_l = gdofs_l.to(device, non_blocking=True).float()
        gtrans_u = gtrans_u.to(device, non_blocking=True).float()
        gtrans_l = gtrans_l.to(device, non_blocking=True).float()
        tweights_u = tweights_u.to(device, non_blocking=True).float()
        tweights_l = tweights_l.to(device, non_blocking=True).float()
        rweights_u = rweights_u.to(device, non_blocking=True).float()
        rweights_l = rweights_l.to(device, non_blocking=True).float()
        mask_index_u = mask_index_u.to(device, non_blocking=True).long()
        mask_index_l = mask_index_l.to(device, non_blocking=True).long()

        B = data_u.shape[0]

        ctx = _autocast_ctx(device) if use_amp else _autocast_ctx(torch.device("cpu"))
        with ctx:
            pdofs_u, ptrans_u = model_u(data_u, center_u)
            pdofs_l, ptrans_l = model_l(data_l, center_l)

            assembled_u = tooth_assembler_u(data_u, center_u, pdofs_u, ptrans_u, device)
            assembled_l = tooth_assembler_l(data_l, center_l, pdofs_l, ptrans_l, device)

        # --------------------------------------------
        # raw_otf metric denorm (ONLY here)
        # normalized -> mm : multiply by 'scale'
        # --------------------------------------------
        if scales is not None:
            scales = scales.to(device, non_blocking=True).float()  # (B,)
            s_pts = scales.view(B, 1, 1, 1)  # (B,1,1,1) for points
            s_tr  = scales.view(B, 1, 1)     # (B,1,1) for (B,T,3)

            tre_u = tre_correspondence(assembled_u * s_pts, label_u * s_pts)
            tre_l = tre_correspondence(assembled_l * s_pts, label_l * s_pts)

            ptrans_u_m = ptrans_u * s_tr
            ptrans_l_m = ptrans_l * s_tr
            gtrans_u_m = gtrans_u * s_tr
            gtrans_l_m = gtrans_l * s_tr
        else:
            tre_u = tre_correspondence(assembled_u, label_u)
            tre_l = tre_correspondence(assembled_l, label_l)

            ptrans_u_m = ptrans_u
            ptrans_l_m = ptrans_l
            gtrans_u_m = gtrans_u
            gtrans_l_m = gtrans_l

        tre = 0.5 * (tre_u + tre_l)

        angle_pre_u = torch.norm(quaternion_to_axis_angle(pdofs_u[mask_index_u]), dim=-1) / math.pi * 180.0
        angle_pre_l = torch.norm(quaternion_to_axis_angle(pdofs_l[mask_index_l]), dim=-1) / math.pi * 180.0
        angle_gt_u  = torch.norm(quaternion_to_axis_angle(gdofs_u[mask_index_u]), dim=-1) / math.pi * 180.0
        angle_gt_l  = torch.norm(quaternion_to_axis_angle(gdofs_l[mask_index_l]), dim=-1) / math.pi * 180.0

        mmrot_u = torch.mean(torch.abs(angle_pre_u - angle_gt_u))
        mmrot_l = torch.mean(torch.abs(angle_pre_l - angle_gt_l))
        mmrot = 0.5 * (mmrot_u + mmrot_l)

        # NOTE: keep original aggregation style, but using scaled translations when raw_otf
        mmtrans_u = torch.mean(torch.norm(ptrans_u_m[mask_index_u] - gtrans_u_m[mask_index_u], dim=1))
        mmtrans_l = torch.mean(torch.norm(ptrans_l_m[mask_index_l] - gtrans_l_m[mask_index_l], dim=1))
        mmtrans = 0.5 * (mmtrans_u + mmtrans_l)

        out: Dict[str, float] = {
            "tre_mm": float(tre),
            "mm_rot_deg": float(mmrot),
            "mm_trans_mm": float(mmtrans),
        }

        if extra_fn is not None:
            extra = extra_fn(
                pack_u, pack_l,
                pdofs_u, ptrans_u,
                pdofs_l, ptrans_l,
                assembled_u, assembled_l,
                device,
            )
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if v is None:
                        continue
                    if _is_number(v):
                        out[k] = float(v)

        avg.update(out, n=B)

    return avg.as_dict()
