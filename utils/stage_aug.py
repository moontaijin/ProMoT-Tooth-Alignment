# utils/stage_aug_compose.py
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from utils.geometry import apply_T_points


def case_id_from_path(p: str) -> int:
    m = re.search(r"data_(\d+)", p)
    if m is None:
        raise ValueError(f"Cannot parse case id from: {p}")
    return int(m.group(1))


@dataclass
class StageAugConfig:
    enabled: bool = True
    start_epoch: int = 30          # E부터 augmentation on
    prob: float = 0.5              # 배치 샘플별 적용 확률
    t_step: int = 4                # 1..K (보통 2..K-1 권장)
    cache_dtype: str = "float16"   # "float16" or "float32"


class StageTransformCache:
    """
    case_id -> (T,4,4) row-convention transform
    """
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        self.up: Dict[int, torch.Tensor] = {}
        self.down: Dict[int, torch.Tensor] = {}

    def clear(self):
        self.up.clear()
        self.down.clear()

    def swap_from(self, other: "StageTransformCache"):
        self.up = other.up
        self.down = other.down

    def set(self, jaw: str, case_ids: List[int], T_step: torch.Tensor):
        """
        jaw: 'up'/'down'
        case_ids: len B
        T_step: (B,T,4,4) on GPU OK
        """
        T_step = T_step.detach().to("cpu").to(self.dtype)
        d = self.up if jaw == "up" else self.down
        for i, cid in enumerate(case_ids):
            d[cid] = T_step[i].contiguous()

    def get(self, jaw: str, case_ids: List[int], device: torch.device, out_dtype: torch.dtype) -> Optional[torch.Tensor]:
        d = self.up if jaw == "up" else self.down
        mats = []
        for cid in case_ids:
            if cid not in d:
                return None
            mats.append(d[cid])
        return torch.stack(mats, dim=0).to(device=device, dtype=out_dtype)


# -------------------------
# Row-vector convention SE(3)
# T = [[R,0],
#      [t,1]] , t is a row vector
# p' = p @ T
# -------------------------
def se3_inv_row(T: torch.Tensor) -> torch.Tensor:
    """
    T: (...,4,4) row-vector convention
    inv(T) = [[R^T,0],
              [-t R^T, 1]]
    """
    R = T[..., :3, :3]
    t = T[..., 3:4, :3]              # (...,1,3)
    Rt = R.transpose(-2, -1)
    t_inv = -torch.matmul(t, Rt)     # (...,1,3)

    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., 3:4, :3] = t_inv
    out[..., 3, 3] = 1.0
    return out


def labels_to_T_row(gdofs: torch.Tensor, gtrans: torch.Tensor) -> torch.Tensor:
    """
    gdofs: (B,T,4) quaternion (w,x,y,z) in pytorch3d convention
    gtrans: (B,T,3) translation (row convention)
    -> T: (B,T,4,4)
    """
    R = quaternion_to_matrix(gdofs)  # (B,T,3,3)
    B, Tn = gdofs.shape[0], gdofs.shape[1]
    T = torch.zeros((B, Tn, 4, 4), device=gdofs.device, dtype=gdofs.dtype)
    T[..., :3, :3] = R
    T[..., 3, :3] = gtrans
    T[..., 3, 3] = 1.0
    return T


def T_to_labels_row(T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    R = T[..., :3, :3]
    gdofs = matrix_to_quaternion(R)  # (B,T,4)
    gtrans = T[..., 3, :3]           # (B,T,3)
    return gdofs, gtrans


def apply_stage_aug_compose(
    cfg: StageAugConfig,
    epoch: int,
    jaw: str,
    case_ids: List[int],
    cache_prev: StageTransformCache,
    start_pts: torch.Tensor,     # (B,T,P,3)  input points
    end_pts: torch.Tensor,       # (B,T,P,3)  label points (kept)
    centers: torch.Tensor,       # (B,T,1,3)
    gdofs_gt: torch.Tensor,      # (B,T,4)   original GT: 0->N
    gtrans_gt: torch.Tensor,     # (B,T,3)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compose 방식:
      pseudo_start = start_pts @ T0k_pred
      GT_kN = inv(T0k_pred) @ GT_0N

    returns:
      start_aug, end_pts, centers_aug, gdofs_aug, gtrans_aug, aug_mask(B)
    """
    B = start_pts.shape[0]
    dev = start_pts.device
    aug_mask = (torch.rand(B, device=dev) < float(cfg.prob))

    if (not cfg.enabled) or (epoch < cfg.start_epoch) or (not aug_mask.any()):
        return start_pts, end_pts, centers, gdofs_gt, gtrans_gt, aug_mask

    T0k_all = cache_prev.get(jaw, case_ids, device=dev, out_dtype=start_pts.dtype)  # (B,T,4,4)
    if T0k_all is None:
        # cache가 아직 덜 찼으면 augmentation skip
        return start_pts, end_pts, centers, gdofs_gt, gtrans_gt, aug_mask

    idx = torch.nonzero(aug_mask, as_tuple=False).squeeze(1)
    s = start_pts[idx]     # (b',T,P,3)
    T0k = T0k_all[idx]     # (b',T,4,4)

    # 1) pseudo-start 만들기
    pseudo = apply_T_points(T0k, s)  # (b',T,P,3)
    new_centers = pseudo.mean(dim=2, keepdim=True)

    # 2) GT compose: TkN = inv(T0k_pred) @ T0N_gt
    T0N = labels_to_T_row(gdofs_gt[idx], gtrans_gt[idx])  # (b',T,4,4)
    Tk0 = se3_inv_row(T0k)                                 # (b',T,4,4)
    TkN = torch.matmul(Tk0, T0N)                           # (b',T,4,4)

    gdofs_new, gtrans_new = T_to_labels_row(TkN)

    # 3) 배치에 반영
    start_out = start_pts.clone()
    centers_out = centers.clone()
    gdofs_out = gdofs_gt.clone()
    gtrans_out = gtrans_gt.clone()

    start_out[idx] = pseudo
    centers_out[idx] = new_centers
    gdofs_out[idx] = gdofs_new
    gtrans_out[idx] = gtrans_new

    return start_out, end_pts, centers_out, gdofs_out, gtrans_out, aug_mask


# ============================================================
# Micro-batch Stage Augmentation (B 방식)
#   - 원본 batch는 그대로 학습 (orig)
#   - 동일 batch에서 여러 step(k)로 pseudo-start를 만들어 추가 학습 (micro-batch 반복)
#   - GT는 compose 방식:  TkN_gt = inv(T0k_teacher) @ T0N_gt
#
# NOTE:
#   - teacher는 보통 "prev_epoch" (캐시) 또는 "current" (현재 batch의 aux) 중 선택
#   - K-step 누적 변환(T_steps)을 case_id 단위로 저장하는 캐시 제공
# ============================================================

from dataclasses import dataclass as _dataclass
from typing import Sequence as _Sequence, Any as _Any

@_dataclass
class StageAugMicroConfig:
    enabled: bool = False
    start_epoch: int = 30
    prob: float = 1.0
    steps: List[int] = None             # e.g., [2,4,6]
    max_pseudos: int = 0                # 0: use all steps; >0: sample this many from steps per batch
    alpha: float = 0.5                  # L = L_orig + alpha * mean(L_pseudo)
    cache_dtype: str = "float16"
    teacher: str = "prev_epoch"         # "prev_epoch" or "current"

    def normalized_steps(self, K: int) -> List[int]:
        if not self.steps:
            return []
        out = []
        for k in self.steps:
            try:
                kk = int(k)
            except Exception:
                continue
            if 1 <= kk <= K:
                out.append(kk)
        # unique & keep order
        seen = set()
        uniq = []
        for kk in out:
            if kk not in seen:
                uniq.append(kk); seen.add(kk)
        return uniq


class StageTransformCacheK:
    """Store per-case cumulative transforms for all K steps.
    Stored value per case:
      (K, N, 4, 4) row convention.
    """
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        self.up: Dict[int, torch.Tensor] = {}
        self.down: Dict[int, torch.Tensor] = {}

    def clear(self):
        self.up.clear()
        self.down.clear()

    def swap_from(self, other: "StageTransformCacheK"):
        self.up = other.up
        self.down = other.down

    def set_all(self, jaw: str, case_ids: List[int], T_steps: torch.Tensor):
        """Save all K steps for each case in the batch.
        T_steps: (K, B, N, 4, 4)
        """
        T_steps = T_steps.detach().to("cpu").to(self.dtype)
        d = self.up if jaw == "up" else self.down
        K, B = T_steps.shape[0], T_steps.shape[1]
        for i, cid in enumerate(case_ids):
            if i >= B:
                break
            d[cid] = T_steps[:, i].contiguous()   # (K,N,4,4)

    def has_both(self, case_ids: List[int]) -> List[bool]:
        return [(cid in self.up) and (cid in self.down) for cid in case_ids]

    def get_step(self, jaw: str, case_ids: List[int], k: int, device: torch.device, out_dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Return (B, N, 4, 4) for a given 1-indexed step k.
        If any case is missing, returns None.
        """
        d = self.up if jaw == "up" else self.down
        mats = []
        for cid in case_ids:
            if cid not in d:
                return None
            Tk = d[cid][k - 1]  # (N,4,4)
            mats.append(Tk)
        return torch.stack(mats, dim=0).to(device=device, dtype=out_dtype)


def compose_gt_from_teacher(
    T0k_teacher: torch.Tensor,  # (B,N,4,4)
    gdofs_gt: torch.Tensor,     # (B,N,4)
    gtrans_gt: torch.Tensor,    # (B,N,3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute composed GT labels for pseudo-start at step k.
    TkN_gt = inv(T0k_teacher) @ T0N_gt

    Returns:
      gdofs_new: (B,N,4)
      gtrans_new: (B,N,3)
    """
    T0N = labels_to_T_row(gdofs_gt, gtrans_gt)   # (B,N,4,4)
    Tk0 = se3_inv_row(T0k_teacher)               # (B,N,4,4)
    TkN = torch.matmul(Tk0, T0N)                 # (B,N,4,4)
    return T_to_labels_row(TkN)
