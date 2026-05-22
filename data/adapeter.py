# data/raw_otf_adapter.py
import os
from typing import List, Tuple, Any

import torch
from torch.utils.data import Dataset
from pytorch3d.transforms import matrix_to_quaternion

from utils.geometry import apply_T_points
from data.dataset import OnTheFlyDataset


def _make_jaw_pack(
    pts_in_k: torch.Tensor,   # (K,P,3) normalized
    pts_gt_k: torch.Tensor,   # (K,P,3) normalized
    T_k: torch.Tensor,        # (K,4,4) normalized, row-convention
    tooth_ids: torch.Tensor,  # (K,)
    jaw: str,                 # "up" or "down"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (T=16 fixed):
      data  : (16,P,3)
      label : (16,P,3)
      center: (16,1,3)
      gdofs : (16,4) quaternion
      gtrans: (16,3)
      tweights, rweights: (16,)
    """
    device = pts_in_k.device
    dtype = pts_in_k.dtype
    K, P, _ = pts_in_k.shape

    data = torch.zeros((16, P, 3), device=device, dtype=dtype)
    label = torch.zeros((16, P, 3), device=device, dtype=dtype)

    R = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3).repeat(16, 1, 1)

    # scatter K -> 16
    for i in range(K):
        tid = int(tooth_ids[i].item())
        if jaw == "up":
            if not (1 <= tid <= 16):
                continue
            j = tid - 1
        else:
            if not (17 <= tid <= 32):
                continue
            j = tid - 17

        data[j] = pts_in_k[i]
        label[j] = pts_gt_k[i]
        R[j] = T_k[i, :3, :3]

    # ---- mimic legacy loader behavior: center each jaw separately ----
    # input: subtract mean of ALL points in this jaw
    rc = data.reshape(-1, 3).mean(dim=0)
    data = data - rc

    # gt: subtract its own mean (legacy did this separately)
    gc = label.reshape(-1, 3).mean(dim=0)
    label = label - gc

    center = data.mean(dim=1, keepdim=True)              # (16,1,3)
    gtrans = label.mean(dim=1) - data.mean(dim=1)        # (16,3)
    gdofs = matrix_to_quaternion(R)                      # (16,4)

    # weights: keep simple = ones
    tweights = torch.ones((16,), device=device, dtype=dtype)
    rweights = torch.ones((16,), device=device, dtype=dtype)

    return data, label, center, gdofs, gtrans, tweights, rweights


class RawOTFCasePackDataset(Dataset):
    """
    Raw STL -> OnTheFly sampling (normalized) -> project pack format.

    IMPORTANT:
      - We DO NOT denormalize points here.
      - We return 'scale' (mm per normalized unit) so metrics can do mm conversion when raw_otf is on.
    """
    def __init__(
        self,
        raw_dir: str,
        case_list: List[str],
        *,
        n_points: int = 512,
        sampling_method: str = "fps",
        cache_meshes: bool = True,
        exclude_missing_teeth: bool = True,
        require_full_16_per_jaw: bool = True,
    ):
        self.raw_dir = raw_dir

        self.otf = OnTheFlyDataset(
            raw_dir=raw_dir,
            case_list=list(case_list),
            n_points=n_points,
            cache_meshes=False,
            sampling_method=sampling_method,
            augment=False,  # 프로젝트 augmentation은 따로 있으니 여기선 off 권장
            exclude_missing_teeth=exclude_missing_teeth,
        )

        # 안전하게: 기존 프로젝트가 jaw당 T=16 고정이라 full-set만 남기는 게 가장 안정적
        if require_full_16_per_jaw:
            kept = []
            for c in self.otf.cases:
                ids = [tid for (tid, _jaw, _p) in c["stl_items"]]
                ok_up = all(t in ids for t in range(1, 17))
                ok_dn = all(t in ids for t in range(17, 33))
                if ok_up and ok_dn:
                    kept.append(c)
            self.otf.cases = kept

    def __len__(self):
        return len(self.otf)

    def __getitem__(self, idx: int):
        s = self.otf[idx]

        pts_in = s["points_in"].float()      # (K,P,3) normalized
        T_gt = s["T_gt"].float()             # (K,4,4) normalized
        tooth_ids = s["tooth_id"].long()     # (K,)
        scale = s["scale"].float()           # scalar tensor (mm scaling)
        case_name = s.get("case_name", None) or self.otf.cases[idx]["name"]
        case_path = os.path.join(self.raw_dir, case_name)  # must contain "data_###" for stage_aug parsing

        # normalized GT points
        pts_gt = apply_T_points(T_gt.unsqueeze(0), pts_in.unsqueeze(0))[0]  # (K,P,3)

        pack_u_7 = _make_jaw_pack(pts_in, pts_gt, T_gt, tooth_ids, jaw="up")
        pack_l_7 = _make_jaw_pack(pts_in, pts_gt, T_gt, tooth_ids, jaw="down")

        return pack_u_7, pack_l_7, case_path, scale


def _stack_pack_7_to_pack_8(packs_7: List[Tuple[Any, ...]]) -> Tuple[torch.Tensor, ...]:
    # packs_7 element: (data,label,center,gdofs,gtrans,tweights,rweights)
    data, label, center, gdofs, gtrans, tweights, rweights = zip(*packs_7)
    B = len(packs_7)
    mask_index = torch.arange(B, dtype=torch.long)  # legacy code expects (B,)
    return (
        torch.stack(data, 0),
        torch.stack(label, 0),
        torch.stack(center, 0),
        torch.stack(gdofs, 0),
        torch.stack(gtrans, 0),
        torch.stack(tweights, 0),
        torch.stack(rweights, 0),
        mask_index,
    )


def collate_raw_otf_train(batch):
    """
    train loop expects: (pack_u, pack_l, down_paths)
    - we provide case_path list as 'down_paths' (stage_aug only needs to parse data_###)
    """
    packs_u_7, packs_l_7, case_paths, _scales = zip(*batch)
    pack_u = _stack_pack_7_to_pack_8(list(packs_u_7))
    pack_l = _stack_pack_7_to_pack_8(list(packs_l_7))
    return pack_u, pack_l, list(case_paths)


def collate_raw_otf_eval(batch):
    """
    evaluate() will accept: (pack_u, pack_l, scales)
    """
    packs_u_7, packs_l_7, _case_paths, scales = zip(*batch)
    pack_u = _stack_pack_7_to_pack_8(list(packs_u_7))
    pack_l = _stack_pack_7_to_pack_8(list(packs_l_7))
    scales = torch.stack([s.view(()) for s in scales], dim=0).float()  # (B,)
    return pack_u, pack_l, scales