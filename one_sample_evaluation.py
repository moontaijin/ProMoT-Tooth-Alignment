"""one_sample_evaluation.py

ProMoT 프로젝트에서 *한 케이스*만 inference/evaluation 하는 스크립트.

핵심 기능
  1) config YAML + ckpt(upper/lower) 로 모델을 로드
  2) 지정한 case_id(또는 down_end.npy 경로) 1개를 로드
  3) (u/l) 예측 quat/trans 및 (가능하면) intermediate T_steps를 추출
  4) TRE/ADD, rotation/translation error를 1샘플 기준으로 계산
  5) 시각화 노트북에서 바로 쓸 수 있도록 .npz/.json 로 결과 저장

주의
  - 이 파일은 VTK 없이 동작하도록, npy(dict) + toothMat.txt 를 직접 읽습니다.
  - 데이터 생성(data/data_processing.py) 규칙(결측치 치아는 tooth_center로 채움)을 그대로 가정합니다.

예시
  python one_sample_evaluation.py \
    --config_yaml outputs/all_0225/config_used.yaml \
    --ckpt_dir    outputs/all_0225/save_model \
    --split val --case_id 450 \
    --device cuda:0 \
    --out_root aligned_stl_pred/one_sample_eval

  # 또는 직접 down_end.npy를 지정
  python one_sample_evaluation.py \
    --config_yaml outputs/all_0225/config_used.yaml \
    --ckpt_dir    outputs/all_0225/save_model \
    --down_end_npy /path/to/data_450_down_end.npy
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

try:
    import yaml
except Exception as e:
    raise ImportError("pyyaml이 필요합니다. `pip install pyyaml` 후 다시 실행하세요.") from e

from pytorch3d.transforms import matrix_to_quaternion

# project imports (repo root 기준)
from models.teeth_model import teeth_arangement_model, teeth_arangement_kalman_ssm_model
from utils.util import Tooth_Assembler
from metric import tre_correspondence
import data.data_config as cfg


# -----------------------------
# helpers
# -----------------------------

def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_rel(base_dir: str, p: str) -> str:
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def _load_json(path: str) -> dict:
    if not path or (not os.path.exists(path)):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _case_id_from_any(x: str) -> int:
    """Extract numeric case id from strings like 'data_001', '.../data_001_down_end.npy'."""
    m = re.search(r"data_(\d+)", x)
    if m is None:
        raise ValueError(f"Cannot parse case id from: {x}")
    return int(m.group(1))


def _load_splits(splits_json_path: str, data_root_override: Optional[str] = None):
    with open(splits_json_path, "r", encoding="utf-8") as f:
        s = json.load(f)

    train_cases = s.get("train", [])
    val_cases = s.get("val", s.get("valid", []))
    test_cases = s.get("test", [])

    data_root = data_root_override or s.get("data_root", None)
    if data_root is None:
        raise ValueError("splits.json에 data_root가 없고, config.data_root도 지정되지 않았습니다.")

    return data_root, train_cases, val_cases, test_cases


def _find_down_end_path(data_root: str, case_id: int) -> str:
    """data_root 아래에서 data_{id}_down_end.npy 를 찾아 반환."""
    # 흔한 naming: data_450_down/data_450_down_end.npy
    pat1 = os.path.join(data_root, f"data_{case_id:03d}_down", f"data_{case_id:03d}_down_end.npy")
    pat2 = os.path.join(data_root, f"data_{case_id}_down", f"data_{case_id}_down_end.npy")
    pat3 = os.path.join(data_root, f"data_{case_id:03d}_down_end.npy")
    pat4 = os.path.join(data_root, f"data_{case_id}_down_end.npy")
    for p in (pat1, pat2, pat3, pat4):
        if os.path.exists(p):
            return p

    # fallback: walk
    target = f"data_{case_id:03d}_down_end.npy"
    for root, _, files in os.walk(data_root):
        if target in files:
            return os.path.join(root, target)
    target2 = f"data_{case_id}_down_end.npy"
    for root, _, files in os.walk(data_root):
        if target2 in files:
            return os.path.join(root, target2)

    raise FileNotFoundError(f"Cannot find down_end.npy for case_id={case_id} under data_root={data_root}")


def _load_state(model: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_dict}
    model_dict.update(filtered)
    model.load_state_dict(model_dict, strict=False)


def resolve_ckpt_pair(
    ckpt_dir: Optional[str] = None,
    ckpt_u: Optional[str] = None,
    ckpt_l: Optional[str] = None,
    epoch: Optional[str] = "last",
) -> Tuple[str, str]:
    """save_model 폴더에서 orth_model_{u/l}_*.pth 를 찾아 pair로 반환."""
    if ckpt_u and ckpt_l:
        return ckpt_u, ckpt_l

    if not ckpt_dir:
        raise ValueError("ckpt_dir 또는 (ckpt_u, ckpt_l) 중 하나를 제공해야 합니다.")

    import glob

    pat_u = os.path.join(ckpt_dir, "orth_model_u_*.pth")
    pat_l = os.path.join(ckpt_dir, "orth_model_l_*.pth")
    u_list = sorted(glob.glob(pat_u))
    l_list = sorted(glob.glob(pat_l))
    if len(u_list) == 0 or len(l_list) == 0:
        raise FileNotFoundError(f"Cannot find orth_model_u_*.pth / orth_model_l_*.pth in {ckpt_dir}")

    def _ep_num(p: str) -> int:
        m = re.search(r"_(\d+)\.pth$", p)
        return int(m.group(1)) if m else -1

    if epoch is None or str(epoch).lower() == "last":
        u = max(u_list, key=_ep_num)
        l = max(l_list, key=_ep_num)
        return u, l

    ep = int(epoch)
    u = os.path.join(ckpt_dir, f"orth_model_u_{ep}.pth")
    l = os.path.join(ckpt_dir, f"orth_model_l_{ep}.pth")
    return u, l


# -----------------------------
# data loading (VTK-free)
# -----------------------------


def _dict_get(d: dict, key: int, default=None):
    if key in d:
        return d[key]
    sk = str(key)
    if sk in d:
        return d[sk]
    return default


def _is_missing_tooth(points: np.ndarray, eps: float = 1e-8) -> bool:
    """data_processing.py에서 결측치 치아는 동일한 점을 512번 반복해서 채움."""
    if points.ndim != 2 or points.shape[-1] != 3:
        return True
    # 모든 점이 동일하면 missing으로 간주
    return float(np.max(np.std(points, axis=0))) < eps


def load_jaw_from_end_path(end_npy: str) -> Dict[str, Any]:
    """end_npy 하나로부터 (input=start, gt=end) + GT pose(6dof/translation) pack을 구성."""
    end_npy = os.path.abspath(end_npy)
    start_npy = end_npy.replace("end.npy", "start.npy")
    if not os.path.exists(start_npy):
        raise FileNotFoundError(f"start npy not found: {start_npy}")

    end_dict = np.load(end_npy, allow_pickle=True).item()
    start_dict = np.load(start_npy, allow_pickle=True).item()

    T = int(cfg.teeth_nums)   # 16
    P = int(cfg.sam_points)   # 512

    end_pts = []
    start_pts = []
    missing_mask = []
    for i in range(1, T + 1):
        pe = _dict_get(end_dict, i)
        ps = _dict_get(start_dict, i)
        if pe is None or ps is None:
            # 극단적 케이스: dict에 키 자체가 없으면 0으로 채움
            pe = np.zeros((P, 3), np.float32)
            ps = np.zeros((P, 3), np.float32)
        pe = np.asarray(pe, dtype=np.float32)
        ps = np.asarray(ps, dtype=np.float32)
        if pe.shape[0] != P:
            pe = pe[:P]
        if ps.shape[0] != P:
            ps = ps[:P]
        end_pts.append(pe)
        start_pts.append(ps)
        missing_mask.append(_is_missing_tooth(pe) and _is_missing_tooth(ps))

    end_pts = np.stack(end_pts, axis=0)      # (T,P,3)
    start_pts = np.stack(start_pts, axis=0)  # (T,P,3)
    valid_mask = ~np.array(missing_mask, dtype=bool)  # (T,)

    # ----- global centering (train_data_load과 동일) -----
    rcpoint = start_pts.reshape(-1, 3).mean(axis=0, keepdims=True)
    gcpoint = end_pts.reshape(-1, 3).mean(axis=0, keepdims=True)
    start_pts_c = start_pts - rcpoint
    end_pts_c = end_pts - gcpoint

    # per-tooth centers (start)
    centers = start_pts_c.mean(axis=1, keepdims=True)   # (T,1,3)

    # ----- GT rotation from toothMat.txt (train_data_load과 동일 파싱) -----
    toothmat = os.path.join(os.path.dirname(end_npy), "toothMat.txt")
    rms = np.tile(np.eye(3, dtype=np.float32)[None, ...], (T, 1, 1))
    if os.path.exists(toothmat):
        with open(toothmat, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                tid = parts[0]
                if tid not in cfg.INDEX:
                    continue
                idx = int(cfg.INDEX[tid]) - 1
                nums = parts[1:]
                if len(nums) == 9:
                    R = np.array(nums, dtype=np.float32).reshape(3, 3)
                elif len(nums) == 16:
                    M = np.array(nums, dtype=np.float32).reshape(4, 4)
                    R = M[:3, :3]
                else:
                    continue
                if 0 <= idx < T:
                    rms[idx] = R
    else:
        # toothMat이 없으면 identity로 두되 warning
        print(f"[WARN] toothMat.txt not found: {toothmat} (gt rotation=I)")

    gdofs = matrix_to_quaternion(torch.from_numpy(rms))  # (T,4)
    gdofs = gdofs.to(dtype=torch.float32)

    # ----- GT translation = (mean(end)-mean(start)) after centering (train_data_load과 동일) -----
    gtrans = (end_pts_c.mean(axis=1) - start_pts_c.mean(axis=1)).astype(np.float32)  # (T,3)

    pack = {
        "input": torch.from_numpy(start_pts_c).unsqueeze(0),   # (1,T,P,3)
        "gt": torch.from_numpy(end_pts_c).unsqueeze(0),        # (1,T,P,3)
        "center": torch.from_numpy(centers).unsqueeze(0),      # (1,T,1,3)
        "gdofs": gdofs.unsqueeze(0),                           # (1,T,4)
        "gtrans": torch.from_numpy(gtrans).unsqueeze(0),       # (1,T,3)
        "valid_mask": torch.from_numpy(valid_mask).unsqueeze(0),  # (1,T)
        "meta": {
            "end_npy": end_npy,
            "start_npy": start_npy,
            "toothMat": toothmat,
            "rcpoint": rcpoint.reshape(-1).tolist(),
            "gcpoint": gcpoint.reshape(-1).tolist(),
        },
    }
    return pack


# -----------------------------
# model build
# -----------------------------


@dataclass
class RuntimeCfg:
    encoder_name: str
    encoder_cfg: Dict[str, Any]
    ssm_name: str
    ssm_cfg: Dict[str, Any]
    splits_json: str
    data_root: Optional[str]


def build_runtime_cfg(cfg_dict: dict, yaml_path: str) -> RuntimeCfg:
    base_dir = os.path.dirname(os.path.abspath(yaml_path))
    data = cfg_dict.get("data", {}) or {}
    model = cfg_dict.get("model", {}) or {}
    enc = (model.get("encoder", {}) or {})
    ssm = (model.get("ssm", {}) or {})

    encoder_name = str(enc.get("name", "swin"))
    ssm_name = str(ssm.get("name", "kalman"))

    enc_cfg_path = _resolve_rel(base_dir, str(enc.get("config", "")) if enc else "")
    ssm_cfg_path = _resolve_rel(base_dir, str(ssm.get("config", "")) if ssm else "")

    encoder_cfg = _load_json(enc_cfg_path) if enc_cfg_path else {}
    ssm_cfg = _load_json(ssm_cfg_path) if ssm_cfg_path else {}

    return RuntimeCfg(
        encoder_name=encoder_name,
        encoder_cfg=encoder_cfg,
        ssm_name=ssm_name,
        ssm_cfg=ssm_cfg,
        splits_json=str(data.get("splits_json", "")),
        data_root=data.get("data_root", None),
    )


def build_models(runtime: RuntimeCfg, device: torch.device):
    enc_l = runtime.encoder_name.lower()
    ssm_l = runtime.ssm_name.lower()

    if ssm_l in ["none", "identity", "no", "false"]:
        model_u = teeth_arangement_model()
        model_l = teeth_arangement_model()
    else:
        if enc_l in ["pointnetpp", "pointnet++", "pn++", "pointnet2", "pn2"]:
            encoder_type = "pointnet++"
        elif enc_l in ["pointnext", "pnx", "pointnextencoder"]:
            encoder_type = "pointnext"
        else:
            encoder_type = "vit"

        out_dim = int(runtime.encoder_cfg.get("out_dim", 256) or 256)
        use_center = bool(runtime.encoder_cfg.get("use_center", True))

        # pointnext 옵션들
        pnx_k = int(runtime.encoder_cfg.get("k", 16) or 16)
        stage_points = runtime.encoder_cfg.get("stage_points", [128, 32, None])
        stage_dims = runtime.encoder_cfg.get("stage_dims", [64, 128, 256])
        expansion = int(runtime.encoder_cfg.get("expansion", 4) or 4)

        def _norm_list(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                return tuple([None if v is None else int(v) for v in x])
            return x

        model_u = teeth_arangement_kalman_ssm_model(
            tooth_id_offset=0,
            encoder_type=encoder_type,
            encoder_out_dim=out_dim,
            pointnet_use_center=use_center,
            ssm_cfg=runtime.ssm_cfg,
            pointnext_k=pnx_k,
            pointnext_stage_points=_norm_list(stage_points),
            pointnext_stage_dims=_norm_list(stage_dims),
            pointnext_expansion=expansion,
        )
        model_l = teeth_arangement_kalman_ssm_model(
            tooth_id_offset=16,
            encoder_type=encoder_type,
            encoder_out_dim=out_dim,
            pointnet_use_center=use_center,
            ssm_cfg=runtime.ssm_cfg,
            pointnext_k=pnx_k,
            pointnext_stage_points=_norm_list(stage_points),
            pointnext_stage_dims=_norm_list(stage_dims),
            pointnext_expansion=expansion,
        )

    model_u.to(device).eval()
    model_l.to(device).eval()
    return model_u, model_l, Tooth_Assembler(), Tooth_Assembler()


# -----------------------------
# metrics
# -----------------------------


def _axis_angle_deg_from_quat(q: torch.Tensor) -> torch.Tensor:
    # q: (...,4)
    # pytorch3d provides quaternion_to_axis_angle, but this repo also has one in data.utils.
    from data.utils import quaternion_to_axis_angle
    aa = quaternion_to_axis_angle(q)
    ang = torch.linalg.norm(aa, dim=-1) / np.pi * 180.0
    return ang


@torch.no_grad()
def eval_one_jaw(
    pack: Dict[str, Any],
    model: torch.nn.Module,
    assembler: Tooth_Assembler,
    device: torch.device,
    *,
    include_missing: bool = False,
    want_intermediates: bool = True,
):
    x = pack["input"].to(device).float()   # (1,T,P,3)
    y = pack["gt"].to(device).float()      # (1,T,P,3)
    c = pack["center"].to(device).float()  # (1,T,1,3)
    gdofs = pack["gdofs"].to(device).float()   # (1,T,4)
    gtrans = pack["gtrans"].to(device).float() # (1,T,3)
    vmask = pack["valid_mask"].to(device)
    if include_missing:
        vmask = torch.ones_like(vmask, dtype=torch.bool)

    # forward
    aux = None
    try:
        if want_intermediates:
            q_pred, t_pred, aux = model(x, c, return_intermediates=True)
        else:
            q_pred, t_pred = model(x, c)
    except TypeError:
        # baseline model
        q_pred, t_pred = model(x, c)

    assembled = assembler(x, c, q_pred, t_pred, device)  # (1,T,P,3)

    # TRE/ADD (correspondence 기반)
    tre = tre_correspondence(assembled, y, mask=vmask)

    # per-tooth ADD
    d = torch.linalg.norm(assembled - y, dim=-1)         # (1,T,P)
    per_tooth_add = d.mean(dim=-1).squeeze(0)            # (T,)

    # rot/trans errors (legacy metric.py와 동일 정의)
    ang_pred = _axis_angle_deg_from_quat(q_pred.squeeze(0))
    ang_gt = _axis_angle_deg_from_quat(gdofs.squeeze(0))
    rot_err = torch.abs(ang_pred - ang_gt)

    trans_err = torch.linalg.norm(t_pred.squeeze(0) - gtrans.squeeze(0), dim=-1)

    m = vmask.squeeze(0).float()
    denom = m.sum().clamp_min(1.0)
    mm_rot = float((rot_err * m).sum() / denom)
    mm_trans = float((trans_err * m).sum() / denom)

    # simple AUC over thresholds (0~AUC_K*AUC_piece)
    ths = torch.arange(0, cfg.AUC_K * cfg.AUC_piece + 1e-9, cfg.AUC_piece, device=device)
    accs = []
    for th in ths:
        ok = (per_tooth_add < th).float() * m
        acc = ok.sum() / denom
        accs.append(acc)
    auc = float(torch.stack(accs).mean())

    out = {
        "tre_mm": float(tre),
        "add_mm": float((per_tooth_add * m).sum() / denom),
        "auc": auc,
        "mm_rot_deg": mm_rot,
        "mm_trans_mm": mm_trans,
        "per_tooth": {
            "add_mm": per_tooth_add.detach().cpu().numpy().tolist(),
            "rot_deg": rot_err.detach().cpu().numpy().tolist(),
            "trans_mm": trans_err.detach().cpu().numpy().tolist(),
            "valid_mask": vmask.squeeze(0).detach().cpu().numpy().astype(bool).tolist(),
        },
        "pred": {
            "quat": q_pred.detach().cpu().numpy(),
            "trans": t_pred.detach().cpu().numpy(),
            "assembled": assembled.detach().cpu().numpy(),
        },
        "gt": {
            "quat": gdofs.detach().cpu().numpy(),
            "trans": gtrans.detach().cpu().numpy(),
            "points": y.detach().cpu().numpy(),
        },
        "input": {
            "points": x.detach().cpu().numpy(),
            "center": c.detach().cpu().numpy(),
        },
        "aux": None,
    }

    if aux is not None and isinstance(aux, dict):
        # T_steps: (K,B,N,4,4)
        if "T_steps" in aux:
            T_steps = aux["T_steps"].detach().cpu().numpy()   # (K,1,T,4,4)
            out["aux"] = {
                "T_steps": T_steps,
                "K": int(T_steps.shape[0]),
            }
    return out


def _apply_steps_to_points(x: np.ndarray, c: np.ndarray, T_steps: np.ndarray) -> np.ndarray:
    """x:(1,T,P,3), c:(1,T,1,3), T_steps:(K,1,T,4,4) -> traj:(K,T,P,3)"""
    x0 = torch.from_numpy(x).float()  # (1,T,P,3)
    c0 = torch.from_numpy(c).float()  # (1,T,1,3)
    K = T_steps.shape[0]
    traj = []
    for k in range(K):
        Tk = torch.from_numpy(T_steps[k]).float()  # (1,T,4,4)
        R = Tk[..., :3, :3]                        # (1,T,3,3)
        t = Tk[..., 3, :3]                         # (1,T,3)
        # (p-c)R + t + c
        p = x0 - c0
        p = torch.matmul(p, R)                     # broadcasted matmul
        p = p + t.unsqueeze(2) + c0
        traj.append(p.squeeze(0).numpy())          # (T,P,3)
    return np.stack(traj, axis=0)                  # (K,T,P,3)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=None, help="ProMoT repo root (PYTHONPATH용). 보통 생략 가능")
    ap.add_argument("--config_yaml", required=True, help="outputs/.../config_used.yaml 또는 config/*.yaml")

    ap.add_argument("--ckpt_dir", default=None, help="save_model 디렉토리 (orth_model_u_*.pth / orth_model_l_*.pth)")
    ap.add_argument("--ckpt_u", default=None)
    ap.add_argument("--ckpt_l", default=None)
    ap.add_argument("--epoch", default="last", help="'last' 또는 epoch 숫자")

    ap.add_argument("--split", default="val", choices=["train", "val", "test"], help="splits.json 기반 case 선택")
    ap.add_argument("--case_id", type=int, default=None, help="예: 450")
    ap.add_argument("--down_end_npy", default=None, help="직접 down_end.npy 경로를 지정")

    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_root", default="one_sample_outputs", help="결과 저장 루트")
    ap.add_argument("--include_missing", action="store_true", help="결측치 치아(채워진 tooth_center)도 metric에 포함")

    ap.add_argument("--save_points", action="store_true", help="input/gt/pred pointcloud 저장")
    ap.add_argument("--save_traj", action="store_true", help="T_steps로부터 trajectory pointcloud 저장")
    ap.add_argument("--traj_stride", type=int, default=1, help="trajectory 저장 stride (1=모든 step)")

    args = ap.parse_args()

    if args.repo_root:
        sys.path.insert(0, os.path.abspath(args.repo_root))

    device = torch.device(args.device)

    cfg_dict = _read_yaml(args.config_yaml)
    runtime = build_runtime_cfg(cfg_dict, args.config_yaml)

    # resolve case path
    if args.down_end_npy:
        down_end = os.path.abspath(args.down_end_npy)
        case_id = _case_id_from_any(down_end)
    else:
        if args.case_id is None:
            raise ValueError("--case_id 또는 --down_end_npy 중 하나는 반드시 필요합니다.")
        case_id = int(args.case_id)
        data_root, train_cases, val_cases, test_cases = _load_splits(runtime.splits_json, runtime.data_root)
        split_cases = {"train": train_cases, "val": val_cases, "test": test_cases}[args.split]
        # split에 해당 case가 없으면 그냥 root에서 찾기
        if len(split_cases) > 0:
            ids = set(_case_id_from_any(c) for c in split_cases)
            if case_id not in ids:
                print(f"[WARN] case_id={case_id} not in split='{args.split}' list. 그래도 data_root에서 탐색합니다.")
        down_end = _find_down_end_path(data_root, case_id)

    up_end = down_end.replace("_down", "_up")
    if not os.path.exists(up_end):
        raise FileNotFoundError(f"Cannot find corresponding up_end.npy: {up_end}")

    # build models
    model_u, model_l, assembler_u, assembler_l = build_models(runtime, device)

    # load ckpt
    ckpt_u_path, ckpt_l_path = resolve_ckpt_pair(args.ckpt_dir, args.ckpt_u, args.ckpt_l, args.epoch)
    print(f"[CKPT] u={ckpt_u_path}")
    print(f"[CKPT] l={ckpt_l_path}")
    _load_state(model_u, ckpt_u_path)
    _load_state(model_l, ckpt_l_path)

    # load data
    pack_u = load_jaw_from_end_path(up_end)
    pack_l = load_jaw_from_end_path(down_end)

    # eval
    out_u = eval_one_jaw(pack_u, model_u, assembler_u, device, include_missing=args.include_missing, want_intermediates=True)
    out_l = eval_one_jaw(pack_l, model_l, assembler_l, device, include_missing=args.include_missing, want_intermediates=True)

    # aggregate
    metrics = {
        "case_id": int(case_id),
        "paths": {
            "up_end": up_end,
            "down_end": down_end,
        },
        "upper": {
            **{k: out_u[k] for k in ["tre_mm", "add_mm", "auc", "mm_rot_deg", "mm_trans_mm"]},
            "per_tooth": out_u.get("per_tooth", {}),
        },
        "lower": {
            **{k: out_l[k] for k in ["tre_mm", "add_mm", "auc", "mm_rot_deg", "mm_trans_mm"]},
            "per_tooth": out_l.get("per_tooth", {}),
        },
        "mean": {
            "tre_mm": 0.5 * (out_u["tre_mm"] + out_l["tre_mm"]),
            "add_mm": 0.5 * (out_u["add_mm"] + out_l["add_mm"]),
            "auc": 0.5 * (out_u["auc"] + out_l["auc"]),
            "mm_rot_deg": 0.5 * (out_u["mm_rot_deg"] + out_l["mm_rot_deg"]),
            "mm_trans_mm": 0.5 * (out_u["mm_trans_mm"] + out_l["mm_trans_mm"]),
        },
    }

    print("[ONE-SAMPLE]",
          f"case={case_id} |",
          f"TRE={metrics['mean']['tre_mm']:.4f}mm",
          f"ADD={metrics['mean']['add_mm']:.4f}mm",
          f"AUC={metrics['mean']['auc']:.4f}",
          f"MMrot={metrics['mean']['mm_rot_deg']:.4f}deg",
          f"MMtrans={metrics['mean']['mm_trans_mm']:.4f}mm")

    # save
    case_dir = os.path.join(args.out_root, f"data_{case_id:03d}")
    _ensure_dir(case_dir)

    with open(os.path.join(case_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # meta
    meta = {
        "runtime": {
            "encoder": runtime.encoder_name,
            "ssm": runtime.ssm_name,
            "encoder_cfg": runtime.encoder_cfg,
            "ssm_cfg": runtime.ssm_cfg,
        },
        "data": {
            "upper": pack_u["meta"],
            "lower": pack_l["meta"],
        },
        "ckpt": {"u": ckpt_u_path, "l": ckpt_l_path},
    }
    with open(os.path.join(case_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # always save T_steps if exists
    T_steps_u = out_u.get("aux", {}) and out_u["aux"].get("T_steps", None)
    T_steps_l = out_l.get("aux", {}) and out_l["aux"].get("T_steps", None)

    save_npz = {
        "T_steps_u": T_steps_u if T_steps_u is not None else np.zeros((0,), np.float32),
        "T_steps_l": T_steps_l if T_steps_l is not None else np.zeros((0,), np.float32),
        "valid_mask_u": np.array(out_u["per_tooth"]["valid_mask"], dtype=bool),
        "valid_mask_l": np.array(out_l["per_tooth"]["valid_mask"], dtype=bool),
    }

    if args.save_points:
        save_npz.update({
            "input_u": out_u["input"]["points"],
            "gt_u": out_u["gt"]["points"],
            "pred_u": out_u["pred"]["assembled"],
            "input_l": out_l["input"]["points"],
            "gt_l": out_l["gt"]["points"],
            "pred_l": out_l["pred"]["assembled"],
        })

    if args.save_traj and (T_steps_u is not None) and (T_steps_l is not None):
        stride = max(1, int(args.traj_stride))
        T_u = T_steps_u[::stride]
        T_l = T_steps_l[::stride]
        traj_u = _apply_steps_to_points(out_u["input"]["points"], out_u["input"]["center"], T_u)
        traj_l = _apply_steps_to_points(out_l["input"]["points"], out_l["input"]["center"], T_l)
        save_npz.update({
            "traj_u": traj_u,
            "traj_l": traj_l,
            "traj_stride": np.array([stride], dtype=np.int32),
        })

    np.savez_compressed(os.path.join(case_dir, "result.npz"), **save_npz)
    print(f"[SAVED] {case_dir}")


if __name__ == "__main__":
    main()
