from __future__ import print_function
import os
import shutil
# os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_divice_id
import time
import argparse
import torch
import torch.nn as nn

try:
    from torch.amp import autocast as _autocast, GradScaler
    def autocast():
        return _autocast(device_type="cuda")
except Exception:
    from torch.cuda.amp import autocast, GradScaler

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import json
import yaml
from torch.utils.data import Dataset
import glob

from data.load_train_data import TrainData, train_data_load
from data.utils import quaternion_to_axis_angle
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from models.teeth_model import teeth_arangement_model, teeth_arangement_kalman_ssm_model
from utils.util import IOStream, Tooth_Assembler
from utils.geometry import apply_T_points
from loss.composer import LossComposer
from metric import MetricAverager
from utils.stage_aug import case_id_from_path, StageAugMicroConfig, StageTransformCacheK, compose_gt_from_teacher
from pytorch3d.transforms import *

# from pytorch3d.transforms import quaternion_to_axis_angle
import math
import re


# -------------------------
# Data loading helpers
# -------------------------

def collate_sttalign(batch_paths):
    """batch_paths: list[str] (down_end.npy paths) -> (pack_u, pack_l)"""
    cafh_l = list(batch_paths)
    cafh_u = [p.replace("_down", "_up") for p in cafh_l]

    pack_u = train_data_load(cafh_u)
    pack_l = train_data_load(cafh_l)
    return pack_u, pack_l


def collate_sttalign_train(batch_paths):
    """train_loader 전용: down path list도 같이 반환."""
    cafh_l = list(batch_paths)
    cafh_u = [p.replace("_down", "_up") for p in cafh_l]
    pack_u = train_data_load(cafh_u)
    pack_l = train_data_load(cafh_l)
    return pack_u, pack_l, cafh_l


class DownPathDataset(Dataset):
    """DataLoader가 batch로 down 경로(list[str])를 뱉도록 하는 Dataset."""
    def __init__(self, down_paths):
        self.down_paths = list(down_paths)

    def __len__(self):
        return len(self.down_paths)

    def __getitem__(self, idx):
        return self.down_paths[idx]


def _case_id(x: str) -> int:
    """Extract numeric case id from strings like 'data_001', '.../data_001_down_end.npy'."""
    m = re.search(r"data_(\d+)", x)
    if m is None:
        raise ValueError(f"Cannot parse case id from: {x}")
    return int(m.group(1))


def _filter_by_cases(all_down_end_paths, case_names):
    ids = set(_case_id(c) for c in case_names)
    return [p for p in all_down_end_paths if _case_id(p) in ids]


def _load_splits(splits_json_path, data_root_override=None):
    with open(splits_json_path, "r", encoding="utf-8") as f:
        s = json.load(f)

    # key 호환: val / valid 둘 다 지원
    train_cases = s.get("train", [])
    val_cases = s.get("val", s.get("valid", []))
    test_cases = s.get("test", [])

    data_root = data_root_override or s.get("data_root", None)
    if data_root is None:
        raise ValueError("splits.json에 data_root가 없고, config.data_root도 지정되지 않았습니다.")

    return data_root, train_cases, val_cases, test_cases

# -------------------------
# Log helpers
# -------------------------

def _resolve_down_path(data_root: str, case_name: str):
    """Resolve a case name to an existing down path/file under data_root."""
    cand = case_name
    if not os.path.isabs(cand):
        cand = os.path.join(data_root, cand)
    if os.path.exists(cand):
        return cand

    base = os.path.join(data_root, case_name)
    for suf in ["_down", "_down_end.npy", "_down.npy"]:
        p = base + suf
        if os.path.exists(p):
            return p

    if case_name.endswith("_up"):
        base2 = os.path.join(data_root, case_name[:-3])
        for suf in ["_down", "_down_end.npy", "_down.npy"]:
            p = base2 + suf
            if os.path.exists(p):
                return p

    raise FileNotFoundError(f"Cannot resolve down path for case='{case_name}' under data_root='{data_root}'")

def _to_float(x):
    try:
        if hasattr(x, "detach"):
            return float(x.detach().item())
        return float(x)
    except Exception:
        return None


def _is_top_level_loss_key(k):
    # 'loss/<name>'만 (예: loss/trans). 'loss/trans/u' 같은 상세 키는 제외
    return isinstance(k, str) and k.startswith("loss/") and (k.count("/") == 1)


def _fmt_delta(cur, prev, precision=4):
    d = cur - prev
    arrow = "↓" if d < 0 else ("↑" if d > 0 else "→")
    return f"({arrow}{d:+.{precision}f})"


def format_loss_terms_inline(logs, prev_top=None, precision=4, max_terms=0, with_delta=True):
    """
    logs: MetricAverager.as_dict() 결과(dict)
    prev_top: 이전 epoch의 top-level loss dict (delta용)
    return: (line_str, cur_top_all_dict)
    """
    cur_all = {}
    for k, v in (logs or {}).items():
        if not _is_top_level_loss_key(k):
            continue
        fv = _to_float(v)
        if fv is None:
            continue
        cur_all[k] = fv

    # total은 train_loss로 이미 찍으니 제외
    cur_all.pop("loss/total", None)

    items = list(cur_all.items())
    if max_terms and max_terms > 0 and len(items) > max_terms:
        # 절댓값 큰 순으로 상위 K개만
        items.sort(key=lambda kv: -abs(kv[1]))
        items = items[:max_terms]
    else:
        # 안정적으로 이름순
        items.sort(key=lambda kv: kv[0])

    parts = []
    for k, val in items:
        name = k.split("/", 1)[1]  # loss/<name> -> <name>
        if with_delta and (prev_top is not None) and (k in prev_top):
            parts.append(f"{name}={val:.{precision}f}{_fmt_delta(val, prev_top[k], precision)}")
        else:
            parts.append(f"{name}={val:.{precision}f}")
    return " | ".join(parts), cur_all


# Metrics are moved to metric.py
from metric import evaluate, format_metrics_line


def save_config_snapshot(config_path: str, output_dir: str, cfg_dict: dict):
    """Save run configs into outputs/<exp>/ for reproducibility.

    Saves:
      - config_used.yaml      : original YAML text
      - config_resolved.yaml  : parsed YAML dict dump
      - config_json/*.json    : referenced encoder/ssm/decoder JSON configs (if any)
    """
    os.makedirs(output_dir, exist_ok=True)
    used_yaml_path = os.path.join(output_dir, "config_used.yaml")
    resolved_yaml_path = os.path.join(output_dir, "config_resolved.yaml")

    config_path_abs = os.path.abspath(config_path)
    base_dir = os.path.dirname(config_path_abs)

    # 1) YAML raw snapshot
    with open(config_path_abs, "r", encoding="utf-8") as f:
        raw = f.read()
    with open(used_yaml_path, "w", encoding="utf-8") as f:
        f.write(raw)

    # 2) resolved YAML (dict dump)
    with open(resolved_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=True)

    # 3) copy referenced JSONs (encoder/ssm/decoder)
    try:
        model_cfg = (cfg_dict or {}).get("model", {}) or {}
        json_paths = []
        for part in ("encoder", "ssm", "decoder"):
            part_cfg = (model_cfg.get(part, {}) or {})
            p = part_cfg.get("config", "") or ""
            if not p:
                continue
            # resolve relative to YAML location
            p_abs = p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))
            if os.path.exists(p_abs):
                json_paths.append((part, p_abs))
        if json_paths:
            dst_dir = os.path.join(output_dir, "config_json")
            os.makedirs(dst_dir, exist_ok=True)
            for part, p_abs in json_paths:
                # keep filename, and also prefix with part to avoid collision
                fname = os.path.basename(p_abs)
                dst = os.path.join(dst_dir, f"{part}__{fname}")
                shutil.copy2(p_abs, dst)
    except Exception as e:
        # Do not crash training due to snapshot failures
        pass

def model_initial(model, model_name):
    pretrained_dict = torch.load(model_name)["model"]

    model_dict = model.state_dict()
    pretrained_dictf = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dictf)
    model.load_state_dict(model_dict)

    print("model initial over")



def _init_(args):
    """Create output directories. Kept minimal to avoid changing training logic."""
    out_dir = getattr(args, 'output_dir', None) or os.path.join('outputs', args.exp_name)
    args.output_dir = out_dir
    os.makedirs('outputs', exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)


def train(args, io, cfg_dict):
    device = args.device
    K_steps = int(getattr(args, 'ssm_cfg', {}).get('K', 8))

    # -------------------------
    # Build loaders (processed vs raw_otf)
    # -------------------------
    if args.splits_json is None or args.splits_json == "":
        raise ValueError("train/valid/test를 쓰려면 --splits_json 경로를 반드시 지정해야 합니다.")
    
    data_root, train_cases, val_cases, test_cases = _load_splits(args.splits_json, args.data_root)
    
    train_loader = None
    val_loader = None
    test_loader = None

    if not getattr(args, "raw_otf", False):
        # -------- processed pipeline (legacy) --------
        all_down_end = TrainData(data_root).train_list
        train_list = _filter_by_cases(all_down_end, train_cases)
        val_list   = _filter_by_cases(all_down_end, val_cases) if len(val_cases) > 0 else []
        test_list  = _filter_by_cases(all_down_end, test_cases) if len(test_cases) > 0 else []

        io.cprint(f"[SPLIT] processed data_root={data_root}")
        io.cprint(f"[SPLIT] #all={len(all_down_end)} #train={len(train_list)} #val={len(val_list)} #test={len(test_list)}")

        train_loader = DataLoader(
            DownPathDataset(train_list),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2,
            collate_fn=collate_sttalign_train,
        )

        if len(val_list) > 0:
            val_loader = DataLoader(
                DownPathDataset(val_list),
                batch_size=args.test_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=2,
                collate_fn=collate_sttalign,
            )

        if len(test_list) > 0:
            test_loader = DataLoader(
                DownPathDataset(test_list),
                batch_size=args.test_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=2,
                collate_fn=collate_sttalign,
            )

    else:
        # -------- raw OTF pipeline --------
        raw_dir = (getattr(args, "raw_dir", "") or "").strip() or data_root

        from data.adapeter import RawOTFCasePackDataset, collate_raw_otf_train, collate_raw_otf_eval

        io.cprint(f"[SPLIT] raw_otf raw_dir={raw_dir}")
        io.cprint(f"[SPLIT] #train_cases={len(train_cases)} #val_cases={len(val_cases)} #test_cases={len(test_cases)}")

        train_ds = RawOTFCasePackDataset(
            raw_dir, train_cases,
            n_points=args.num_points,
            sampling_method="fps",
            cache_meshes=True,
            exclude_missing_teeth=False,
            require_full_16_per_jaw=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2,
            collate_fn=collate_raw_otf_train,
        )

        if len(val_cases) > 0:
            val_ds = RawOTFCasePackDataset(
                raw_dir, val_cases,
                n_points=args.num_points,
                sampling_method="fps",
                cache_meshes=True,
                exclude_missing_teeth=False,
                require_full_16_per_jaw=False,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=args.test_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=2,
                collate_fn=collate_raw_otf_eval,   # <-- scale이 같이 나옴
            )

        if len(test_cases) > 0:
            test_ds = RawOTFCasePackDataset(
                raw_dir, test_cases,
                n_points=args.num_points,
                sampling_method="fps",
                cache_meshes=True,
                exclude_missing_teeth=False,
                require_full_16_per_jaw=False,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=args.test_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                prefetch_factor=2,
                collate_fn=collate_raw_otf_eval,  # <-- scale이 같이 나옴
            )

    device = torch.device("cuda" if args.cuda else "cpu")

    # 
    # -------------------------
    # Build models from config/args (encoder / ssm / decoder)
    # -------------------------
    encoder_name = getattr(args, 'encoder', None) or getattr(args, 'model', 'swin')
    ssm_name = getattr(args, 'ssm', 'kalman')
    decoder_name = getattr(args, 'decoder', 'baseline')

    enc_l = str(encoder_name).lower()
    ssm_l = str(ssm_name).lower()

    io.cprint(f"[MODEL] encoder={encoder_name} | ssm={ssm_name} | decoder={decoder_name}")

    if ssm_l in ['none', 'identity', 'no', 'false']:
        # 현재 코드베이스에서 SSM 없는 baseline은 Swin/Vit 경로만 제공됨
        if enc_l not in ['swin', 'vision', 'vit', 'swin-t', 'swin_t']:
            raise ValueError(f"SSM 없이 encoder='{encoder_name}'는 아직 지원하지 않습니다. (현재는 encoder=swin/vit만 가능)")
        model_u = teeth_arangement_model()
        model_l = teeth_arangement_model()

    else:
        # Kalman/S4 계열은 teeth_arangement_kalman_ssm_model을 사용
        if ssm_l not in ['kalman', 'kalman_smoothing', 's4']:
            raise ValueError(f"Unknown ssm: {ssm_name}")

        if enc_l in ['pointnetpp', 'pointnet++', 'pn++', 'pointnet2', 'pn2']:
            encoder_type = 'pointnet++'
        elif enc_l in ['pointnext', 'pnx', 'pointnextencoder']:
            encoder_type = 'pointnext'
        else:
            encoder_type = 'vit'

        model_u = teeth_arangement_kalman_ssm_model(
            tooth_id_offset=0,
            encoder_type=encoder_type,
            encoder_out_dim=args.emb_dims,
            pointnet_use_center=getattr(args, 'pointnet_use_center', True),
            ssm_cfg=args.ssm_cfg,
            pointnext_k=getattr(args, 'pointnext_k', 16),
            pointnext_stage_points=getattr(args, 'pointnext_stage_points', (128,32,None)),
            pointnext_stage_dims=getattr(args, 'pointnext_stage_dims', (64,128,256)),
            pointnext_expansion=getattr(args, 'pointnext_expansion', 4),
        )

        model_l = teeth_arangement_kalman_ssm_model(
            tooth_id_offset=16,
            encoder_type=encoder_type,
            encoder_out_dim=args.emb_dims,
            pointnet_use_center=getattr(args, 'pointnet_use_center', True),
            ssm_cfg=args.ssm_cfg,
            pointnext_k=getattr(args, 'pointnext_k', 16),
            pointnext_stage_points=getattr(args, 'pointnext_stage_points', (128,32,None)),
            pointnext_stage_dims=getattr(args, 'pointnext_stage_dims', (64,128,256)),
            pointnext_expansion=getattr(args, 'pointnext_expansion', 4),
        )

    tooth_assembler_u = Tooth_Assembler()
    tooth_assembler_l = Tooth_Assembler()

    # loss composer (config-driven)
    loss_composer = LossComposer.from_cfg(cfg_dict).to(device)

    _terms = (cfg_dict or {}).get('loss', {}).get('terms', []) or []
    need_intermediates = any(
        isinstance(tc, dict)
        and bool(tc.get('enabled', True))
        and float(tc.get('weight', 1.0)) != 0.0
        and str(tc.get('name', '')).lower() in ['geodesic_step', 'movement_balance']
        for tc in _terms
    )
 
    # -------------------------
    # Stage augmentation (B: micro-batch pseudo-starts)
    # -------------------------
    tr_cfg = (cfg_dict or {}).get('training', {}) or {}
    sa_micro = (tr_cfg.get('stage_aug_micro', None) or {})  # alias
    micro_cfg = StageAugMicroConfig(
        enabled=bool(sa_micro.get('enabled', False)),
        start_epoch=int(sa_micro.get('start_epoch', 30)),
        prob=float(sa_micro.get('prob', 1.0)),
        steps=list(sa_micro.get('steps', [])) if sa_micro.get('steps', None) is not None else None,
        max_pseudos=int(sa_micro.get('max_pseudos', 0)),
        alpha=float(sa_micro.get('alpha', 0.5)),
        cache_dtype=str(sa_micro.get('cache_dtype', 'float16')),
        teacher=str(sa_micro.get('teacher', 'prev_epoch')),
    )
    micro_steps = micro_cfg.normalized_steps(K_steps)
    if len(micro_steps) == 0:
        micro_cfg.enabled = False

    # SSM이 없으면 stage augmentation 불가
    if ssm_l in ['none', 'identity', 'no', 'false']:
        micro_cfg.enabled = False

    # stage augmentation을 쓰면 intermediates(T_steps)가 필요
    need_intermediates = need_intermediates or micro_cfg.enabled

    _cache_dtype = torch.float16 if micro_cfg.cache_dtype.lower() in ['float16', 'fp16'] else torch.float32
    cache_prev = StageTransformCacheK(dtype=_cache_dtype)
    cache_cur  = StageTransformCacheK(dtype=_cache_dtype)
   
    if args.use_sgd:
        print("Use SGD")
        opt_u = optim.SGD(model_u.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        opt_l = optim.SGD(model_l.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    else:
        print("Use Adam")
        opt_u = optim.Adam(model_u.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt_l = optim.Adam(model_l.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'cos':
        scheduler_u = CosineAnnealingLR(opt_u, args.epochs, eta_min=1e-6, last_epoch = -1)
        scheduler_l = CosineAnnealingLR(opt_l, args.epochs, eta_min=1e-6, last_epoch = -1)
    elif args.scheduler == 'step':
        scheduler_u = StepLR(opt_u, step_size=20, gamma=0.7)
        scheduler_l = StepLR(opt_l, step_size=20, gamma=0.7)

    model_u = model_u.to(device)
    model_l = model_l.to(device)
    model_u.train()
    model_l.train()
    scaler_u = GradScaler()
    scaler_l = GradScaler()
    best_test_acc = 0
    inter_nums = len(train_loader)
    prev_top_terms = None

    for epoch in range(args.epochs):
        if micro_cfg.enabled:
            cache_cur.clear()
        ####################
        # Train
        ####################

        if epoch > 0:
            if args.scheduler == 'cos':
                scheduler_u.step()
                scheduler_l.step()
            elif args.scheduler == 'step':
                if opt_u.param_groups[0]['lr'] > 1e-5:
                    scheduler_u.step()
                if opt_l.param_groups[0]['lr'] > 1e-5:
                    scheduler_l.step()
                if opt_u.param_groups[0]['lr'] < 1e-5:
                    for param_group in opt_u.param_groups:
                        param_group['lr'] = 1e-5
                if opt_l.param_groups[0]['lr'] < 1e-5:
                    for param_group in opt_l.param_groups:
                        param_group['lr'] = 1e-5

        # epoch-wise aggregators
        loss_avg = MetricAverager()
        nums = 0
        tic = time.time()
        train_data_u, train_label_u, teeth_center_u, dof_u = [], [], [], []
        train_data_l, train_label_l, teeth_center_l, dof_l = [], [], [], []
        nnums = 0

        last_val_tre = float("nan")
        last_val_mmrot = float("nan")
        last_val_mmtrans = float("nan")

        for idx, (pack_u, pack_l, down_paths) in enumerate(train_loader):
            case_ids = [case_id_from_path(p) for p in down_paths]
            (train_data_u, train_label_u, teeth_center_u, gdofs_u, gtrans_u, tweights_u, rweights_u, mask_index_u) = pack_u
            (train_data_l, train_label_l, teeth_center_l, gdofs_l, gtrans_l, tweights_l, rweights_l, mask_index_l) = pack_l

            train_data_u = train_data_u.to(device, non_blocking=True).float()
            train_data_l = train_data_l.to(device, non_blocking=True).float()
            train_label_u = train_label_u.to(device, non_blocking=True).float()
            train_label_l = train_label_l.to(device, non_blocking=True).float()
            teeth_center_u = teeth_center_u.to(device, non_blocking=True).float()
            teeth_center_l = teeth_center_l.to(device, non_blocking=True).float()
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

            weights_u = rweights_u - 1 + tweights_u
            weights_l = rweights_l - 1 + tweights_l

            gdofs_u = gdofs_u
            gdofs_l = gdofs_l

            nums = nums + 1
            batch_size = train_data_u.size()[0]
            opt_u.zero_grad()
            opt_l.zero_grad()

            aux_u = None
            aux_l = None

            # -------------------------
            # 1) Original batch (always)
            # -------------------------
            with autocast():
                if ssm_l in ['none', 'identity', 'no', 'false']:
                    pdofs_u, ptrans_u = model_u(train_data_u, teeth_center_u)
                    pdofs_l, ptrans_l = model_l(train_data_l, teeth_center_l)
                else:
                    # need_intermediates is forced True when stage_aug_micro is enabled
                    pdofs_u, ptrans_u, aux_u = model_u(train_data_u, teeth_center_u, return_intermediates=need_intermediates)
                    pdofs_l, ptrans_l, aux_l = model_l(train_data_l, teeth_center_l, return_intermediates=need_intermediates)

                assembled_u = tooth_assembler_u(train_data_u, teeth_center_u, pdofs_u, ptrans_u, device)
                assembled_l = tooth_assembler_l(train_data_l, teeth_center_l, pdofs_l, ptrans_l, device)

                pred_u = {'quat': pdofs_u, 'trans': ptrans_u, 'assembled': assembled_u}
                pred_l = {'quat': pdofs_l, 'trans': ptrans_l, 'assembled': assembled_l}
                if aux_u is not None: pred_u['aux'] = aux_u
                if aux_l is not None: pred_l['aux'] = aux_l

                batch_u = {
                    'label': train_label_u,
                    'gdofs': gdofs_u,
                    'gtrans': gtrans_u,
                    'tweights': tweights_u,
                    'rweights': rweights_u,
                    'mask_index': mask_index_u,
                    'weights': weights_u,
                    'data': train_data_u,
                    'center': teeth_center_u,
                }
                batch_l = {
                    'label': train_label_l,
                    'gdofs': gdofs_l,
                    'gtrans': gtrans_l,
                    'tweights': tweights_l,
                    'rweights': rweights_l,
                    'mask_index': mask_index_l,
                    'weights': weights_l,
                    'data': train_data_l,
                    'center': teeth_center_l,
                }

                loss_u, loss_l, loss_logs = loss_composer(
                    pred_u, batch_u, pred_l, batch_l,
                    ctx={'device': device}
                )

            # backward (orig)
            scaler_u.scale(loss_u).backward()
            scaler_l.scale(loss_l).backward()

            # cache current epoch teacher candidates (K-step cumulative transforms) for next epoch
            if micro_cfg.enabled and (aux_u is not None) and (aux_l is not None) and ('T_steps' in aux_u) and ('T_steps' in aux_l):
                cache_cur.set_all('up', case_ids, aux_u['T_steps'])
                cache_cur.set_all('down', case_ids, aux_l['T_steps'])

            # -------------------------
            # 2) Micro-batch pseudo-starts (B 방식)
            #    original batch는 유지 + steps 리스트만큼 pseudo 추가 학습
            # -------------------------
            pseudo_u_sum = 0.0
            pseudo_l_sum = 0.0
            pseudo_used = 0

            do_micro = micro_cfg.enabled and (epoch >= micro_cfg.start_epoch) and (len(micro_steps) > 0)
            if do_micro:
                # choose steps for this batch
                used_steps = list(micro_steps)
                if micro_cfg.max_pseudos and micro_cfg.max_pseudos > 0 and len(used_steps) > micro_cfg.max_pseudos:
                    used_steps = list(np.random.choice(used_steps, size=micro_cfg.max_pseudos, replace=False))
                    used_steps = [int(x) for x in used_steps]

                # sample which cases in this batch to augment (original always included)
                B0 = train_data_u.shape[0]
                aug_mask = (torch.rand(B0, device=device) < float(micro_cfg.prob))
                if micro_cfg.teacher.lower() in ['prev', 'prev_epoch', 'cache']:
                    avail = torch.tensor(cache_prev.has_both(case_ids), device=device, dtype=torch.bool)
                    aug_mask = aug_mask & avail
                if aug_mask.any():
                    sel_idx = torch.nonzero(aug_mask, as_tuple=False).squeeze(1)
                    sel_case_ids = [case_ids[i] for i in sel_idx.detach().cpu().tolist()]
                    w_pseudo = float(micro_cfg.alpha) / max(len(used_steps), 1)

                    # teacher source
                    #  - prev_epoch: cache_prev.get_step(...)
                    #  - current   : use aux_*['T_steps'] from the original forward (detached)
                    for k in used_steps:
                        k = int(k)
                        if micro_cfg.teacher.lower() in ['prev', 'prev_epoch', 'cache']:
                            T0k_u = cache_prev.get_step('up', sel_case_ids, k, device=device, out_dtype=train_data_u.dtype)
                            T0k_l = cache_prev.get_step('down', sel_case_ids, k, device=device, out_dtype=train_data_l.dtype)
                            if (T0k_u is None) or (T0k_l is None):
                                continue
                        else:
                            if aux_u is None or aux_l is None:
                                continue
                            T0k_u = aux_u['T_steps'][k-1, sel_idx].detach()
                            T0k_l = aux_l['T_steps'][k-1, sel_idx].detach()

                        # build pseudo-start batches (subset only)
                        x0_u = train_data_u[sel_idx]
                        x0_l = train_data_l[sel_idx]
                        y_u  = train_label_u[sel_idx]
                        y_l  = train_label_l[sel_idx]

                        pseudo_u = apply_T_points(T0k_u, x0_u)
                        pseudo_l = apply_T_points(T0k_l, x0_l)
                        center_u_p = pseudo_u.mean(dim=2, keepdim=True)
                        center_l_p = pseudo_l.mean(dim=2, keepdim=True)

                        gdofs_u_p, gtrans_u_p = compose_gt_from_teacher(T0k_u.detach(), gdofs_u[sel_idx], gtrans_u[sel_idx])
                        gdofs_l_p, gtrans_l_p = compose_gt_from_teacher(T0k_l.detach(), gdofs_l[sel_idx], gtrans_l[sel_idx])

                        # forward on pseudo-starts
                        with autocast():
                            pd_u, pt_u, a_u = model_u(pseudo_u, center_u_p, return_intermediates=need_intermediates)
                            pd_l, pt_l, a_l = model_l(pseudo_l, center_l_p, return_intermediates=need_intermediates)

                            as_u = tooth_assembler_u(pseudo_u, center_u_p, pd_u, pt_u, device)
                            as_l = tooth_assembler_l(pseudo_l, center_l_p, pd_l, pt_l, device)

                            pr_u = {'quat': pd_u, 'trans': pt_u, 'assembled': as_u}
                            pr_l = {'quat': pd_l, 'trans': pt_l, 'assembled': as_l}
                            if a_u is not None: pr_u['aux'] = a_u
                            if a_l is not None: pr_l['aux'] = a_l

                            bt_u = {
                                'label': y_u,
                                'gdofs': gdofs_u_p,
                                'gtrans': gtrans_u_p,
                                'tweights': tweights_u[sel_idx],
                                'rweights': rweights_u[sel_idx],
                                'mask_index': mask_index_u[sel_idx],
                                'weights': weights_u[sel_idx],
                                'data': pseudo_u,
                                'center': center_u_p,
                            }
                            bt_l = {
                                'label': y_l,
                                'gdofs': gdofs_l_p,
                                'gtrans': gtrans_l_p,
                                'tweights': tweights_l[sel_idx],
                                'rweights': rweights_l[sel_idx],
                                'mask_index': mask_index_l[sel_idx],
                                'weights': weights_l[sel_idx],
                                'data': pseudo_l,
                                'center': center_l_p,
                            }

                            lu_p, ll_p, _ = loss_composer(pr_u, bt_u, pr_l, bt_l, ctx={'device': device})

                        # scale & backward (pseudo)
                        scaler_u.scale(lu_p * w_pseudo).backward()
                        scaler_l.scale(ll_p * w_pseudo).backward()
                        pseudo_u_sum += float((lu_p.detach() * w_pseudo).item())
                        pseudo_l_sum += float((ll_p.detach() * w_pseudo).item())
                        pseudo_used += 1

            # optimizer step (single step for orig + all pseudos)
            scaler_u.step(opt_u)
            scaler_l.step(opt_l)
            scaler_u.update()
            scaler_l.update()

            # aggregate loss logs (dynamic)
            loss_logs = dict(loss_logs)
            loss_logs['loss/total_orig/u'] = float(loss_logs.get('loss/total/u', 0.0))
            loss_logs['loss/total_orig/l'] = float(loss_logs.get('loss/total/l', 0.0))
            loss_logs['loss/total_orig']   = float(loss_logs.get('loss/total', 0.0))
            loss_logs['loss/total_pseudo/u'] = float(pseudo_u_sum)
            loss_logs['loss/total_pseudo/l'] = float(pseudo_l_sum)
            loss_logs['loss/total_pseudo']   = 0.5 * (loss_logs['loss/total_pseudo/u'] + loss_logs['loss/total_pseudo/l'])
            loss_logs['loss/total/u'] = loss_logs['loss/total_orig/u'] + loss_logs['loss/total_pseudo/u']
            loss_logs['loss/total/l'] = loss_logs['loss/total_orig/l'] + loss_logs['loss/total_pseudo/l']
            loss_logs['loss/total']   = loss_logs['loss/total_orig']   + loss_logs['loss/total_pseudo']

            loss_avg.update(loss_logs, n=batch_size)
        if micro_cfg.enabled:
            cache_prev.swap_from(cache_cur)

        # (step-wise logging removed; epoch-wise logging is used)

        # -------------------------
        # Epoch-wise validation + logging
        # -------------------------
        train_logs_epoch = loss_avg.as_dict()
        avg_train_loss = float(train_logs_epoch.get('loss/total', float('nan')))

        terms_block = ""
        if getattr(args, "print_loss_terms", False):
            terms_line, cur_top_terms = format_loss_terms_inline(
                train_logs_epoch,
                prev_top=prev_top_terms,
                precision=getattr(args, "loss_terms_precision", 4),
                max_terms=getattr(args, "loss_terms_max_terms", 0),
                with_delta=getattr(args, "loss_terms_delta", True),
            )
            if terms_line:
                terms_block = terms_line + " | "
            prev_top_terms = cur_top_terms

        do_eval = (val_loader is not None) and (((epoch + 1) % getattr(args, 'eval_every', 1) == 0) or (epoch == args.epochs - 1))
        val_metrics = {}
        if do_eval:
            def _val_extra_fn(pack_u, pack_l, pdofs_u, ptrans_u, pdofs_l, ptrans_l, assembled_u, assembled_l, device):
                (data_u, label_u, center_u, gdofs_u, gtrans_u, tweights_u, rweights_u, mask_index_u) = pack_u
                (data_l, label_l, center_l, gdofs_l, gtrans_l, tweights_l, rweights_l, mask_index_l) = pack_l

                label_u = label_u.to(device, non_blocking=True).float()
                label_l = label_l.to(device, non_blocking=True).float()
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

                weights_u = rweights_u - 1 + tweights_u
                weights_l = rweights_l - 1 + tweights_l

                pred_u = {'quat': pdofs_u, 'trans': ptrans_u, 'assembled': assembled_u}
                pred_l = {'quat': pdofs_l, 'trans': ptrans_l, 'assembled': assembled_l}
                batch_u = {
                    'label': label_u,
                    'gdofs': gdofs_u,
                    'gtrans': gtrans_u,
                    'tweights': tweights_u,
                    'rweights': rweights_u,
                    'mask_index': mask_index_u,
                    'weights': weights_u,
                }
                batch_l = {
                    'label': label_l,
                    'gdofs': gdofs_l,
                    'gtrans': gtrans_l,
                    'tweights': tweights_l,
                    'rweights': rweights_l,
                    'mask_index': mask_index_l,
                    'weights': weights_l,
                }

                lu, ll, _ = loss_composer(pred_u, batch_u, pred_l, batch_l, ctx={'device': device})
                loss = 0.5 * (lu + ll)
                return {'loss': float(loss.detach().item())}

            val_metrics = evaluate(
                val_loader,
                model_u, model_l,
                tooth_assembler_u, tooth_assembler_l,
                device,
                extra_fn=_val_extra_fn,
                use_amp=True,
            )

        val_loss = float(val_metrics.get('loss', float('nan')))
        metric_line = format_metrics_line(
            val_metrics,
            exclude={'loss'},
            rename={
                'tre_mm': 'TRE(mm)',
                'mm_rot_deg': 'MMrot(deg)',
                'mm_trans_mm': 'MMtrans(mm)',
            },
            precision=4
        )

        lr_now = float(opt_u.param_groups[0]['lr'])
        epoch_time = time.time() - tic
        io.cprint(
            f"[Epoch {epoch+1:03d}/{args.epochs}] "
            f"train_loss={avg_train_loss:.4f} | "
            f"{terms_block}"
            f"val_loss={val_loss:.4f} "
            f"{metric_line} "
            f"| lr={lr_now:.2e}  {epoch_time:.1f}s"
        )

        save_model_path = os.path.join(args.output_dir, "save_model")
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        do_save = (((epoch + 1) % getattr(args, 'save_every', 1) == 0) or (epoch == args.epochs - 1))
        if do_save:
            torch.save({'model': model_u.state_dict(), 'epoch': epoch}, os.path.join(save_model_path, 'orth_model_u_') + str(epoch) + '.pth')
            torch.save({'model': model_l.state_dict(), 'epoch': epoch}, os.path.join(save_model_path, 'orth_model_l_') + str(epoch) + '.pth')

# =========================
# Config-driven entrypoint
# =========================

def _read_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _resolve_rel(base_dir: str, p: str) -> str:
    if not p:
        return ''
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def _load_json(path: str) -> dict:
    if not path:
        print("[NO JSON PATH]")
        return {}
    if not os.path.exists(path):
        print("[NO JSON FILE]")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        print(f"[{path} JSON LOADED]")
        return json.load(f)


def build_args_from_config(cfg_dict: dict, yaml_path: str):
    """Build an args-like object (SimpleNamespace) from YAML + referenced JSON."""
    from types import SimpleNamespace

    base_dir = os.path.dirname(os.path.abspath(yaml_path))

    exp = cfg_dict.get('experiment', {}) or {}
    data = cfg_dict.get('data', {}) or {}
    tr = cfg_dict.get('training', {}) or {}
    opt = cfg_dict.get('optimizer', {}) or {}
    sch = cfg_dict.get('scheduler', {}) or {}
    model = cfg_dict.get('model', {}) or {}
    runtime = cfg_dict.get('runtime', {}) or {}
    log = cfg_dict.get('logging', {}) or {}
    
    enc = (model.get('encoder', {}) or {})
    ssm = (model.get('ssm', {}) or {})
    dec = (model.get('decoder', {}) or {})

    enc_json = _load_json(_resolve_rel(base_dir, enc.get('config', '')))
    ssm_json = _load_json(_resolve_rel(base_dir, ssm.get('config', '')))
    dec_json = _load_json(_resolve_rel(base_dir, dec.get('config', '')))

    args = SimpleNamespace()

    # experiment
    args.exp_name = exp.get('name', 'exp_1')
    args.seed = int(exp.get('seed', 1))
    args.device = exp.get('device', 'cuda:0')
    args.output_dir = exp.get('output_dir', os.path.join('outputs', args.exp_name))
    args.num_workers = int(exp.get('num_workers', 8))
    args.save_every = int(exp.get('save_every', 1))
    args.eval_every = int(exp.get('eval_every', 1))

    # data
    args.splits_json = data.get('splits_json', '')
    args.data_root = data.get('data_root', '')
    args.num_points = int(data.get('num_points', 2048))
    args.raw_otf = bool(data.get('raw_otf', False))

    # training
    args.epochs = int(tr.get('epochs', 200))
    args.batch_size = int(tr.get('batch_size', 8))
    args.test_batch_size = int(tr.get('test_batch_size', 1))
    args.lr = float(tr.get('lr', 1e-4))
    args.weight_decay = float(tr.get('weight_decay', 1e-4))

    # optimizer
    opt_name = str(opt.get('name', 'adamw')).lower()
    args.use_sgd = (opt_name == 'sgd')
    args.momentum = float(opt.get('momentum', 0.9))

    # scheduler
    args.scheduler = str(sch.get('name', 'cos')).lower()

    # model parts
    args.encoder = enc.get('name', enc_json.get('type', 'swin'))
    args.pointnext_k = int(enc_json.get('k', 16))
    args.pointnext_stage_points = enc_json.get('stage_points', [128, 32, None])
    args.pointnext_stage_dims = enc_json.get('stage_dims', [64, 128, 256])
    args.pointnext_expansion = int(enc_json.get('expansion', 4))

    # Full SSM config dict (forwarded to the model as-is)
    args.ssm = ssm.get('name', ssm_json.get('type', 'kalman'))
    args.ssm_cfg = dict(ssm_json)
    args.ssm_cfg.setdefault("name", args.ssm)   # yaml의 name이 우선
    args.ssm_cfg.setdefault("type", args.ssm)   # 호환용
    args.decoder = dec.get('name', dec_json.get('type', 'baseline'))

    # dims / steps
    # K is defined ONLY in SSM config (config/model/ssm/*.json or YAML model.ssm.K fallback)
    args.ssm_cfg['K'] = int(ssm_json.get('K', ssm.get('K', 8)))
    # embedding dimension always follows encoder output dimension (SSM does not own hidden_dim)
    args.emb_dims = int(enc_json.get('out_dim', 256))
    # remove any legacy hidden-dim fields from SSM config to avoid confusion
    args.ssm_cfg.pop('hidden_dim', None)
    args.ssm_cfg.pop('d_model', None)
    # pointnet flag
    args.pointnet_use_center = bool(enc_json.get('use_center', True))

    # runtime
    args.model_path = runtime.get('resume_ckpt', '') or ''
    args.eval = bool(runtime.get('eval', False))

    # logging
    args.print_loss_terms = bool(log.get('print_loss_terms', False))
    args.loss_terms_precision = int(log.get('loss_terms_precision', 4))
    args.loss_terms_max_terms = int(log.get('loss_terms_max_terms', 0))
    args.loss_terms_delta = bool(log.get('loss_terms_delta', True))
    
    # keep legacy flags
    args.no_cuda = False
    args.cuda = True

    return args


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False

    parser = argparse.ArgumentParser(description='Orth-Tooth (config-driven)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--print_config', action='store_true')
    cli = parser.parse_args()

    cfg_dict = _read_yaml(cli.config)
    if cli.print_config:
        print(json.dumps(cfg_dict, indent=2, ensure_ascii=False))
        raise SystemExit(0)

    args = build_args_from_config(cfg_dict, cli.config)
    _init_(args)

    # save YAML snapshot into outputs for reproducibility
    save_config_snapshot(cli.config, args.output_dir, cfg_dict)

    runlog_path = os.path.join(args.output_dir, 'run_.log')
    io = IOStream(runlog_path)
    io.cprint(f"[CONFIG] {cli.config}")
    io.cprint(f"[MODEL] encoder={args.encoder} ssm={args.ssm} decoder={args.decoder}")

    # dump args once at the start (both to console/log and to file)
    try:
        args_dict = {k: getattr(args, k) for k in sorted(vars(args).keys())}
        io.cprint("[ARGS]\n" + json.dumps(args_dict, indent=2, ensure_ascii=False, default=str))
        with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass


    # seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.eval:
        train(args, io, cfg_dict)
