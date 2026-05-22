"""
Data utilities: path discovery, mesh I/O, normalization, and train/val/test splitting.

Usage (via main.py):
    python main.py --mode split
"""
import os
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def list_cases(root_dir: str) -> List[str]:
    """List case directories (data_001, data_002, ...) under root_dir."""
    cases = []
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and name.startswith("data_"):
            cases.append(p)
    return cases


def parse_target_matrix_txt(path: str) -> Dict[int, np.ndarray]:
    """Parse target_matrix.txt → {tooth_id: (4,4) float32}."""
    out: Dict[int, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.replace(",", " ").split()
            try:
                tid = int(float(toks[0]))
            except Exception:
                continue
            vals = []
            for t in toks[1:]:
                try:
                    vals.append(float(t))
                except Exception:
                    pass
            if len(vals) != 16:
                raise ValueError(f"Line for tooth {tid} does not have 16 floats. got={len(vals)}")
            M = np.array(vals, dtype=np.float32).reshape(4, 4)
            out[tid] = M
    return out


def read_mesh(stl_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(stl_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load STL: {stl_path}")
    return mesh


def _farthest_point_sample(pts: np.ndarray, n: int) -> np.ndarray:
    """Farthest-point sampling (greedy, O(N*n))."""
    N = pts.shape[0]
    if N <= n:
        return pts[:n].copy() if N >= n else np.pad(pts, ((0, n - N), (0, 0)), mode='edge')
    selected = [0]
    dists = np.full(N, np.inf, dtype=np.float32)
    for _ in range(n - 1):
        last = pts[selected[-1]]
        d = np.linalg.norm(pts - last, axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return pts[selected].astype(np.float32)


def sample_points(mesh: trimesh.Trimesh, n_points: int, rng: np.random.Generator,
                  include_normals: bool = False,
                  method: str = "random") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if method == "fps":
        oversample = max(n_points * 4, 8192)
        raw_pts, face_idx = trimesh.sample.sample_surface(mesh, oversample)
        pts = _farthest_point_sample(raw_pts.astype(np.float32), n_points)
        normals = None
        if include_normals:
            from scipy.spatial import cKDTree
            tree = cKDTree(raw_pts)
            _, idx = tree.query(pts)
            normals = mesh.face_normals[face_idx[idx]].astype(np.float32)
        return pts, normals
    else:
        pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
        pts = pts.astype(np.float32)
        normals = None
        if include_normals:
            normals = mesh.face_normals[face_idx].astype(np.float32)
        return pts, normals


def apply_T(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply row-major SE(3) transform: p' = [p, 1] @ T."""
    P = points.shape[0]
    homo = np.concatenate([points, np.ones((P, 1), dtype=np.float32)], axis=1)
    out = (homo @ T)[:, :3]
    return out.astype(np.float32)


def normalize_case(points_list: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray, float]:
    all_pts = np.concatenate(points_list, axis=0)
    center = all_pts.mean(axis=0).astype(np.float32)
    centered = all_pts - center
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-8:
        scale = 1.0
    normalized = [((pts - center) / scale).astype(np.float32) for pts in points_list]
    return normalized, center, scale


def collect_stls(case_dir: str, source_subdir: str) -> List[Tuple[int, str, str]]:
    """Return [(tooth_id, jaw, stl_path), ...] for a case directory."""
    items: List[Tuple[int, str, str]] = []
    base = os.path.join(case_dir, source_subdir)

    for jaw in ["up", "down"]:
        jaw_dir = os.path.join(base, jaw)
        if not os.path.isdir(jaw_dir):
            continue
        for fn in sorted(os.listdir(jaw_dir)):
            if not fn.lower().endswith(".stl"):
                continue
            name = os.path.splitext(fn)[0]
            try:
                tid = int(name)
            except Exception:
                continue
            items.append((tid, jaw, os.path.join(jaw_dir, fn)))

    items.sort(key=lambda x: (x[0], x[1]))
    return items


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def run_split(cfg):
    """Generate train/val/test splits from raw data directory.

    Uses cfg.data.raw_dir, cfg.data.split_json.
    """
    raw_dir = cfg.data.raw_dir
    # Find valid cases (those with target_matrix.txt)
    cases = []
    for name in sorted(os.listdir(raw_dir)):
        case_dir = os.path.join(raw_dir, name)
        if os.path.isdir(case_dir) and os.path.isfile(os.path.join(case_dir, "target_matrix.txt")):
            cases.append(name)

    if not cases:
        raise RuntimeError(f"No cases found under: {raw_dir}")

    train_r, val_r, test_r = 0.8, 0.1, 0.1

    random.shuffle(cases)

    n = len(cases)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    train_cases = cases[:n_train]
    val_cases = cases[n_train:n_train + n_val]
    test_cases = cases[n_train + n_val:]

    out_path = cfg.data.split_json
    out_abs = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_abs)

    raw_data_root_rel = os.path.relpath(os.path.abspath(raw_dir), out_dir)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_data_root": raw_data_root_rel.replace("\\", "/"),
        "ratios": {"train": train_r, "val": val_r, "test": test_r},
        "counts": {"total": n, "train": len(train_cases), "val": len(val_cases), "test": len(test_cases)},
        "train": train_cases,
        "val": val_cases,
        "test": test_cases,
    }
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Also write per-split .txt files (absolute paths, one per line)
    # compatible with STTAlign_SSM/test.py --input_txt
    raw_abs = os.path.abspath(raw_dir)
    for split_name, split_cases in [("train", train_cases), ("val", val_cases), ("test", test_cases)]:
        txt_path = os.path.join(out_dir, f"{split_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for case_name in split_cases:
                f.write(os.path.join(raw_abs, case_name) + "\n")

    print(f"Saved splits to: {out_path}")
    print(f"Saved .txt files to: {out_dir}/{{train,val,test}}.txt")
    print(payload["counts"])
