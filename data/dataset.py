import os, json, struct, warnings
import numpy as np
import torch
from torch.utils.data import Dataset
import threading

# Maximum face count we consider plausible for a single-tooth STL
_MAX_STL_FACES = 5_000_000


def _validate_stl_header(stl_path):
    """Check STL file for corrupt binary headers that would cause OOM.

    Returns True if the file looks safe to load, False only for corrupt binary
    STLs whose bad face count would trigger trimesh's ASCII fallback OOM.
    ASCII STL files (starting with 'solid') are always accepted.
    """
    try:
        file_size = os.path.getsize(stl_path)
        if file_size < 84:
            return True  # too small for binary STL; let trimesh handle
        with open(stl_path, "rb") as f:
            header = f.read(84)
        # ASCII STL starts with "solid" — always safe
        if header[:5].lower() == b"solid":
            return True
        # Binary STL: validate face count vs file size
        n_faces = struct.unpack("<I", header[80:84])[0]
        if n_faces > _MAX_STL_FACES:
            return False
        expected_size = 84 + n_faces * 50
        if expected_size > file_size + 100:
            return False
        return True
    except OSError:
        return False


def augment_points(points_in, T_gt, jitter=0.005, dropout=0.05,
                    aug_rotation=0.0, aug_scale=0.0, aug_translation=0.0):
    """Online point cloud augmentation.

    Args:
        points_in: (N, P, 3) per-tooth point clouds
        T_gt: (N, 4, 4) ground truth transforms
        jitter: std of Gaussian noise added to points (normalized units)
        dropout: fraction of points to zero out per tooth
        aug_rotation: max rotation angle in radians for tooth-relative rotation
        aug_scale: max scale perturbation (e.g. 0.03 for ±3%)
        aug_translation: max per-tooth translation perturbation per axis (normalized units)

    Returns:
        points_in: augmented (N, P, 3)
        T_gt: corrected (N, 4, 4)
    """
    N, P, _ = points_in.shape

    # Per-tooth translation perturbation
    if aug_translation > 0:
        t_aug = (torch.rand(N, 1, 3, device=points_in.device, dtype=points_in.dtype) * 2 - 1) * aug_translation
        points_in = points_in + t_aug
        T_gt = T_gt.clone()
        T_gt[..., 3, :3] = T_gt[..., 3, :3] - t_aug.squeeze(1)

    # Random scaling (per-case)
    if aug_scale > 0:
        s = 1.0 + (torch.rand(1).item() * 2 - 1) * aug_scale
        points_in = points_in * s
        T_gt = T_gt.clone()
        T_gt[..., 3, :3] = T_gt[..., 3, :3] * s

    # Tooth-relative rotation
    if aug_rotation > 0:
        from utils.geometry import exp_so3
        w = torch.randn(N, 3, device=points_in.device, dtype=points_in.dtype) * aug_rotation
        R_aug = exp_so3(w)
        centroids = points_in.mean(dim=1, keepdim=True)
        points_in = centroids + (points_in - centroids) @ R_aug
        T_gt = T_gt.clone()
        R_gt = T_gt[..., :3, :3]
        t_gt = T_gt[..., 3, :3]
        R_aug_T = R_aug.transpose(-1, -2)
        I3 = torch.eye(3, device=points_in.device, dtype=points_in.dtype)
        c = centroids.squeeze(1)
        T_gt[..., :3, :3] = R_aug_T @ R_gt
        T_gt[..., 3, :3] = t_gt - (c.unsqueeze(1) @ (I3 - R_aug) @ R_aug_T @ R_gt).squeeze(1)

    # Jitter
    if jitter > 0:
        noise = torch.randn_like(points_in) * jitter
        points_in = points_in + noise

    # Point dropout
    if dropout > 0:
        keep_mask = torch.rand(N, P) > dropout
        points_in = points_in * keep_mask.unsqueeze(-1).float()

    return points_in, T_gt


def collate_pad(batch, max_teeth=32):
    """
    Pads per-case teeth axis to max_teeth.
    points_in: (B, max_teeth, P, 3)
    T_gt:      (B, max_teeth, 4, 4)
    tooth_id:  (B, max_teeth)
    mask:      (B, max_teeth)
    scale:     (B,)
    center:    (B, 3)
    """
    B = len(batch)
    P = batch[0]["points_in"].shape[1]

    points_in = torch.zeros((B, max_teeth, P, 3), dtype=torch.float32)
    T_gt = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(B, max_teeth, 1, 1)
    mask = torch.zeros((B, max_teeth), dtype=torch.bool)
    tooth_id = torch.full((B, max_teeth), -1, dtype=torch.int64)
    scale = torch.zeros((B, ), dtype=torch.float32)
    center = torch.zeros((B, 3), dtype=torch.float32)

    has_points_gt = "points_gt" in batch[0]
    points_gt = torch.zeros((B, max_teeth, P, 3), dtype=torch.float32) if has_points_gt else None

    case_names = []
    for b, item in enumerate(batch):
        case_names.append(item.get("case_name", f"case_{b}"))
        n = min(item["points_in"].shape[0], max_teeth)

        points_in[b, :n] = item["points_in"][:n]
        T_gt[b, :n] = item["T_gt"][:n]
        tooth_id[b, :n] = item["tooth_id"][:n]
        scale[b] = item["scale"]
        center[b] = item["center"]
        mask[b, :n] = True

        if has_points_gt:
            points_gt[b, :n] = item["points_gt"][:n]

    out = {"case_name": case_names, "points_in": points_in, "T_gt": T_gt,
           "mask": mask, "tooth_id": tooth_id, "scale": scale, "center": center}
    if has_points_gt:
        out["points_gt"] = points_gt
    return out


class OnTheFlyDataset(Dataset):
    """Dataset that samples points from raw STL meshes on-the-fly.

    Each __getitem__ call produces fresh random surface samples, providing
    data augmentation through sampling diversity across epochs.

    Args:
        raw_dir: path to raw data directory (e.g., data/raw)
        case_list: list of case names to include
        n_points: number of points to sample per tooth
        cache_meshes: if True, cache trimesh objects in memory
        sampling_method: "random" or "fps" — farthest-point sampling
        augment: enable online augmentation
        aug_cfg: dict with augmentation parameters
        exclude_missing_teeth: filter out cases with missing intermediate teeth
    """

    def __init__(self, raw_dir, case_list, n_points=2048,
                 cache_meshes=True, sampling_method="random",
                 augment=False, aug_cfg=None,
                 exclude_missing_teeth=False):
        from data.prepare import collect_stls, parse_target_matrix_txt

        self.raw_dir = raw_dir
        self.n_points = n_points
        self.cache_meshes = cache_meshes
        self.sampling_method = sampling_method
        self.augment = augment
        self.aug_cfg = aug_cfg or {}

        # Build case metadata
        self.cases = []
        for case_name in case_list:
            case_dir = os.path.join(raw_dir, case_name)
            matrix_path = os.path.join(case_dir, "target_matrix.txt")
            if not os.path.isfile(matrix_path):
                continue
            try:
                T_dict = parse_target_matrix_txt(matrix_path)
                stl_items = collect_stls(case_dir, "crown")
            except Exception:
                continue
            # Filter to teeth with both STL and matrix, validate STL header
            valid_items = []
            for tid, jaw, path in stl_items:
                if tid not in T_dict:
                    continue
                if not _validate_stl_header(path):
                    continue
                valid_items.append((tid, jaw, path))
            if len(valid_items) == 0:
                continue

            # Optionally exclude cases with missing intermediate teeth
            if exclude_missing_teeth:
                tids = sorted(t[0] for t in valid_items)
                upper = [t for t in tids if 1 <= t <= 16]
                lower = [t for t in tids if 17 <= t <= 32]
                has_gap = False
                for seq in (upper, lower):
                    if len(seq) >= 2:
                        for a, b in zip(seq[:-1], seq[1:]):
                            if b - a > 1:
                                has_gap = True
                                break
                    if has_gap:
                        break
                if has_gap:
                    continue

            self.cases.append({
                "name": case_name,
                "case_dir": case_dir,
                "T_dict": T_dict,
                "stl_items": valid_items,
            })

        # Mesh cache (thread-safe lazy loading)
        self._mesh_cache = {}
        self._cache_lock = threading.Lock()

    def __getstate__(self):
        """Exclude unpicklable lock and per-process cache for multiprocessing."""
        state = self.__dict__.copy()
        state["_mesh_cache"] = {}
        state["_cache_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache_lock = threading.Lock()

    def __len__(self):
        return len(self.cases)

    def _load_mesh(self, stl_path):
        """Load a mesh, using cache if enabled."""
        if not self.cache_meshes:
            import trimesh
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value", category=RuntimeWarning)
                mesh = trimesh.load_mesh(stl_path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
            return mesh

        with self._cache_lock:
            if stl_path in self._mesh_cache:
                return self._mesh_cache[stl_path]

        import trimesh
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value", category=RuntimeWarning)
            mesh = trimesh.load_mesh(stl_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

        with self._cache_lock:
            self._mesh_cache[stl_path] = mesh
        return mesh

    def __getitem__(self, idx):
        from data.prepare import normalize_case, _farthest_point_sample

        case = self.cases[idx]
        rng = np.random.default_rng()

        tooth_ids = []
        points_in_list = []
        T_list = []

        for tid, jaw, stl_path in case["stl_items"]:
            mesh = self._load_mesh(stl_path)

            if self.sampling_method == "fps":
                # Oversample then FPS
                import trimesh as _trimesh
                oversample = max(self.n_points * 4, 8192)
                raw_pts, _ = _trimesh.sample.sample_surface(mesh, oversample)
                pts = _farthest_point_sample(raw_pts.astype(np.float32), self.n_points)
            else:
                pts, _ = mesh.sample(self.n_points, return_index=True) if hasattr(mesh, 'sample') else (
                    __import__('trimesh').sample.sample_surface(mesh, self.n_points)
                )
                if isinstance(pts, tuple):
                    pts = pts[0]
                idx_perm = rng.permutation(len(pts))[:self.n_points]
                pts = pts[idx_perm].astype(np.float32) if len(pts) >= self.n_points else pts.astype(np.float32)

            tooth_ids.append(tid)
            points_in_list.append(pts)
            T_list.append(case["T_dict"][tid])

        # Normalize
        points_in_list, center, scale = normalize_case(points_in_list)
        # Conjugate T: T_norm = A @ T_raw @ Ainv
        A = np.eye(4, dtype=np.float32)
        A[0, 0] = A[1, 1] = A[2, 2] = scale
        A[3, 0:3] = center
        Ainv = np.eye(4, dtype=np.float32)
        Ainv[0, 0] = Ainv[1, 1] = Ainv[2, 2] = 1.0 / scale
        Ainv[3, 0:3] = -center / scale
        T_list = [A @ T @ Ainv for T in T_list]

        out = {
            "case_name": case["name"],
            "points_in": torch.from_numpy(np.stack(points_in_list, axis=0)),
            "T_gt": torch.from_numpy(np.stack(T_list, axis=0).astype(np.float32)),
            "tooth_id": torch.tensor(tooth_ids, dtype=torch.int64),
            "scale": torch.tensor([scale], dtype=torch.float32),
            "center": torch.from_numpy(center),
        }

        # Online augmentation
        if self.augment:
            out["points_in"], out["T_gt"] = augment_points(
                out["points_in"], out["T_gt"],
                jitter=self.aug_cfg.get("jitter", 0.005),
                dropout=self.aug_cfg.get("dropout", 0.05),
                aug_rotation=self.aug_cfg.get("aug_rotation", 0.0),
                aug_scale=self.aug_cfg.get("aug_scale", 0.0),
                aug_translation=self.aug_cfg.get("aug_translation", 0.0),
            )

        return out


class CollateWrapper:
    """Picklable collate_fn wrapper (required for Windows multiprocessing)."""

    def __init__(self, max_teeth=32):
        self.max_teeth = max_teeth

    def __call__(self, batch):
        return collate_pad(batch, max_teeth=self.max_teeth)
